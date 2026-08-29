// Daily refresh. Runs twice daily via GitHub Actions cron.
//
// Two jobs, in order:
//   1. Score any show that has slipped into the past without an outcome —
//      research it, then write the result straight into index.html's DMB array.
//   2. Bump the "Updated" timestamp.
//
// Only high-confidence findings are written. Anything the model is unsure about
// is left unscored and reported in the run summary for a human to settle, so a
// bad guess never silently becomes site data.

import { readFileSync, writeFileSync } from 'node:fs';
import Anthropic from '@anthropic-ai/sdk';
import { z } from 'zod';
import { zodOutputFormat } from '@anthropic-ai/sdk/helpers/zod';

const FILE = 'index.html';

// Hard ceiling on how many shows one run will research. A backlog spills over
// into later runs rather than letting a single run compound context unbounded.
const MAX_PER_RUN = 6;

// ── dates ────────────────────────────────────────────────────────────────────

// The site pins "today" to Pacific; match it so the two never disagree.
function todayPacific() {
  return new Intl.DateTimeFormat('en-CA', {
    timeZone: 'America/Los_Angeles',
    year: 'numeric', month: '2-digit', day: '2-digit',
  }).format(new Date());
}

function prettyDate(iso) {
  return new Date(iso + 'T12:00:00Z').toLocaleDateString('en-GB', {
    day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC',
  });
}

// ── reading the DMB array ────────────────────────────────────────────────────

const SHOW_RE = /^\s*\{\s*date:'(\d{4}-\d{2}-\d{2})'/;

function parseShows(html) {
  const lines = html.split('\n');
  const start = lines.findIndex(l => l.includes('const DMB = ['));
  if (start === -1) throw new Error('Could not find the DMB array in index.html.');

  const shows = [];
  for (let i = start + 1; i < lines.length; i++) {
    if (/^\];/.test(lines[i])) break;
    const m = lines[i].match(SHOW_RE);
    if (!m) continue;
    shows.push({
      index: i,
      date: m[1],
      line: lines[i],
      venue: (lines[i].match(/venue:\s*(['"])(.*?)\1/) || [])[2] || '',
      city:  (lines[i].match(/city:\s*(['"])(.*?)\1/)  || [])[2] || '',
      scored: /\boutcome:/.test(lines[i]),
    });
  }
  if (!shows.length) throw new Error('DMB array parsed to zero shows — refusing to continue.');
  return shows;
}

// ── research ─────────────────────────────────────────────────────────────────

const Finding = z.object({
  date: z.string(),
  sitin: z.boolean(),
  player: z.string().nullable(),
  songs: z.number().int().nullable(),
  note: z.string().nullable(),
  confidence: z.enum(['high', 'medium', 'low']),
  source: z.string(),
});
const Findings = z.object({ shows: z.array(Finding) });

const PROMPT_HEADER = `You are checking Dave Matthews Band setlists to determine whether a fiddle or violin guest sat in.

Tracked fiddlers: Jake Simpson (Lukas Nelson's band, the most frequent guest), Casey Driessen, Tatiana Hargreaves, Jason Crosby. A sit-in by any other fiddle/violin player counts too — record their name.

Check setlist.fm/artists/dave-matthews-band, antsmarching.org, dmbalmanac.com, jambase, and the r/davematthewsband recaps.

Rules that matter more than completeness:
- Report confidence "high" ONLY when you found the actual setlist for that specific date and can see whether a fiddle guest appeared. A setlist showing no guest is a legitimate high-confidence "no sit-in".
- If you cannot find the setlist, if the show appears not to have happened, or if sources disagree, report confidence "low" or "medium" and we will skip it. Guessing is worse than skipping.
- "songs" is the number of songs the guest played, or null if unknown.
- "note" is a short detail (song names, festival context), or null.
- "source" is the URL you actually relied on.

Return one entry for every date listed below, in the same order.

Shows to check:
`;

async function research(pending) {
  const client = new Anthropic();
  const list = pending
    .map(s => `- ${s.date} — ${s.venue}, ${s.city}`)
    .join('\n');

  const params = {
    model: 'claude-opus-5',
    max_tokens: 32000,
    thinking: { type: 'adaptive' },
    output_config: {
      // Setlist lookup is retrieval, not hard reasoning; medium is plenty and
      // materially cheaper. The high-confidence gate below is what guards accuracy.
      effort: 'medium',
      format: zodOutputFormat(Findings, 'findings'),
    },
    tools: [{
      // 20260318 + response_inclusion:'excluded' keeps raw search blocks out of
      // the response we're billed for; we only need the final JSON.
      type: 'web_search_20260318',
      name: 'web_search',
      response_inclusion: 'excluded',
      max_uses: Math.min(24, pending.length * 3 + 3),
      allowed_domains: ['setlist.fm', 'antsmarching.org', 'dmbalmanac.com', 'jambase.com', 'reddit.com'],
    }],
    messages: [{ role: 'user', content: PROMPT_HEADER + list }],
  };

  // Server tools can hand back `pause_turn` on long research runs; resume until done.
  const messages = [...params.messages];
  const spend = { input: 0, output: 0, searches: 0 };
  let final;
  for (let turn = 0; turn < 6; turn++) {
    const stream = client.messages.stream({ ...params, messages });
    final = await stream.finalMessage();
    const u = final.usage ?? {};
    spend.input   += (u.input_tokens ?? 0) + (u.cache_read_input_tokens ?? 0) + (u.cache_creation_input_tokens ?? 0);
    spend.output  += u.output_tokens ?? 0;
    spend.searches += u.server_tool_use?.web_search_requests ?? 0;
    if (final.stop_reason !== 'pause_turn') break;
    messages.push({ role: 'assistant', content: final.content });
  }

  // Opus 5: $5/M in, $25/M out. Web search: $10/1000.
  const cost = (spend.input / 1e6) * 5 + (spend.output / 1e6) * 25 + spend.searches * 0.01;
  console.log(
    `Spend: ${spend.input.toLocaleString()} in + ${spend.output.toLocaleString()} out tokens, ` +
    `${spend.searches} searches \u2248 $${cost.toFixed(3)}`
  );

  if (final.stop_reason === 'refusal') {
    throw new Error(`Model declined the request: ${final.stop_details?.explanation ?? 'no explanation'}`);
  }

  const text = final.content.filter(b => b.type === 'text').map(b => b.text).join('');
  const raw = text.trim().startsWith('{')
    ? text
    : (text.match(/```(?:json)?\s*([\s\S]*?)```/) || [])[1];
  if (!raw) throw new Error(`No JSON found in model response:\n${text.slice(0, 800)}`);

  return Findings.parse(JSON.parse(raw)).shows;
}

// ── writing outcomes back ────────────────────────────────────────────────────

const esc = s => String(s).replace(/\\/g, '\\\\').replace(/'/g, "\\'");

function outcomeFields(f) {
  if (!f.sitin) return `, outcome:'no-sitin'`;
  let out = `, outcome:'sitin'`;
  if (f.player) out += `, sitinPlayer:'${esc(f.player)}'`;
  if (f.songs)  out += `, sitinSongs:${f.songs}`;
  if (f.note)   out += `, sitinNote:'${esc(f.note)}'`;
  return out;
}

function applyOutcome(line, f) {
  // An empty ticket link is dead weight once a show has been played.
  const cleaned = line.replace(/,\s*ticket:''/, '');
  const m = cleaned.match(/^(.*?)(\s*\},?\s*)$/s);
  if (!m) throw new Error(`Unexpected show line shape: ${line}`);
  return m[1] + outcomeFields(f) + m[2];
}

// ── main ─────────────────────────────────────────────────────────────────────

const today = todayPacific();
let html = readFileSync(FILE, 'utf8');
const shows = parseShows(html);
const backlog = shows.filter(s => !s.scored && s.date < today);
const pending = backlog.slice(0, MAX_PER_RUN);

console.log(`Today (Pacific): ${today}. ${shows.length} shows tracked, ${backlog.length} awaiting an outcome.`);
if (backlog.length > pending.length) {
  console.log(`Researching the oldest ${pending.length}; ${backlog.length - pending.length} will carry to the next run.`);
}

let wrote = 0;
if (pending.length) {
  if (!process.env.ANTHROPIC_API_KEY) {
    // Fail loudly. A silent skip is what let this rot for two months.
    throw new Error(
      `${pending.length} played show(s) need scoring but ANTHROPIC_API_KEY is not set. ` +
      `Add it as a repository secret (Settings → Secrets and variables → Actions).`
    );
  }

  const findings = await research(pending);
  const byDate = new Map(findings.map(f => [f.date, f]));
  const lines = html.split('\n');

  for (const show of pending) {
    const f = byDate.get(show.date);
    if (!f) { console.log(`  ${show.date}  SKIP — model returned no entry`); continue; }
    if (f.confidence !== 'high') {
      console.log(`  ${show.date}  SKIP — ${f.confidence} confidence. Best guess: ${f.sitin ? `sit-in (${f.player})` : 'no sit-in'} (${f.source})`);
      continue;
    }
    lines[show.index] = applyOutcome(show.line, f);
    console.log(`  ${show.date}  ${f.sitin ? `SIT-IN — ${f.player}` : 'no sit-in'} (${f.source})`);
    wrote++;
  }

  if (wrote) {
    const next = lines.join('\n');
    // Guard against a bad edit silently mangling the array.
    if (parseShows(next).length !== shows.length) {
      throw new Error('Show count changed after patching — aborting without writing.');
    }
    html = next;
  }
}

const stamped = html.replace(
  /(<b>UPDATED<\/b>&nbsp; )\d{1,2} [A-Za-z]+ \d{4}/,
  `$1${prettyDate(today)}`
);
if (stamped === html && !wrote) {
  console.log('Nothing to write.');
} else {
  writeFileSync(FILE, stamped);
  console.log(`\nWrote ${wrote} outcome(s); Updated stamp is ${prettyDate(today)}.`);
}

// Surface the count so the workflow can write an honest commit message.
if (process.env.GITHUB_ENV) {
  writeFileSync(process.env.GITHUB_ENV, `OUTCOMES_WRITTEN=${wrote}\n`, { flag: 'a' });
}
