// Daily refresh. Runs twice daily via GitHub Actions cron.
//
// Three passes, in order:
//   1. SCORE     — any show that slipped into the past without an outcome gets
//                  researched, and the result written into index.html's DMB array.
//   2. DISCOVER  — newly announced DMB dates (and new Lukas Nelson dates, which
//                  drive Jake Simpson's availability signal) get added. Runs once
//                  a day, not twice; tour dates are announced in batches.
//   3. STAMP     — bump the "Updated" timestamp.
//
// Only high-confidence findings are written. Anything the model is unsure about
// is left alone and reported in the run summary for a human to settle, so a bad
// guess never silently becomes site data.

import { readFileSync, writeFileSync, appendFileSync } from 'node:fs';
import Anthropic from '@anthropic-ai/sdk';
import { z } from 'zod';
import { zodOutputFormat } from '@anthropic-ai/sdk/helpers/zod';

const FILE = 'index.html';

// Ceilings on how much one run will research. A backlog spills into later runs
// rather than letting a single run compound context — and cost — unbounded.
const MAX_SCORE_PER_RUN = 6;
const MAX_NEW_SHOWS = 12;
// Give up on a show after this many inconclusive attempts. Without it, a
// postponed show with no setlist is retried every run forever, occupying a
// batch slot and costing money while never resolving.
const MAX_TRIES = 3;
// Discovery also looks back this far. A show played but never listed — a
// weather makeup date, or anything announced while the job was broken — is
// otherwise invisible forever, since the scoring pass only ever revisits rows
// that already exist.
const LOOKBACK_DAYS = 90;

// allowed_domains rejects any host that blocks Anthropic's crawler (reddit and
// most ticketing sites do), and the API 400s on the whole request if one slips
// in — so keep this list to sources known to be fetchable.
const SOURCES = ['davematthewsband.com', 'dmbalmanac.com', 'antsmarching.org', 'setlist.fm', 'jambase.com'];
const TOUR_SOURCES = ['davematthewsband.com', 'lukasnelson.com', 'jambase.com', 'songkick.com'];

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

// ── reading the data arrays ──────────────────────────────────────────────────

const DATE_RE = /^\s*\{\s*date:'(\d{4}-\d{2}-\d{2})'/;

function readArray(lines, decl) {
  const start = lines.findIndex(l => l.includes(decl));
  if (start === -1) throw new Error(`Could not find "${decl}" in ${FILE}.`);

  const entries = [];
  let end = -1;
  for (let i = start + 1; i < lines.length; i++) {
    if (/^\];/.test(lines[i])) { end = i; break; }
    const m = lines[i].match(DATE_RE);
    if (!m) continue;
    entries.push({
      index: i,
      date: m[1],
      line: lines[i],
      venue: (lines[i].match(/venue:\s*(['"])(.*?)\1/) || [])[2] || '',
      city:  (lines[i].match(/city:\s*(['"])(.*?)\1/)  || [])[2] || '',
      done: /\b(outcome|unresolved|cancelled):/.test(lines[i]),
      tries: Number((lines[i].match(/\btries:(\d+)/) || [])[1] || 0),
    });
  }
  if (end === -1) throw new Error(`Could not find the end of "${decl}".`);
  if (!entries.length) throw new Error(`"${decl}" parsed to zero entries — refusing to continue.`);
  return { start, end, entries };
}

const esc = s => String(s).replace(/\\/g, '\\\\').replace(/'/g, "\\'");

// ── model plumbing ───────────────────────────────────────────────────────────

const spend = { input: 0, output: 0, searches: 0 };

async function ask({ prompt, schema, schemaName, domains, maxUses }) {
  const client = new Anthropic();
  const params = {
    model: 'claude-opus-5',
    max_tokens: 32000,
    thinking: { type: 'adaptive' },
    output_config: {
      // Do not lower this. At medium the model does ~2 searches for 6 shows and
      // takes the "skip if unsure" exit instead of looking; setlists that plainly
      // exist come back low-confidence. The saving is not worth the useless run.
      effort: 'high',
      format: zodOutputFormat(schema, schemaName),
    },
    tools: [{
      // response_inclusion:'excluded' keeps raw search blocks out of the billed
      // response; we only need the final JSON.
      type: 'web_search_20260318',
      name: 'web_search',
      response_inclusion: 'excluded',
      max_uses: maxUses,
      allowed_domains: domains,
    }],
    messages: [{ role: 'user', content: prompt }],
  };

  // Server tools can hand back `pause_turn` on long research runs; resume until done.
  const messages = [...params.messages];
  let final;
  for (let turn = 0; turn < 6; turn++) {
    const stream = client.messages.stream({ ...params, messages });
    final = await stream.finalMessage();
    const u = final.usage ?? {};
    spend.input += (u.input_tokens ?? 0) + (u.cache_read_input_tokens ?? 0) + (u.cache_creation_input_tokens ?? 0);
    spend.output += u.output_tokens ?? 0;
    spend.searches += u.server_tool_use?.web_search_requests ?? 0;
    if (final.stop_reason !== 'pause_turn') break;
    messages.push({ role: 'assistant', content: final.content });
  }

  if (final.stop_reason === 'refusal') {
    throw new Error(`Model declined: ${final.stop_details?.explanation ?? 'no explanation'}`);
  }

  const text = final.content.filter(b => b.type === 'text').map(b => b.text).join('');
  const raw = text.trim().startsWith('{') ? text : (text.match(/```(?:json)?\s*([\s\S]*?)```/) || [])[1];
  if (!raw) throw new Error(`No JSON in model response:\n${text.slice(0, 800)}`);
  return schema.parse(JSON.parse(raw));
}

// ── pass 1: score played shows ───────────────────────────────────────────────

const Findings = z.object({
  shows: z.array(z.object({
    date: z.string(),
    happened: z.boolean(),
    sitin: z.boolean(),
    player: z.string().nullable(),
    songs: z.number().int().nullable(),
    note: z.string().nullable(),
    confidence: z.enum(['high', 'medium', 'low']),
    source: z.string(),
  })),
});

const SCORE_PROMPT = `You are checking Dave Matthews Band setlists to determine whether a fiddle or violin guest sat in.

Tracked fiddlers: Jake Simpson (Lukas Nelson's band, the most frequent guest), Casey Driessen, Tatiana Hargreaves, Jason Crosby. A sit-in by any other fiddle/violin player counts too — record their name.

PRIMARY SOURCE: davematthewsband.com/setlists/ is the band's own setlist archive. It lists every show by date and marks guest musicians with footnote notation (e.g. "* Jake Simpson" against the songs they played). Check it first and treat it as authoritative. dmbalmanac.com is a thorough second opinion; antsmarching.org, setlist.fm and jambase are corroboration only.

Because the official archive marks guests affirmatively, a show listed there with no fiddle/violin guest footnote is a confident "no sit-in" — not an incomplete record. Do not hedge on that.

Rules that matter more than completeness:
- "happened" is false if the show was postponed, cancelled, or rescheduled to another date. A postponed show is NOT a "no sit-in" — it is not a data point at all. Set happened:false, sitin:false, and give the confidence you have in the postponement itself.
- Search for EACH DATE INDIVIDUALLY. A tour index or date-list page is not evidence about a specific night — open the setlist page for that exact date. Budget at least two searches per show before concluding anything.
- Confidence "high" ONLY when you found the actual setlist for that specific date and can see whether a fiddle guest appeared. A setlist that lists songs and shows no guest musician is a legitimate high-confidence "no sit-in" — that is the expected answer for most shows, and reporting it is the job, not a failure.
- Report "low" or "medium" only when you genuinely could not retrieve that date's setlist, the show did not happen, or sources disagree. Do not use low confidence as a shortcut past searching.
- Reserve "medium"/"low" for when you could not reach that date on the official archive and the other sources conflict — not for ordinary doubt about crowd-sourced completeness.
- "songs" is the number of songs the guest played, or null if unknown. "note" is a short detail (song names, festival context), or null.
- "source" is the URL you actually relied on.

Return one entry for every date listed, in the same order.

Shows to check:
`;

function outcomeFields(f) {
  if (!f.happened) return `, cancelled:true`;
  if (!f.sitin) return `, outcome:'no-sitin'`;
  let out = `, outcome:'sitin'`;
  if (f.player) out += `, sitinPlayer:'${esc(f.player)}'`;
  if (f.songs) out += `, sitinSongs:${f.songs}`;
  if (f.note) out += `, sitinNote:'${esc(f.note)}'`;
  return out;
}

// Record an inconclusive attempt, or retire the show once we've tried enough.
function markAttempt(line, tries) {
  const stripped = line.replace(/,\s*tries:\d+/, '');
  const m = stripped.match(/^(.*?)(\s*\},?\s*)$/s);
  if (!m) throw new Error(`Unexpected show line shape: ${line}`);
  const field = tries + 1 >= MAX_TRIES ? `, unresolved:true` : `, tries:${tries + 1}`;
  return m[1] + field + m[2];
}

function applyOutcome(line, f) {
  // An empty ticket link is dead weight once a show has been played.
  const cleaned = line.replace(/,\s*ticket:''/, '').replace(/,\s*tries:\d+/, '');
  const m = cleaned.match(/^(.*?)(\s*\},?\s*)$/s);
  if (!m) throw new Error(`Unexpected show line shape: ${line}`);
  return m[1] + outcomeFields(f) + m[2];
}

async function scorePass(lines, dmb, today) {
  const backlog = dmb.entries.filter(s => !s.done && s.date < today);
  const batch = backlog.slice(0, MAX_SCORE_PER_RUN);
  if (!batch.length) { console.log('SCORE: nothing to score.'); return 0; }

  console.log(`SCORE: ${backlog.length} awaiting an outcome; researching the oldest ${batch.length}.`);
  const { shows } = await ask({
    prompt: SCORE_PROMPT + batch.map(s => `- ${s.date} — ${s.venue}, ${s.city}`).join('\n'),
    schema: Findings, schemaName: 'findings',
    domains: SOURCES, maxUses: Math.min(24, batch.length * 3 + 3),
  });

  const byDate = new Map(shows.map(f => [f.date, f]));
  let wrote = 0;
  for (const show of batch) {
    const f = byDate.get(show.date);
    if (!f || f.confidence !== 'high') {
      const why = f ? `${f.confidence} confidence; best guess ${f.sitin ? `sit-in (${f.player})` : 'no sit-in'} (${f.source})` : 'no entry returned';
      lines[show.index] = markAttempt(show.line, show.tries);
      const retired = show.tries + 1 >= MAX_TRIES;
      console.log(`  ${show.date}  ${retired ? 'GIVING UP' : `SKIP (try ${show.tries + 1}/${MAX_TRIES})`} — ${why}`);
      wrote++;
      continue;
    }
    lines[show.index] = applyOutcome(show.line, f);
    console.log(`  ${show.date}  ${!f.happened ? 'DID NOT HAPPEN' : f.sitin ? `SIT-IN — ${f.player}` : 'no sit-in'} (${f.source})`);
    wrote++;
  }
  return wrote;
}

// ── pass 2: discover newly announced dates ───────────────────────────────────

const Discovered = z.object({
  dmb: z.array(z.object({
    date: z.string(),
    venue: z.string(),
    city: z.string(),
    ticket: z.string().nullable(),
    score: z.number().int().min(0).max(100),
    rationale: z.string(),
    confidence: z.enum(['high', 'medium', 'low']),
    source: z.string(),
  })),
  jake: z.array(z.object({
    date: z.string(),
    city: z.string(),
    confidence: z.enum(['high', 'medium', 'low']),
    source: z.string(),
  })),
});

function discoverPrompt(dmbDates, jakeDates, today, since) {
  return `Today is ${today}. You are maintaining a tracker that forecasts fiddle sit-ins at Dave Matthews Band shows.

Find tour dates that have been ANNOUNCED BUT ARE NOT YET IN THE LISTS BELOW, for two artists:

1. Dave Matthews Band — check davematthewsband.com/tour first, then jambase/songkick/ticketmaster to confirm.
2. Lukas Nelson — check lukasnelson.com. Jake Simpson plays fiddle in Lukas's band, so Lukas's calendar is how we infer Jake's availability. This matters as much as the DMB list.

Report any date from ${since} onward — INCLUDING shows that have already been played. A show that happened but is missing from the list below still matters: rescheduled and makeup dates are exactly what goes missing, and we need them. Do not report any date already listed. Do not invent dates: if the tour page shows nothing new, return empty arrays. That is a perfectly good answer.

For each NEW DMB date, assign a "score" from 0-100 using this rubric, and explain it in "rationale":
- Every show starts at a baseline of 18.
- +50 if DMB shares a festival/lineup bill with a tracked fiddler (Jake Simpson, Casey Driessen, Tatiana Hargreaves, Jason Crosby). This is the strongest signal.
- +30 if Jake Simpson looks available — i.e. the date falls in a gap in Lukas Nelson's tour calendar, or near a known Jake sit-in.
- +25 for regional signals: Casey Driessen and Tatiana Hargreaves (the Carolinas), Jason Crosby (NY/NJ/PA/CT tri-state).
- Cap at 100. A plain amphitheatre show with no signal should score close to 18.

Set confidence "high" only when the date appears on the artist's own official tour page or a ticketing page for that venue. Anything rumoured, tentative, or single-sourced from a fan post is "medium" or "low" and we will skip it.

"ticket" is the official ticket URL if you find one, else null. "source" is the URL you relied on.

DMB dates already tracked:
${dmbDates.join(', ')}

Lukas Nelson / Jake dates already tracked:
${jakeDates.join(', ')}`;
}

function dmbLine(s) {
  let out = `  { date:'${s.date}', venue:'${esc(s.venue)}', city:'${esc(s.city)}', score:${s.score}`;
  out += `, ticket:'${s.ticket ? esc(s.ticket) : ''}' },`;
  return out;
}

const jakeLine = s => `  { date:'${s.date}', city:'${esc(s.city)}' },`;

// Insert each new line so the array stays in date order.
function insertSorted(lines, arr, newItems, render) {
  const additions = [];
  for (const item of newItems) {
    const after = arr.entries.filter(e => e.date <= item.date).pop();
    additions.push({ at: after ? after.index : arr.start, line: render(item) });
  }
  // Descending so earlier insertions don't shift later targets.
  additions.sort((a, b) => b.at - a.at);
  for (const a of additions) lines.splice(a.at + 1, 0, a.line);
  return additions.length;
}

async function discoverPass(lines, dmb, jake, today) {
  const since = new Date(Date.parse(today + 'T00:00:00Z') - LOOKBACK_DAYS * 864e5)
    .toISOString().slice(0, 10);
  const found = await ask({
    prompt: discoverPrompt(
      dmb.entries.map(e => e.date),
      jake.entries.map(e => e.date),
      today,
      since,
    ),
    schema: Discovered, schemaName: 'discovered',
    domains: TOUR_SOURCES, maxUses: 20,
  });

  const known = new Set(dmb.entries.map(e => e.date + '|' + e.venue));
  const knownJake = new Set(jake.entries.map(e => e.date));

  const newDmb = found.dmb.filter(s => {
    if (s.confidence !== 'high') { console.log(`  DMB ${s.date} SKIP — ${s.confidence} confidence (${s.source})`); return false; }
    if (s.date < since) return false;
    if (known.has(s.date + '|' + s.venue)) return false;
    return true;
  }).slice(0, MAX_NEW_SHOWS);

  const newJake = found.jake.filter(s => {
    if (s.confidence !== 'high') return false;
    return s.date >= since && !knownJake.has(s.date);
  }).slice(0, MAX_NEW_SHOWS * 2);

  for (const s of newDmb) console.log(`  + DMB  ${s.date}  ${s.venue}, ${s.city} — score ${s.score} (${s.rationale})`);
  for (const s of newJake) console.log(`  + JAKE ${s.date}  ${s.city}`);

  // Jake first: it is the later array, so inserting there cannot shift DMB indices.
  const j = insertSorted(lines, jake, newJake, jakeLine);
  const dm = insertSorted(lines, dmb, newDmb, dmbLine);
  console.log(`DISCOVER: added ${dm} DMB date(s), ${j} Lukas/Jake date(s).`);
  return dm + j;
}

// ── main ─────────────────────────────────────────────────────────────────────

const today = todayPacific();
let lines = readFileSync(FILE, 'utf8').split('\n');
const dmb0 = readArray(lines, 'const DMB = [');
const jake0 = readArray(lines, 'const JAKE = [');

console.log(`Today (Pacific): ${today}. Tracking ${dmb0.entries.length} DMB shows, ${jake0.entries.length} Jake dates.`);

const needsScoring = dmb0.entries.some(s => !s.done && s.date < today);
const discover = process.env.DISCOVER === 'true';
if (!discover) console.log('DISCOVER: skipped (runs once daily).');

if ((needsScoring || discover) && !process.env.ANTHROPIC_API_KEY) {
  // Fail loudly. A silent skip is what let this rot for two months.
  throw new Error('Work is pending but ANTHROPIC_API_KEY is not set. Add it as a repository secret.');
}

let changed = 0;
if (needsScoring) changed += await scorePass(lines, dmb0, today);
// Re-read after scoring so discovery sees current line numbers.
if (discover) changed += await discoverPass(lines, readArray(lines, 'const DMB = ['), readArray(lines, 'const JAKE = ['), today);

if (changed) {
  const before = dmb0.entries.length;
  const after = readArray(lines, 'const DMB = [').entries.length;
  if (after < before) throw new Error(`DMB array shrank (${before} -> ${after}) — aborting without writing.`);
}

let html = lines.join('\n').replace(
  /(<b>UPDATED<\/b>&nbsp; )\d{1,2} [A-Za-z]+ \d{4}/,
  `$1${prettyDate(today)}`
);
writeFileSync(FILE, html);

if (spend.searches || spend.input) {
  // Opus 5: $5/M in, $25/M out. Web search: $10/1000.
  const cost = (spend.input / 1e6) * 5 + (spend.output / 1e6) * 25 + spend.searches * 0.01;
  console.log(`\nSpend: ${spend.input.toLocaleString()} in + ${spend.output.toLocaleString()} out tokens, ${spend.searches} searches ≈ $${cost.toFixed(3)}`);
} else {
  console.log('\nNo API calls made this run. Cost: $0.00');
}
console.log(`Updated stamp is ${prettyDate(today)}. ${changed} data change(s).`);

if (process.env.GITHUB_ENV) appendFileSync(process.env.GITHUB_ENV, `DATA_CHANGES=${changed}\n`);
