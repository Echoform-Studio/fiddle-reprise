// Daily refresh script. Runs twice daily via GitHub Actions cron.
//
// Always bumps the "Updated" timestamp in index.html.
// If ANTHROPIC_API_KEY is set as a repo secret, also scans for new fiddle
// sit-ins on recent DMB shows and logs findings to the Actions log.
// Manually patch index.html's DMB array based on the log output.

import { readFileSync, writeFileSync } from 'node:fs';

const file = 'index.html';

function bumpDate() {
  const today = new Date().toLocaleDateString('en-GB', {
    day:   'numeric',
    month: 'long',
    year:  'numeric',
    timeZone: 'America/Los_Angeles',
  });
  const html = readFileSync(file, 'utf8');
  const next = html.replace(
    /(<b>UPDATED<\/b>&nbsp; )\d{1,2} [A-Za-z]+ \d{4}/,
    `$1${today}`
  );
  if (next !== html) {
    writeFileSync(file, next);
    console.log(`Bumped date to ${today}.`);
  } else {
    console.log(`Date already ${today}.`);
  }
}

async function scanForNewSitIns() {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    console.log('No ANTHROPIC_API_KEY set. Skipping sit-in scan.');
    return;
  }

  let Anthropic;
  try {
    ({ default: Anthropic } = await import('@anthropic-ai/sdk'));
  } catch (e) {
    console.warn('@anthropic-ai/sdk not installed. Skipping scan.');
    return;
  }

  const client = new Anthropic({ apiKey });
  const today = new Date().toISOString().split('T')[0];

  try {
    const response = await client.messages.create({
      model: 'claude-opus-4-7',
      max_tokens: 4000,
      tools: [{ type: 'web_search_20250305', name: 'web_search', max_uses: 5 }],
      messages: [{
        role: 'user',
        content:
`Today is ${today}. Search for Dave Matthews Band shows played in the last 72 hours and report whether any fiddle/violin guest sat in.

For each DMB show in the last 72 hours, return:
- Date (YYYY-MM-DD)
- Venue + city
- Sit-in (yes/no)
- If yes: guest name + songs played

Tracked recurring fiddlers we already know about: Jake Renick Simpson (Lukas Nelson's band), Casey Driessen (solo/Bela Fleck collaborator), Tatiana Hargreaves, Jason Crosby (NY/NJ multi-instrumentalist).

Sources to check: setlist.fm/artists/dave-matthews-band, antsmarching.org, jambase, jambands.com, /r/davematthewsband.

If no DMB shows in the last 72 hours, say so. Output plain text findings, then a JSON block at the end summarizing any new sit-ins.`
      }],
    });

    const text = response.content
      .filter(b => b.type === 'text')
      .map(b => b.text)
      .join('\n');

    console.log('=== Sit-in scan ===');
    console.log(text);
    console.log('=== End scan ===');
  } catch (err) {
    console.error('Scan failed:', err.message);
  }
}

bumpDate();
await scanForNewSitIns();
