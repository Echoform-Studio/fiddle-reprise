// Daily refresh script.
//
// v1 (current): bumps the "UPDATED" date in index.html so the page header
// reflects today. Tour data is updated manually for now.
//
// v2 (planned): uses @anthropic-ai/sdk to call Claude with web-search tools
// to re-scrape DMB/Lukas tour pages + new sit-ins from setlist.fm and
// dmbalmanac, then patches the DMB/LUKAS/DRIESSEN/TATIANA arrays inline.

import { readFileSync, writeFileSync } from 'node:fs';

const file = 'index.html';
const today = new Date().toLocaleDateString('en-GB', {
  day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC'
});

const html = readFileSync(file, 'utf8');
const next = html.replace(
  /(<b>UPDATED<\/b>&nbsp; )\d{1,2} [A-Za-z]+ \d{4}/,
  `$1${today}`
);

if (next === html) {
  console.log(`No change. Date already ${today}.`);
} else {
  writeFileSync(file, next);
  console.log(`Updated to ${today}.`);
}
