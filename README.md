# Fiddle (Reprise)

A probability index for fiddle sit-ins on the Dave Matthews Band 2026 tour.

## Stack

- Single static `index.html` (HTML + CSS + vanilla JS)
- Inter via Google Fonts
- No build step
- Deployed on Vercel

## Data sources

- DMB tour: `davematthewsband.com/tour`
- Lukas Nelson tour: `lukasnelson.com`, setlist.fm, Songkick
- Casey Driessen schedule: Bandsintown
- Sit-in outcomes: setlist.fm, antsmarching.org, dmbalmanac.com, jambase

## Daily refresh

A GitHub Action runs nightly to bump the "Updated" date. Full data refresh (re-scraping tour calendars + outcomes) is wired in `scripts/refresh.mjs` — fill in the data-fetch logic and add `ANTHROPIC_API_KEY` as a repo secret to enable.

## Local dev

```sh
python3 -m http.server 8765
```

Then open `http://localhost:8765`.
