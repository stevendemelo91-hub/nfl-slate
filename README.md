# NFL Win-Confidence Live Pipeline — Setup

## What this is
A weekly-updating dashboard showing win-confidence ratings for every NFL
game, powered by the validated model from `nfl_model_spec.md` (63.1%
out-of-sample accuracy).

## Current status (read before relying on this)
**All 17 model categories are now live**, including road trip's full set
of sub-components (sandwich home game, last-before-trip, first-after-trip
— previously deferred as "negligible, half-weight") and the last-game-of-
season stakes logic. This represents the full learned weight of the
validated model.

**Still not live** (no data source — genuinely deferred, not a scoring
gap):
- Player trades

**One honest caveat**: the last-game-of-season logic was documented in the
original spec as "face-value, not independently backtested to the same
rigor as the rest of this category" — it's a reasonable, clearly-labeled
addition (see `nfl_model_spec.md`), not a rigorously validated one like
everything else.

Validated on a real week (2025 Week 10, partial-model version): 8/14
correct (57.1%) — expect the current, complete version to track closer to
the full validated 63.1%.

## One-time setup (~15 minutes)

1. **Create a free GitHub account** if you don't have one.
2. **Create a new repository** (public or private — Actions works for both
   on the free tier).
3. **Upload the whole `live_pipeline/` folder**, preserving structure —
   everything in it is needed:
   - `score_week.py`, `power_ratings.py`, `snapshot_lines.py`
   - `production_model.pkl`
   - `power_ratings.csv`, `team_hfa.csv`, `qb_overrides.csv`,
     `elite_qb_2025.csv`, `ratings_and_hfa_combined.csv`
   - `docs/index.html`, `docs/table.html`
   - `.github/workflows/weekly_score.yml`,
     `.github/workflows/line_snapshot.yml`
   - `docs/data/` (optional — starter files from testing; the first real
     run will populate this properly regardless)
4. **Enable GitHub Pages**: repo Settings → Pages → Source: "Deploy from a
   branch" → Branch: `main`, folder: `/docs` → Save. The card view will be
   live at `https://<your-username>.github.io/<repo-name>/`, chart view at
   the same URL + `table.html`.
5. **Add the odds API secret** (needed for line movement — see that section
   below for getting the key itself): repo Settings → Secrets and
   variables → Actions → New repository secret → name it `ODDS_API_KEY`.
6. **Check the workflow schedule** in `weekly_score.yml` — it's set to run
   Monday and Tuesday mornings (8am ET). The season-start date used to guess
   the current week is approximate; update the `season_start` line each
   September, or just trigger runs manually (see below) if it drifts.
7. **First run**: go to the Actions tab → "Weekly NFL Scoring" → "Run
   workflow" → enter the season and week manually to test it end-to-end.
   Do the same for "Line Movement Snapshot" to confirm the odds API key
   works. Check both runs' logs for errors before trusting the schedule.

## Running manually / locally
```
cd live_pipeline
pip install pandas numpy scikit-learn requests
python3 score_week.py --season 2026 --week 5
```
This writes directly to `docs/data/2026_5.json` and updates
`docs/data/manifest.json` — no manual copy step needed.

To preview the dashboard locally (fetch() won't work by just double-clicking
the HTML file — browsers block local file fetches):
```
cd docs
python3 -m http.server 8000
```
Then open `http://localhost:8000` (card view) or
`http://localhost:8000/table.html` (chart view) in a browser.

## Weekly cadence (as designed)
- **Monday morning**: captures data through Sunday, excludes that week's
  Monday Night Football participants (game hasn't happened yet)
- **Tuesday morning**: re-runs with the completed MNF result included
- Each run also scores **next week** in advance (`continue-on-error`, since
  next week's schedule may not exist yet late in a season), so it's never
  empty when you check ahead

## Line movement (new)
Pulls current NFL spreads AND totals from **The Odds API** (free tier, 500
credits/month, no card required) every 4 hours around the clock via a
separate GitHub Actions workflow (`line_snapshot.yml`), building a
persistent movement log in `line_history.csv`. Only logs a new row when
the spread or total actually changes — not on every poll — so the history
stays to genuine movement events. Spread and total each get their own
independently-tracked, deduplicated history (shown as separate SPR/TOT
sections in the chart view).

**One-time setup:**
1. Sign up for a free key at [the-odds-api.com](https://the-odds-api.com/) —
   no credit card needed.
2. In your GitHub repo: Settings → Secrets and variables → Actions → New
   repository secret → name it `ODDS_API_KEY`, paste your key. I never see
   this value — it's referenced by name in the workflow.
3. The `line_snapshot.yml` workflow runs automatically every 4 hours. You
   can also trigger it manually from the Actions tab to test it.

**Credit budget**: pulling both spreads and totals roughly doubles the
per-call cost, so the cadence was halved from every 2 hours to every 4 to
stay within budget: 6 runs/day × 30 days × ~2 credits ≈ 360 credits/month
against the 500/month free allowance. Watch the `x-requests-remaining`
value the script prints each run to see real consumption before adjusting
either direction.

Dashboard shows "Current Line" (most recent snapshot) and "Line History"
(all prior distinct values, most recent first) in the table view, always
relative to the home team.

## QB status (now automated)
QB status is pulled live from nflverse's depth-chart data (updated frequently,
same-day accuracy as of this build) and cross-referenced against a prior-
season elite-QB list (top 12 by EPA/play, matching the validated backtest
methodology). This is now wired into the win-confidence model automatically
— no more manual placeholder.

**What's automatic:** each team's current depth-chart QB1, and whether that
player rates as elite-tier.

**What still needs your input:** the depth chart doesn't reliably capture
same-week injury news the moment it happens. Edit `qb_overrides.csv` to
tell the scorer when you know a backup or rookie is starting:
```
team,backup_starting,rookie_starting,normal_starter_override
SEA,yes,no,
```
Leave `normal_starter_override` blank unless the depth chart has *already*
updated to show the backup as QB1 (in which case, fill in the name of the
elite starter who's actually out, so the elite-starter-out penalty still
applies correctly).

Each game's dashboard card shows the depth-chart starter and any BACKUP/
ROOKIE/elite-tier flags currently active, so you can sanity-check before
trusting the rating.

## Non-QB injury surfacing (new)
Pulls the current season's injury report (nflverse) and surfaces any
Out/Doubtful non-QB player per team, right below the QB check section on
each game's card (card view only — not yet in the chart/table view). **This
is surfacing only — it does not feed into the model's rating or the power
rating.** There's no validated weight for skill-position injuries the way
there is for QB status, so rather than guess at one, this just gives you
fast context to factor in yourself (via your own guess, or when
cross-referencing handicapper picks).

Like the depth-chart data, this comes from nflverse and won't have real
entries until the season is underway and injury reports start getting
published weekly.

## Next steps
1. Real end-to-end test on GitHub's actual infrastructure — everything so
   far has been validated locally; the live Actions environment hasn't
   been exercised for real yet.
2. Player trades (no data source currently identified).
