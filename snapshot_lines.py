"""
snapshot_lines.py — lightweight, frequent line-movement snapshotter.
Separate from score_week.py (which runs 2x/week and does the heavy feature
computation) - this does ONE thing: pull current NFL spreads from The Odds
API and append a new row to line_history.csv, but ONLY when the line has
actually changed since the last logged value for that game. This keeps the
log to genuine movement events, not dozens of near-duplicate polls.

Requires an API key from the-odds-api.com (free tier: 500 credits/month).
Read from the ODDS_API_KEY environment variable (set as a GitHub Secret).

Usage: python3 snapshot_lines.py
"""
import pandas as pd
import requests
import os
import csv
from datetime import datetime, timezone

TEAM_CODE_MAP = {'SD': 'LAC', 'STL': 'LA', 'SL': 'LA', 'OAK': 'LV',
                  'ARZ': 'ARI', 'BLT': 'BAL', 'CLV': 'CLE', 'HST': 'HOU'}

FULL_NAME_TO_ABBR = {
    'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
    'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
    'Los Angeles Rams': 'LA', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
    'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
    'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
    'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
    'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'
}

LOG_PATH = 'line_history.csv'
LOG_COLUMNS = ['game_id', 'home_team', 'away_team', 'commence_time', 'snapshot_time',
               'bookmaker', 'home_spread', 'home_price', 'away_spread', 'away_price',
               'total_point', 'over_price', 'under_price']

# Preferred bookmaker order - use the first one available in the response,
# for consistency over time (avoids spurious "movement" from which books
# happen to respond on a given poll)
PRIMARY_BOOK = 'fanduel'  # FanDuel only - no fallback to other books, so it's
                          # never ambiguous which book a given number came from


def fetch_current_odds(api_key):
    url = 'https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds'
    params = {'apiKey': api_key, 'regions': 'us', 'markets': 'spreads,totals', 'oddsFormat': 'american'}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    remaining = resp.headers.get('x-requests-remaining')
    used = resp.headers.get('x-requests-used')
    print(f'API credits used this call cycle - remaining: {remaining}, used: {used}')
    return resp.json()


def extract_game_row(game, snapshot_time):
    home_full, away_full = game['home_team'], game['away_team']
    home_abbr = FULL_NAME_TO_ABBR.get(home_full)
    away_abbr = FULL_NAME_TO_ABBR.get(away_full)
    if not home_abbr or not away_abbr:
        return None

    chosen_book = next((bm for bm in game.get('bookmakers', []) if bm['key'] == PRIMARY_BOOK), None)
    if not chosen_book:
        return None  # FanDuel-only by design - no fallback to a different book,
                      # so it's never ambiguous which book a number came from

    row = {
        'game_id': f"{game['commence_time'][:10].replace('-','')}_{away_abbr}_{home_abbr}",
        'home_team': home_abbr, 'away_team': away_abbr,
        'commence_time': game['commence_time'], 'snapshot_time': snapshot_time,
        'bookmaker': chosen_book['key'],
        'home_spread': None, 'home_price': None, 'away_spread': None, 'away_price': None,
        'total_point': None, 'over_price': None, 'under_price': None,
    }

    spreads_market = next((m for m in chosen_book['markets'] if m['key'] == 'spreads'), None)
    if spreads_market:
        home_outcome = next((o for o in spreads_market['outcomes'] if o['name'] == home_full), None)
        away_outcome = next((o for o in spreads_market['outcomes'] if o['name'] == away_full), None)
        if home_outcome and away_outcome:
            row['home_spread'] = home_outcome['point']
            row['home_price'] = home_outcome['price']
            row['away_spread'] = away_outcome['point']
            row['away_price'] = away_outcome['price']

    totals_market = next((m for m in chosen_book['markets'] if m['key'] == 'totals'), None)
    if totals_market:
        over_outcome = next((o for o in totals_market['outcomes'] if o['name'] == 'Over'), None)
        under_outcome = next((o for o in totals_market['outcomes'] if o['name'] == 'Under'), None)
        if over_outcome and under_outcome:
            row['total_point'] = over_outcome['point']
            row['over_price'] = over_outcome['price']
            row['under_price'] = under_outcome['price']

    if row['home_spread'] is None and row['total_point'] is None:
        return None
    return row


def load_existing_log():
    try:
        return pd.read_csv(LOG_PATH)
    except FileNotFoundError:
        return pd.DataFrame(columns=LOG_COLUMNS)


def ensure_log_file_exists():
    """Guarantees LOG_PATH exists on disk (even as just a header row) so
    the workflow's `git add` step always has something valid to target -
    otherwise zero games/zero changes on a cold start leaves no file at
    all, and git fails with 'pathspec did not match any files'."""
    import os
    if not os.path.exists(LOG_PATH):
        pd.DataFrame(columns=LOG_COLUMNS).to_csv(LOG_PATH, index=False)
        print(f'Created empty {LOG_PATH} (header only) so git has something to track')


def main():
    api_key = os.environ.get('ODDS_API_KEY')
    if not api_key:
        print('ODDS_API_KEY environment variable not set. Exiting.')
        return

    snapshot_time = datetime.now(timezone.utc).isoformat()
    games = fetch_current_odds(api_key)
    print(f'Fetched odds for {len(games)} games')

    existing = load_existing_log()
    new_rows = []

    for game in games:
        row = extract_game_row(game, snapshot_time)
        if row is None:
            continue

        # only append if this is a genuinely new line (or the first ever
        # entry for this game) - compare against the most recent logged row.
        # Triggers on a change to EITHER spread OR total (independently
        # tracked series, but logged together since they come from one call)
        prior = existing[existing['game_id'] == row['game_id']]
        if not prior.empty:
            last = prior.sort_values('snapshot_time').iloc[-1]
            spread_unchanged = (last['home_spread'] == row['home_spread'] and
                                 last['home_price'] == row['home_price'])
            total_unchanged = (last['total_point'] == row['total_point'] and
                                last['over_price'] == row['over_price'])
            if spread_unchanged and total_unchanged:
                continue  # neither changed - skip

        new_rows.append(row)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(LOG_PATH, index=False)
        print(f'Logged {len(new_rows)} new line-movement entries')
        for r in new_rows:
            spread_str = f"{r['home_spread']} ({r['home_price']})" if r['home_spread'] is not None else 'n/a'
            total_str = f"{r['total_point']} O{r['over_price']}/U{r['under_price']}" if r['total_point'] is not None else 'n/a'
            print(f"  {r['away_team']} @ {r['home_team']}: spread {r['home_team']} {spread_str} | total {total_str}")
    else:
        print('No line movement detected - nothing new to log')
        ensure_log_file_exists()


if __name__ == '__main__':
    main()
