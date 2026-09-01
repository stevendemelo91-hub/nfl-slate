"""
score_week.py — Live weekly NFL win-confidence scorer.
Usage: python3 score_week.py --season 2026 --week 5
"""
import pandas as pd
import numpy as np
import pickle, json, argparse, warnings, os
warnings.filterwarnings('ignore')

TEAM_CODE_MAP = {'SD': 'LAC', 'STL': 'LA', 'SL': 'LA', 'OAK': 'LV',
                  'ARZ': 'ARI', 'BLT': 'BAL', 'CLV': 'CLE', 'HST': 'HOU'}
DIVISIONS = {
    'AFC_EAST': ['BUF', 'MIA', 'NE', 'NYJ'], 'AFC_NORTH': ['BAL', 'CIN', 'CLE', 'PIT'],
    'AFC_SOUTH': ['HOU', 'IND', 'JAX', 'TEN'], 'AFC_WEST': ['DEN', 'KC', 'LV', 'LAC'],
    'NFC_EAST': ['DAL', 'NYG', 'PHI', 'WAS'], 'NFC_NORTH': ['CHI', 'DET', 'GB', 'MIN'],
    'NFC_SOUTH': ['ATL', 'CAR', 'NO', 'TB'], 'NFC_WEST': ['ARI', 'LA', 'SF', 'SEA'],
}
ALL_TEAMS = [t for teams in DIVISIONS.values() for t in teams]
TEAM_TO_CONF = {t: d.split('_')[0] for d, teams in DIVISIONS.items() for t in teams}


def normalize_team_col(s):
    return s.map(lambda x: TEAM_CODE_MAP.get(x, x))


def fetch_games(seasons):
    g = pd.read_csv('https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv')
    g = g[g['season'].isin(seasons)].copy()
    for c in ['home_team', 'away_team']:
        g[c] = normalize_team_col(g[c])
    return g

def fetch_pbp(season):
    try:
        return pd.read_csv(f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz',
                            compression='gzip', low_memory=False)
    except Exception:
        return pd.DataFrame()


def build_offense_stats(pbp):
    if pbp.empty:
        return pd.DataFrame()
    scrimmage = pbp[(pbp['play_type'].isin(['pass', 'run'])) & pbp['posteam'].notna()].copy()
    grp = scrimmage.groupby(['game_id', 'season', 'week', 'posteam'])

    def third_stats(df):
        third = df[df['down'] == 3]
        return pd.Series({'third_down_conv': third['third_down_converted'].fillna(0).sum(),
                           'third_down_failed': third['third_down_failed'].fillna(0).sum()})

    off = grp.agg(plays=('epa', 'count'), yards=('yards_gained', 'sum'),
                  pass_yards=('yards_gained', lambda x: x[scrimmage.loc[x.index, 'pass'] == 1].sum()),
                  rush_yards=('yards_gained', lambda x: x[scrimmage.loc[x.index, 'rush'] == 1].sum()),
                  interceptions=('interception', 'sum'), fumbles_lost=('fumble_lost', 'sum')).reset_index()
    td = grp.apply(third_stats, include_groups=False).reset_index()
    off = off.merge(td, on=['game_id', 'season', 'week', 'posteam'], how='left')
    off['giveaways'] = off['interceptions'].fillna(0) + off['fumbles_lost'].fillna(0)
    return off.rename(columns={'posteam': 'team'})


def build_full_game_table(games, offense):
    g = games[['game_id', 'season', 'week', 'game_type', 'home_team', 'away_team', 'home_score',
               'away_score', 'weekday', 'away_rest', 'home_rest']].copy()
    off = offense.merge(g, on=['game_id', 'season', 'week'], how='left')
    off['opponent'] = off.apply(lambda r: r['home_team'] if r['team'] == r['away_team'] else r['away_team'], axis=1)
    off['is_home'] = off['team'] == off['home_team']
    off['team_score'] = off.apply(lambda r: r['home_score'] if r['is_home'] else r['away_score'], axis=1)
    off['opp_score'] = off.apply(lambda r: r['away_score'] if r['is_home'] else r['home_score'], axis=1)
    off['margin'] = off['team_score'] - off['opp_score']
    off['won'] = off['margin'] > 0
    off['rest_days'] = off.apply(lambda r: r['home_rest'] if r['is_home'] else r['away_rest'], axis=1)

    opp_stats = offense.rename(columns={'team': 'opponent', 'plays': 'opp_plays', 'yards': 'yards_allowed',
        'third_down_conv': 'opp_third_conv', 'third_down_failed': 'opp_third_failed',
        'giveaways': 'takeaways'})[['game_id', 'opponent', 'opp_plays', 'yards_allowed', 'takeaways']]
    return off.merge(opp_stats, on=['game_id', 'opponent'], how='left')


def compute_schedule_situational(team, target_week, target_is_home, games_all_season):
    """Road trip position, and schedule milestones (home opener, last game
    before bye) - all derivable from the team's full-season schedule (past
    results + future schedule, both known in advance). Mirrors the
    validated historical logic and weights exactly."""
    team_games = games_all_season[
        (games_all_season['home_team'] == team) | (games_all_season['away_team'] == team)
    ].copy()
    team_games = team_games.sort_values('week')
    team_games['is_home_game'] = team_games['home_team'] == team

    played = team_games[team_games['week'] < target_week].copy()
    if not played.empty and 'home_score' in played.columns:
        played['won'] = played.apply(
            lambda r: (r['home_score'] > r['away_score']) if r['is_home_game'] else (r['away_score'] > r['home_score']),
            axis=1)

    # Count consecutive road games IMMEDIATELY before the target week
    consecutive_prior_road = 0
    road_wins, road_losses = 0, 0
    if not played.empty:
        weeks_played_desc = played.sort_values('week', ascending=False)
        wl = []
        for _, row in weeks_played_desc.iterrows():
            if not row['is_home_game']:
                consecutive_prior_road += 1
                wl.append(row.get('won', None))
            else:
                break
        for w in reversed(wl):
            if w is True:
                road_wins += 1
            elif w is False:
                road_losses += 1

    # This game's position in the trip (only meaningful if target game is away)
    road_trip_position = (consecutive_prior_road + 1) if not target_is_home else 0

    # Home opener: this game is home AND no prior home games played this season
    home_opener = False
    if target_is_home:
        home_opener = played.empty or not played['is_home_game'].any()

    # Bye week detection from FULL schedule (past + future)
    scheduled_weeks = set(team_games['week'].tolist())
    if scheduled_weeks:
        full_range = range(min(scheduled_weeks), max(scheduled_weeks) + 1)
        bye_weeks = [w for w in full_range if w not in scheduled_weeks]
    else:
        bye_weeks = []
    last_before_bye = (target_week + 1) in bye_weeks

    # Future schedule lookups (for sandwich/before-trip/after-trip)
    future = team_games[team_games['week'] > target_week].sort_values('week')
    next_game = future.iloc[0] if not future.empty else None
    next_next_game = future.iloc[1] if len(future) > 1 else None
    next_is_away = next_game is not None and not next_game['is_home_game']
    next_next_is_away = next_next_game is not None and not next_next_game['is_home_game']

    # Sandwich home game: home this week, away last week, away next week
    prior_is_away = False
    if not played.empty:
        last_played = played.sort_values('week').iloc[-1]
        prior_is_away = not last_played['is_home_game']
    is_sandwich_home = target_is_home and prior_is_away and next_is_away

    # Last game before a 2+ game road trip starts next week (this week's
    # own home/away status doesn't matter - it's about what's coming)
    is_last_before_trip = next_is_away and next_next_is_away

    # First home game after a 2+ game road trip that just concluded
    is_first_after_trip = target_is_home and consecutive_prior_road >= 2

    # Last game of the team's regular season (max scheduled week)
    is_last_game_of_season = bool(scheduled_weeks) and target_week == max(scheduled_weeks)

    return {
        'road_trip_position': road_trip_position,
        'road_trip_wins': road_wins,
        'road_trip_losses': road_losses,
        'is_home_opener': home_opener,
        'is_last_before_bye': last_before_bye,
        'is_sandwich_home': is_sandwich_home,
        'is_last_before_trip': is_last_before_trip,
        'is_first_after_trip': is_first_after_trip,
        'is_last_game_of_season': is_last_game_of_season,
    }


def compute_coaching_change(team, season, games_all):
    """New HC flag: compares the team's current coach (from the most recent
    scheduled/played game this season) against last season's coach. Also
    checks for a mid-season change (different coach across this season's
    already-played games)."""
    this_season = games_all[games_all['season'] == season]
    prior_season = games_all[games_all['season'] == season - 1]

    def team_coach_from(df, team):
        home_rows = df[df['home_team'] == team][['week', 'home_coach']].rename(columns={'home_coach': 'coach'})
        away_rows = df[df['away_team'] == team][['week', 'away_coach']].rename(columns={'away_coach': 'coach'})
        combined = pd.concat([home_rows, away_rows]).dropna(subset=['coach']).sort_values('week')
        return combined

    cur = team_coach_from(this_season, team)
    prior = team_coach_from(prior_season, team)

    if cur.empty or prior.empty:
        return False

    current_coach = cur.iloc[-1]['coach']
    prior_coach = prior.iloc[-1]['coach']
    offseason_change = current_coach != prior_coach

    # mid-season change: coach differs across this season's own games so far
    same_season_change = cur['coach'].nunique() > 1

    return bool(offseason_change or same_season_change)


def compute_stakes(team, season, target_week, games_all):
    """Simplified clinched/eliminated/trying-to-stay-alive logic, matching
    the validated historical approach: only applies from week 15 onward,
    approximate (win% ranking, no exact tiebreakers)."""
    if target_week < 15:
        return {'not_playing_for_anything': False, 'trying_to_stay_alive': False,
                'clinched_playoff_berth': False, 'conf_rank': None}

    reg = games_all[(games_all['season'] == season) & (games_all['game_type'] == 'REG') &
                     (games_all['week'] < target_week)].copy()
    if reg.empty or reg['home_score'].isna().all():
        return {'not_playing_for_anything': False, 'trying_to_stay_alive': False,
                'clinched_playoff_berth': False, 'conf_rank': None}

    reg = reg.dropna(subset=['home_score', 'away_score'])
    long_rows = pd.concat([
        reg[['week', 'home_team', 'home_score', 'away_score']].rename(
            columns={'home_team': 'team', 'home_score': 'team_score', 'away_score': 'opp_score'}),
        reg[['week', 'away_team', 'home_score', 'away_score']].rename(
            columns={'away_team': 'team', 'away_score': 'team_score', 'home_score': 'opp_score'}),
    ])
    long_rows['won'] = long_rows['team_score'] > long_rows['opp_score']
    standings = long_rows.groupby('team').agg(wins=('won', 'sum'), games=('won', 'count')).reset_index()
    standings['conference'] = standings['team'].map(TEAM_TO_CONF)
    standings['win_pct'] = standings['wins'] / standings['games']
    standings['conf_rank'] = standings.groupby('conference')['win_pct'].rank(ascending=False, method='first')

    season_games = 17 if season >= 2021 else 16
    standings['remaining'] = season_games - standings['games']
    standings['max_wins'] = standings['wins'] + standings['remaining']
    playoff_cutoff = 7 if season >= 2020 else 6

    team_row = standings[standings['team'] == team]
    if team_row.empty:
        return {'not_playing_for_anything': False, 'trying_to_stay_alive': False,
                'clinched_playoff_berth': False, 'conf_rank': None}
    team_row = team_row.iloc[0]
    conf = team_row['conference']
    conf_standings = standings[standings['conference'] == conf]

    cutoff_team = conf_standings[conf_standings['conf_rank'] == playoff_cutoff]
    first_out = conf_standings[conf_standings['conf_rank'] == playoff_cutoff + 1]
    cutoff_wins = cutoff_team.iloc[0]['wins'] if not cutoff_team.empty else None
    first_out_max = first_out.iloc[0]['max_wins'] if not first_out.empty else None

    mathematically_eliminated = cutoff_wins is not None and team_row['max_wins'] < cutoff_wins
    clinched = (team_row['conf_rank'] <= playoff_cutoff) and (first_out_max is not None) and (team_row['wins'] > first_out_max)
    trying_to_stay_alive = (not mathematically_eliminated) and (not clinched) and (team_row['conf_rank'] > playoff_cutoff)

    return {
        'not_playing_for_anything': bool(mathematically_eliminated),
        'trying_to_stay_alive': bool(trying_to_stay_alive),
        'clinched_playoff_berth': bool(clinched),
        'conf_rank': float(team_row['conf_rank']),
    }


def load_line_history(path='line_history.csv'):
    """Loads the persistent line-movement log built by snapshot_lines.py.
    Matches by (home_team, away_team) rather than game_id, since the odds
    API and nflverse use different ID schemes."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()


def short_date(iso_or_timestamp):
    try:
        dt = pd.to_datetime(iso_or_timestamp, utc=True)
        dt_et = dt.tz_convert('America/New_York')
        return f"{dt_et.month}/{dt_et.day}"
    except Exception:
        return '?'


def get_line_movement(home_team, away_team, line_log):
    """Returns a dict with separate 'spread' and 'total' movement series,
    each as (current_str, [history_strs]). Since the log stores both
    together per snapshot (one API call covers both markets), each series
    is independently deduplicated against ITS OWN consecutive values - a
    snapshot where only the total changed doesn't create a spurious
    "spread movement" entry, and vice versa."""
    empty = {'current': None, 'history': []}
    if line_log.empty:
        return {'spread': dict(empty), 'total': dict(empty)}
    matches = line_log[(line_log['home_team'] == home_team) & (line_log['away_team'] == away_team)].copy()
    if matches.empty:
        return {'spread': dict(empty), 'total': dict(empty)}
    matches = matches.sort_values('snapshot_time')

    def build_series(rows, value_cols, fmt_fn):
        distinct = []
        last_key = None
        for _, row in rows.iterrows():
            key = tuple(row[c] for c in value_cols)
            if any(pd.isna(v) for v in key):
                continue
            if key != last_key:
                distinct.append(row)
                last_key = key
        if not distinct:
            return dict(empty)
        current_str = fmt_fn(distinct[-1])
        history_strs = [fmt_fn(r) for r in reversed(distinct[:-1])]
        return {'current': current_str, 'history': history_strs}

    def fmt_spread(row):
        spread = row['home_spread']
        spread_str = f"{'+' if spread > 0 else ''}{spread}"
        price = row['home_price']
        price_str = f"{'+' if price > 0 else ''}{price}"
        return f"{spread_str} {price_str} ({short_date(row['snapshot_time'])})"

    def fmt_total(row):
        total = row['total_point']
        over_str = f"{'+' if row['over_price'] > 0 else ''}{row['over_price']}"
        return f"O/U {total} ({over_str}) ({short_date(row['snapshot_time'])})"

    return {
        'spread': build_series(matches, ['home_spread', 'home_price'], fmt_spread),
        'total': build_series(matches, ['total_point', 'over_price'], fmt_total),
    }


def prior_year_weight(week):
    return max(0.20, 1.0 - (week - 1) * 0.10)


def compute_team_inputs(team, cur_full, prior_full, target_week):
    w = prior_year_weight(target_week)
    reg_cur = cur_full[(cur_full['game_type'] == 'REG') & (cur_full['week'] < target_week) & (cur_full['team'] == team)] if not cur_full.empty else pd.DataFrame()
    reg_prior = prior_full[(prior_full['game_type'] == 'REG') & (prior_full['team'] == team)] if not prior_full.empty else pd.DataFrame()

    def stats(df):
        if len(df) == 0:
            return {k: np.nan for k in ['pass_ypg', 'rush_ypg', 'ypp_off', 'ypp_def', 'third_down_pct', 'turnover_margin']}
        return {
            'pass_ypg': df['pass_yards'].sum() / len(df),
            'rush_ypg': df['rush_yards'].sum() / len(df),
            'ypp_off': df['yards'].sum() / df['plays'].sum() if df['plays'].sum() else np.nan,
            'ypp_def': df['yards_allowed'].sum() / df['opp_plays'].sum() if df['opp_plays'].sum() else np.nan,
            'third_down_pct': df['third_down_conv'].sum() / (df['third_down_conv'].sum() + df['third_down_failed'].sum()) if (df['third_down_conv'].sum() + df['third_down_failed'].sum()) else np.nan,
            'turnover_margin': df['takeaways'].sum() - df['giveaways'].sum(),
        }

    cur_s, prior_s = stats(reg_cur), stats(reg_prior)

    def blend(k):
        c, p = cur_s[k], prior_s[k]
        if pd.isna(c) and pd.isna(p):
            return np.nan
        if pd.isna(c):
            return p
        if pd.isna(p):
            return c
        return w * p + (1 - w) * c

    return {k: blend(k) for k in cur_s}


def compute_streak(team_games):
    if len(team_games) == 0:
        return 0
    results = team_games.sort_values('week')['won'].tolist()
    streak = 0
    for won in results:
        streak = (streak + 1 if streak > 0 else 1) if won else (streak - 1 if streak < 0 else -1)
    return streak


def streak_to_value(streak):
    if abs(streak) <= 1:
        return 0.0
    return -0.5 * (1 if streak > 0 else -1)


def rank_comparison(rank_a, rank_b):
    if pd.isna(rank_a) or pd.isna(rank_b):
        return 0.0
    if (11 <= rank_a <= 19) and (11 <= rank_b <= 19):
        return 0.0
    gap = rank_b - rank_a
    if abs(gap) <= 1:
        return 0.0
    return 0.5 if gap > 0 else -0.5


def straight(hv, av, lower_better=False):
    if pd.isna(hv) or pd.isna(av) or hv == av:
        return 0.0
    if lower_better:
        return 1.0 if hv < av else -1.0
    return 1.0 if hv > av else -1.0


def load_power_ratings(path='power_ratings.csv'):
    """Optional — returns {} if the file doesn't exist yet."""
    try:
        df = pd.read_csv(path)
        return dict(zip(df['team'], df['rating']))
    except FileNotFoundError:
        return {}


DEFAULT_HFA = 1.89  # recency-weighted league baseline


def load_team_hfa(path='team_hfa.csv'):
    try:
        df = pd.read_csv(path)
        return dict(zip(df['team'], df['final_team_hfa']))
    except FileNotFoundError:
        return {}


# Dynamic, month-specific overrides for teams where the seasonal split is
# real and well-supported (see nfl_model_spec.md). Miami: home edge is
# actually WEAKER in September (1.76, near league-average) than the rest of
# the season (4.42) — opposite of the "September heat helps Miami" theory,
# confirmed via recency-weighted backtest.
DYNAMIC_TEAM_HFA = {
    'MIA': {9: 2.5, 'default': 1.92},
}


def get_team_hfa(home_team, game_month, team_hfa_table):
    if home_team in DYNAMIC_TEAM_HFA:
        rule = DYNAMIC_TEAM_HFA[home_team]
        return rule.get(game_month, rule['default'])
    return team_hfa_table.get(home_team, DEFAULT_HFA)


def implied_power_spread(home, away, ratings, team_hfa, game_month=None):
    if home not in ratings or away not in ratings:
        return None
    applied_hfa = get_team_hfa(home, game_month, team_hfa)
    return round((ratings[away] - ratings[home]) + applied_hfa, 2)


ELITE_QB_MIN_DROPBACKS = 150
ELITE_QB_TOP_N = 12


def to_pbp_format(full_name):
    if not isinstance(full_name, str) or len(full_name.split()) < 2:
        return None
    parts = full_name.split()
    return f"{parts[0][0]}.{parts[-1]}"


def fetch_elite_qb_list(prior_season):
    """Top-12 QBs by EPA/play in the prior completed season (150+ dropbacks),
    same methodology validated in the historical backtest."""
    try:
        pbp = pd.read_csv(
            f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{prior_season}.csv.gz',
            compression='gzip', low_memory=False,
            usecols=['season', 'posteam', 'passer_player_name', 'qb_epa', 'play_type', 'pass'])
    except Exception:
        return set()
    dropback = pbp[pbp['pass'] == 1].dropna(subset=['passer_player_name', 'qb_epa'])
    agg = dropback.groupby('passer_player_name').agg(
        dropbacks=('qb_epa', 'count'), epa_per_play=('qb_epa', 'mean')).reset_index()
    qualified = agg[agg['dropbacks'] >= ELITE_QB_MIN_DROPBACKS].copy()
    qualified['rank'] = qualified['epa_per_play'].rank(ascending=False, method='min')
    elite = qualified[qualified['rank'] <= ELITE_QB_TOP_N]
    return set(elite['passer_player_name'])


def fetch_injury_report(season, week):
    """Current-season injury report for the target week. Returns empty
    DataFrame gracefully if not published yet (early season, or the file
    doesn't exist for this season) - matches the graceful-fallback pattern
    used elsewhere (depth charts, elite QB list, etc)."""
    try:
        inj = pd.read_csv(f'https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.csv',
                           low_memory=False)
    except Exception:
        return pd.DataFrame()
    inj = inj[inj['week'] == week]
    inj = inj[inj['report_status'].isin(['Out', 'Doubtful'])]
    inj = inj[inj['position'] != 'QB']  # QB handled separately by compute_qb_coefficient
    return inj


def get_injury_candidates(team, injury_df, injury_history):
    """Surfacing only - no coefficient/scoring impact, since there's no
    validated weight for non-QB injuries. Just gives the user fast context
    to factor into their own guess/handicapper cross-referencing, matching
    the hybrid design's 'surface candidates, human judges impact' pattern.
    Includes the day-by-day trend (from injury_history.csv) when available,
    e.g. Wed: DNP -> Thu: Limited -> Fri: Full, not just the final status."""
    if injury_df.empty:
        return []
    team_injuries = injury_df[injury_df['team'] == team]
    candidates = []
    for _, row in team_injuries.iterrows():
        trend = []
        if not injury_history.empty and pd.notna(row.get('gsis_id')):
            player_log = injury_history[
                (injury_history['gsis_id'] == row['gsis_id']) & (injury_history['week'] == row['week'])
            ].sort_values('snapshot_date')
            trend = [
                {'date': r['snapshot_date'], 'report_status': r.get('report_status'), 'practice_status': r.get('practice_status')}
                for _, r in player_log.iterrows()
            ]
        candidates.append({
            'name': row['full_name'], 'position': row['position'], 'status': row['report_status'],
            'trend': trend,
        })
    return candidates


def fetch_current_starters(season):
    """Most recent depth-chart snapshot's QB1 per team, plus rookie status."""
    try:
        dc = pd.read_csv(f'https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_{season}.csv',
                          low_memory=False)
    except Exception:
        return {}
    qb_dc = dc[dc['pos_abb'] == 'QB'].copy()
    if qb_dc.empty:
        return {}
    latest_dt = qb_dc.groupby('team')['dt'].transform('max')
    qb1 = qb_dc[(qb_dc['dt'] == latest_dt) & (qb_dc['pos_rank'] == 1)][['team', 'player_name', 'gsis_id']]

    try:
        roster = pd.read_csv(f'https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{season}.csv', low_memory=False)
        roster_qb = roster[roster['position'] == 'QB'][['gsis_id', 'rookie_year']]
        qb1 = qb1.merge(roster_qb, on='gsis_id', how='left')
        qb1['is_rookie'] = qb1['rookie_year'] == season
    except Exception:
        qb1['is_rookie'] = False

    result = {}
    for _, row in qb1.iterrows():
        result[row['team']] = {'starter_name': row['player_name'], 'is_rookie': bool(row.get('is_rookie', False))}
    return result


def load_situational_overrides(path='situational_overrides.csv'):
    """One-off situational adjustments that don't fit any systematic,
    backtested category - new stadium openers, unusual one-time
    circumstances, anything genuinely idiosyncratic. Unlike team_hfa.csv
    (season-long, per-team), these are scoped to a specific team+week so
    they only apply to the one game they're meant for.
    Format: team,week,adjustment,note"""
    try:
        df = pd.read_csv(path)
        overrides = {}
        for _, row in df.iterrows():
            key = (row['team'], int(row['week']))
            overrides[key] = {'adjustment': float(row['adjustment']), 'note': row.get('note', '')}
        return overrides
    except FileNotFoundError:
        return {}


def get_situational_adjustment(team, week, overrides):
    entry = overrides.get((team, week))
    return entry['adjustment'] if entry else 0.0


def load_qb_overrides(path='qb_overrides.csv'):
    """Optional manual override file: team,backup_starting,rookie_starting,
    normal_starter_override (all optional except team). The last field
    matters when the depth chart has ALREADY updated to show the backup as
    QB1 (since it's a live-updating source) - fill it in with the name of
    the elite starter who's actually out, so the elite-starter-out bonus
    still applies correctly. Leave blank if the depth chart hasn't caught
    up yet (most common case) - the current depth-chart name is used as-is.
    Returns {} if file absent - meaning every team's depth-chart starter is
    assumed to play, no QB penalty applied."""
    try:
        df = pd.read_csv(path)
        result = {}
        for _, row in df.iterrows():
            result[row['team']] = {
                'backup_starting': str(row.get('backup_starting', 'no')).strip().lower() == 'yes',
                'rookie_starting': str(row.get('rookie_starting', 'no')).strip().lower() == 'yes',
                'normal_starter_override': row.get('normal_starter_override') if pd.notna(row.get('normal_starter_override')) else None,
            }
        return result
    except FileNotFoundError:
        return {}


LAMAR_HALF_WEIGHT_NAME = 'Lamar Jackson'


def compute_qb_coefficient(home_team, away_team, starters, elite_qbs, overrides):
    """Mirrors the validated historical logic exactly: backup starting = -1,
    rookie starting = -1 (stacks with backup), elite-starter-out = an
    ADDITIONAL -1 (-0.5 specifically for Lamar Jackson, per the user's
    original hand-tiering exception) on top of backup/rookie, only when the
    team's normal starter (now out) was elite-tier."""
    def team_qb_value(team):
        info = starters.get(team, {})
        depth_chart_name = info.get('starter_name')
        override = overrides.get(team, {})
        backup_starting = override.get('backup_starting', False)
        rookie_starting = override.get('rookie_starting', False)
        # Which name represents the "normal" starter for elite-check purposes:
        # the manual override if provided (depth chart already updated to
        # show the backup), otherwise the current depth-chart name as-is.
        normal_starter_name = override.get('normal_starter_override') or depth_chart_name

        value = 0.0
        notes = []
        if backup_starting:
            value += -1.0
            notes.append('Backup QB starting')
        if rookie_starting:
            value += -1.0
            notes.append('Rookie QB starting')
        if backup_starting:
            starter_pbp = to_pbp_format(normal_starter_name)
            if starter_pbp in elite_qbs:
                if normal_starter_name == LAMAR_HALF_WEIGHT_NAME:
                    value += -0.5
                    notes.append('Elite starter (Lamar Jackson - half weight) out')
                else:
                    value += -1.0
                    notes.append(f'Elite starter ({normal_starter_name}) out')

        candidate = {
            'team': team, 'depth_chart_starter': depth_chart_name,
            'normal_starter_used_for_elite_check': normal_starter_name,
            'is_elite_starter': to_pbp_format(normal_starter_name) in elite_qbs if normal_starter_name else False,
            'backup_starting': backup_starting, 'rookie_starting': rookie_starting,
            'qb_value': round(value, 2), 'notes': notes,
        }
        return value, candidate

    home_val, home_candidate = team_qb_value(home_team)
    away_val, away_candidate = team_qb_value(away_team)
    return home_val - away_val, [home_candidate, away_candidate]


def main(season, week):
    print(f'=== Scoring {season} Week {week} ===\n')
    games_all = fetch_games([season - 1, season])
    target_games = games_all[(games_all['season'] == season) & (games_all['week'] == week)].copy()
    if target_games.empty:
        print(f'No games found for {season} week {week}.')
        return

    print('Fetching play-by-play (current + prior season)...')
    off_cur = build_offense_stats(fetch_pbp(season))
    off_prior = build_offense_stats(fetch_pbp(season - 1))
    cur_full = build_full_game_table(games_all[games_all['season'] == season], off_cur) if not off_cur.empty else pd.DataFrame()
    prior_full = build_full_game_table(games_all[games_all['season'] == season - 1], off_prior) if not off_prior.empty else pd.DataFrame()

    print(f'Blend weight this week: {prior_year_weight(week)*100:.0f}% prior-year / {(1-prior_year_weight(week))*100:.0f}% current-season\n')

    all_inputs = {t: compute_team_inputs(t, cur_full, prior_full, week) for t in ALL_TEAMS}
    league_df = pd.DataFrame(all_inputs).T
    league_df['rank_pass_off'] = league_df['pass_ypg'].rank(ascending=False, method='min')
    league_df['rank_rush_off'] = league_df['rush_ypg'].rank(ascending=False, method='min')

    power_ratings = load_power_ratings()
    team_hfa = load_team_hfa()
    line_log = load_line_history()
    print(f'Line history log: {len(line_log)} total entries loaded\n' if not line_log.empty else 'No line_history.csv found yet - line movement columns will be empty\n')
    if power_ratings:
        print(f'Loaded power ratings for {len(power_ratings)} teams (team-specific HFA: {len(team_hfa)} teams)\n')
    else:
        print('No power_ratings.csv found - skipping power-rating spread column\n')

    print('Fetching current depth charts and prior-season elite QB list...')
    starters = fetch_current_starters(season)
    elite_qbs = fetch_elite_qb_list(season - 1)
    qb_overrides = load_qb_overrides()
    situational_overrides = load_situational_overrides()
    print(f'  {len(starters)} teams with current QB1 identified, {len(elite_qbs)} elite QBs from {season-1}, '
          f'{len(qb_overrides)} manual overrides loaded\n')

    injury_report = fetch_injury_report(season, week)
    print(f'Injury report: {len(injury_report)} Out/Doubtful non-QB players found for week {week}'
          if not injury_report.empty else 'No injury report available yet for this week\n')

    try:
        injury_history = pd.read_csv('injury_history.csv')
    except FileNotFoundError:
        injury_history = pd.DataFrame()

    with open('production_model.pkl', 'rb') as f:
        saved = pickle.load(f)
    model, components = saved['model'], saved['components']

    results = []
    for _, game in target_games.iterrows():
        home, away = game['home_team'], game['away_team']
        h, a = league_df.loc[home], league_df.loc[away]

        home_streak = compute_streak(cur_full[cur_full['team'] == home]) if not cur_full.empty else 0
        away_streak = compute_streak(cur_full[cur_full['team'] == away]) if not cur_full.empty else 0
        streak_coef = streak_to_value(home_streak) - streak_to_value(away_streak)

        home_rest, away_rest, weekday = game.get('home_rest', 7), game.get('away_rest', 7), game.get('weekday')

        def rest_value(rest_days, is_home_side):
            if pd.isna(rest_days):
                return 0.0
            if rest_days >= 13:
                return 1.0
            elif rest_days > 7:
                return 0.5
            elif rest_days < 7:
                return (0.5 if is_home_side else -0.5) if weekday == 'Thursday' else -0.5
            return 0.0

        rest_coef = rest_value(home_rest, True) - rest_value(away_rest, False)

        h_to, a_to = h['turnover_margin'], a['turnover_margin']
        turnover_coef = 0.0
        if pd.notna(h_to) and pd.notna(a_to):
            straddles = (h_to >= 0 and a_to <= 0) or (h_to <= 0 and a_to >= 0)
            if straddles and not (h_to == 0 and a_to == 0):
                turnover_coef = 1.0 if h_to > a_to else (-1.0 if h_to < a_to else 0.0)

        ypp_off_coef = straight(h['ypp_off'], a['ypp_off'])
        ypp_def_coef = straight(h['ypp_def'], a['ypp_def'], lower_better=True)
        third_down_coef = straight(h['third_down_pct'], a['third_down_pct'])
        rank_coef = rank_comparison(h['rank_pass_off'], a['rank_pass_off']) + \
                    rank_comparison(h['rank_rush_off'], a['rank_rush_off'])

        qb_coef, qb_candidates = compute_qb_coefficient(home, away, starters, elite_qbs, qb_overrides)

        # --- Road trip, coaching, stakes, schedule milestones (home vs away) ---
        home_sched = compute_schedule_situational(home, week, True, games_all[games_all['season'] == season])
        away_sched = compute_schedule_situational(away, week, False, games_all[games_all['season'] == season])

        def road_trip_value(sched):
            v = 0.0
            rtp = sched['road_trip_position']
            if rtp == 1:
                v += -0.5
            elif rtp is not None and rtp >= 2:
                net = sched['road_trip_wins'] - sched['road_trip_losses']
                if net > 0:
                    v += 1.0
                elif net < 0:
                    v += -1.0
            if sched.get('is_sandwich_home'):
                v += -0.5
            if sched.get('is_last_before_trip'):
                v += -0.5
            if sched.get('is_first_after_trip'):
                v += 0.5
            return v

        road_trip_coef = road_trip_value(home_sched) - road_trip_value(away_sched)

        def milestones_value(sched):
            v = 0.0
            if sched['is_home_opener']:
                v += 0.5
            if sched['is_last_before_bye']:
                v += 1.0
            return v

        milestones_coef = milestones_value(home_sched) - milestones_value(away_sched)

        home_new_hc = compute_coaching_change(home, season, games_all)
        away_new_hc = compute_coaching_change(away, season, games_all)
        coaching_coef = (1.0 if home_new_hc else 0.0) - (1.0 if away_new_hc else 0.0)

        home_stakes = compute_stakes(home, season, week, games_all)
        away_stakes = compute_stakes(away, season, week, games_all)

        def stakes_value(s, sched):
            v = 0.0
            if s['not_playing_for_anything']:
                v += -1.0
            if s['trying_to_stay_alive']:
                v += 1.0
            if s['clinched_playoff_berth']:
                v += 1.0 if (s['conf_rank'] is not None and s['conf_rank'] <= 2) else -1.0
            # Last game of season: asymmetric per the original design (face-value,
            # not independently backtested to the same rigor as the rest of this
            # category) - only boost when something is genuinely still on the
            # line; no penalty guessed for a meaningless finale, since that
            # direction was never validated.
            if sched.get('is_last_game_of_season'):
                if s['trying_to_stay_alive'] or (s['clinched_playoff_berth'] and s['conf_rank'] is not None and s['conf_rank'] <= 2):
                    v += 1.0
            return v

        stakes_coef = stakes_value(home_stakes, home_sched) - stakes_value(away_stakes, away_sched)

        feature_vec = {c: 0.0 for c in components}
        feature_vec.update({
            'turnover_coefficient_home': turnover_coef, 'ypp_off_coefficient_home': ypp_off_coef,
            'ypp_def_coefficient_home': ypp_def_coef, 'third_down_coefficient_home': third_down_coef,
            'rank_coefficient_home': rank_coef, 'rest_coefficient_home': rest_coef,
            'streak_coefficient_home': streak_coef, 'qb_coefficient_home': qb_coef,
            'road_trip_coefficient_home': road_trip_coef, 'schedule_milestones_coefficient_home': milestones_coef,
            'coaching_coefficient_home': coaching_coef, 'stakes_coefficient_home': stakes_coef,
        })
        X = np.array([[feature_vec[c] for c in components]])
        model_win_prob = float(model.predict_proba(X)[0][1])
        model_raw_strength = float(model.decision_function(X)[0])

        # One-off situational adjustment (new stadium openers, etc.) - applied
        # as a transparent layer ON TOP of the trained model's prediction,
        # not folded into any learned-weight category (which would misrepresent
        # what that category's backtested weight actually means).
        situational_adj = get_situational_adjustment(home, week, situational_overrides) - \
                           get_situational_adjustment(away, week, situational_overrides)
        raw_strength_score = model_raw_strength + situational_adj
        win_prob_home = 1 / (1 + np.exp(-raw_strength_score)) if situational_adj != 0 else model_win_prob

        power_spread = implied_power_spread(home, away, power_ratings, team_hfa,
                                             game_month=pd.to_datetime(game.get('gameday')).month if pd.notna(game.get('gameday')) else None)
        is_neutral = game.get('location') == 'Neutral'
        if is_neutral:
            # No home-field advantage at a neutral site - pure rating differential
            if home in power_ratings and away in power_ratings:
                power_spread = round(power_ratings[away] - power_ratings[home], 2)

        line_movement = get_line_movement(home, away, line_log)

        results.append({
            'game_id': game['game_id'], 'home_team': home, 'away_team': away,
            'gameday': str(game.get('gameday')), 'gametime': str(game.get('gametime')),
            'weekday': str(game.get('weekday')), 'home_win_probability': round(win_prob_home, 3),
            'away_win_probability': round(1 - win_prob_home, 3),
            'favored': home if win_prob_home > 0.5 else away,
            'raw_strength_score': round(raw_strength_score, 2),
            'feature_breakdown': {k: round(v, 2) for k, v in feature_vec.items() if v != 0},
            'power_rating_implied_spread': power_spread,
            'neutral_site': is_neutral,
            'qb_candidates': qb_candidates,
            'home_injuries': get_injury_candidates(home, injury_report, injury_history),
            'away_injuries': get_injury_candidates(away, injury_report, injury_history),
            'situational_note': situational_overrides.get((home, week), situational_overrides.get((away, week), {})).get('note', ''),
            'spread_current': line_movement['spread']['current'],
            'spread_history': line_movement['spread']['history'],
            'total_current': line_movement['total']['current'],
            'total_history': line_movement['total']['history'],
            'home_score': float(game['home_score']) if pd.notna(game.get('home_score')) else None,
            'away_score': float(game['away_score']) if pd.notna(game.get('away_score')) else None,
            'is_final': pd.notna(game.get('home_score')) and pd.notna(game.get('away_score')),
        })
        if is_neutral:
            print(f"  ** NEUTRAL SITE ({game.get('stadium', 'unknown venue')}) - win probability above still reflects the model's learned home-field weighting, which is NOT corrected for this game. Power rating spread IS corrected (no HFA applied). **")
        print(f"{away} @ {home}: {home} {win_prob_home*100:.1f}% / {away} {(1-win_prob_home)*100:.1f}%")

    output = {
        'season': season, 'week': week,
        'blend_weight_prior_year': round(prior_year_weight(week), 2),
        'limitations': [
            'Player trades, sandwich-home-game/last-before-trip/first-after-trip '
            '(the "negligible, half-weight" road-trip sub-components), and the '
            'fully-conditional "last game of season" asymmetric-stakes logic are '
            'still not live (low-frequency/low-weight, deferred - see spec). '
            'Everything else (rank/YPP/3rddown/turnover/rest/streak/QB/road trip '
            'position/coaching/stakes/schedule milestones) is now live. '
            'Full spec in nfl_model_spec.md.',
        ],
        'games': results,
    }

    os.makedirs('docs/data', exist_ok=True)
    out_path = f'docs/data/{season}_{week}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f'\nSaved {out_path}')

    update_manifest(season, week)


def update_manifest(season, week, manifest_path='docs/data/manifest.json'):
    """Scans docs/data/ for all archived week files and rewrites the
    manifest, so the dashboard can build a week-selector without needing
    to know in advance which weeks exist."""
    data_dir = os.path.dirname(manifest_path)
    entries = []
    if os.path.isdir(data_dir):
        for fname in os.listdir(data_dir):
            if fname.endswith('.json') and fname != 'manifest.json':
                try:
                    s, w = fname[:-5].split('_')
                    entries.append({'season': int(s), 'week': int(w), 'file': fname})
                except ValueError:
                    continue
    entries.sort(key=lambda e: (e['season'], e['week']))
    manifest = {'weeks': entries, 'latest': {'season': season, 'week': week}}
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'Updated manifest.json - {len(entries)} weeks archived')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--season', type=int, required=True)
    parser.add_argument('--week', type=int, required=True)
    args = parser.parse_args()
    main(args.season, args.week)
