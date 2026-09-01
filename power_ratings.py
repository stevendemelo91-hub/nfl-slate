"""
power_ratings.py — converts user's subjective power ratings into an implied
point spread for a given matchup.

Scale: lower rating = better team. Implied home spread = (away - home) + HFA.
Positive result = home favored by that many points.

HFA default (1.92) is grounded in this project's own backtest: average home
margin in the 2020+ era (see nfl_model_spec.md, "home-field advantage
stability over time"). Override if you want a different assumption.
"""
import pandas as pd

DEFAULT_HFA = 1.89  # recency-weighted league baseline (see nfl_model_spec.md)

DYNAMIC_TEAM_HFA = {
    'MIA': {9: 2.5, 'default': 1.92},
}


def get_team_hfa(home_team, game_month, team_hfa_table):
    if home_team in DYNAMIC_TEAM_HFA:
        rule = DYNAMIC_TEAM_HFA[home_team]
        return rule.get(game_month, rule['default'])
    return team_hfa_table.get(home_team, DEFAULT_HFA)


def load_team_hfa(path='team_hfa.csv'):
    """Team-specific HFA, blended from margin + ATS signal, recency-weighted,
    floored at 0.25 (never punished to zero/negative). Falls back to the
    league baseline for any team not found."""
    try:
        df = pd.read_csv(path)
        return dict(zip(df['team'], df['final_team_hfa']))
    except FileNotFoundError:
        return {}


def load_ratings(path='power_ratings.csv'):
    df = pd.read_csv(path)
    return dict(zip(df['team'], df['rating']))


def implied_spread(home_team, away_team, ratings, hfa=None, team_hfa=None, game_month=None):
    """Returns (implied_home_spread, missing_teams). Positive spread = home favored.
    If team_hfa dict is provided, uses the home team's specific (and, for
    Miami, month-specific) HFA value; otherwise falls back to the flat
    hfa/DEFAULT_HFA."""
    missing = [t for t in [home_team, away_team] if t not in ratings]
    if missing:
        return None, missing
    home_r, away_r = ratings[home_team], ratings[away_team]
    if hfa is not None:
        applied_hfa = hfa
    elif team_hfa:
        applied_hfa = get_team_hfa(home_team, game_month, team_hfa)
    else:
        applied_hfa = DEFAULT_HFA
    return round((away_r - home_r) + applied_hfa, 2), []


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--home', required=True)
    parser.add_argument('--away', required=True)
    parser.add_argument('--hfa', type=float, default=None, help='Override: use one flat HFA for all teams instead of team-specific')
    parser.add_argument('--ratings', default='power_ratings.csv')
    parser.add_argument('--team-hfa', default='team_hfa.csv')
    args = parser.parse_args()

    ratings = load_ratings(args.ratings)
    team_hfa = load_team_hfa(args.team_hfa) if args.hfa is None else {}
    spread, missing = implied_spread(args.home, args.away, ratings, hfa=args.hfa, team_hfa=team_hfa)
    if missing:
        print(f'Missing rating(s) for: {missing}')
    else:
        favored = args.home if spread > 0 else args.away
        applied = team_hfa.get(args.home, args.hfa or DEFAULT_HFA)
        print(f'{args.away} @ {args.home}: implied spread {favored} -{abs(spread)}  (HFA used: {applied})')
