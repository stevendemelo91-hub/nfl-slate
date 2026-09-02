"""
snapshot_injuries.py — daily injury/practice-report snapshotter.
Separate from score_week.py, on its own more-frequent schedule (daily,
every day of the week - injury news matters Tue-Fri, not just Mon/Tue when
the main scoring model runs). Captures the practice_status trend
(DNP/Limited/Full) and report_status (Out/Doubtful/Questionable) that
nflverse's own file does NOT preserve as a history - it's a single
snapshot, overwritten each time nflverse updates it. This script builds
the trend ourselves by snapshotting repeatedly and logging only genuine
changes, same pattern as snapshot_lines.py for market movement.

Usage: python3 snapshot_injuries.py --season 2026
"""
import pandas as pd
import argparse
from datetime import datetime, timezone

LOG_PATH = 'injury_history.csv'
LOG_COLUMNS = ['gsis_id', 'team', 'full_name', 'position', 'week',
               'report_status', 'practice_status', 'report_primary_injury',
               'snapshot_date']


def fetch_current_injuries(season):
    try:
        return pd.read_csv(
            f'https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.csv',
            low_memory=False)
    except Exception as e:
        print(f'Could not fetch injuries_{season}.csv: {e}')
        return pd.DataFrame()


def load_existing_log():
    try:
        return pd.read_csv(LOG_PATH)
    except FileNotFoundError:
        return pd.DataFrame(columns=LOG_COLUMNS)


def ensure_log_file_exists():
    """Guarantees LOG_PATH exists on disk (even as just a header row) so
    the workflow's `git add` step always has something valid to target -
    otherwise a cold start with nothing to log yet (e.g. before real
    injury reports are published) leaves no file at all, and git fails
    with 'pathspec did not match any files'."""
    import os
    if not os.path.exists(LOG_PATH):
        pd.DataFrame(columns=LOG_COLUMNS).to_csv(LOG_PATH, index=False)
        print(f'Created empty {LOG_PATH} (header only) so git has something to track')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--season', type=int, required=True)
    args = parser.parse_args()

    snapshot_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    current = fetch_current_injuries(args.season)
    if current.empty:
        print('No injury data available - nothing to snapshot')
        ensure_log_file_exists()
        return

    existing = load_existing_log()
    new_rows = []

    for _, row in current.iterrows():
        if pd.isna(row.get('gsis_id')) or pd.isna(row.get('week')):
            continue
        key_report = row.get('report_status')
        key_practice = row.get('practice_status')
        if pd.isna(key_report) and pd.isna(key_practice):
            continue  # no real status to log (player not on the report)

        prior = existing[(existing['gsis_id'] == row['gsis_id']) & (existing['week'] == row['week'])]
        if not prior.empty:
            last = prior.sort_values('snapshot_date').iloc[-1]
            same_report = (pd.isna(last['report_status']) and pd.isna(key_report)) or (last['report_status'] == key_report)
            same_practice = (pd.isna(last['practice_status']) and pd.isna(key_practice)) or (last['practice_status'] == key_practice)
            if same_report and same_practice:
                continue  # no change since last snapshot - skip

        new_rows.append({
            'gsis_id': row['gsis_id'], 'team': row['team'], 'full_name': row['full_name'],
            'position': row['position'], 'week': row['week'],
            'report_status': key_report, 'practice_status': key_practice,
            'report_primary_injury': row.get('report_primary_injury'),
            'snapshot_date': snapshot_date,
        })

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(LOG_PATH, index=False)
        print(f'Logged {len(new_rows)} new injury-status changes')
        for r in new_rows[:20]:
            print(f"  Wk{r['week']} {r['team']} {r['full_name']} ({r['position']}): "
                  f"report={r['report_status']} practice={r['practice_status']}")
        if len(new_rows) > 20:
            print(f'  ... and {len(new_rows) - 20} more')
    else:
        print('No injury/practice status changes detected since last snapshot')
        ensure_log_file_exists()


if __name__ == '__main__':
    main()
