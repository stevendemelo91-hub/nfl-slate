UPLOAD MAP (repo root = top level of your GitHub repo)
 root:                 score_week.py, qb_overrides.csv, player_impact_overrides.csv, pool_results_history.csv
 docs/:                table.html, index.html, pool_history.html   (ONLY in docs/, not at root)
 .github/workflows/:   weekly_score.yml
THEN run Actions > Weekly NFL Scoring > Run workflow with season 2026, in this order: week 4, then week 5
(then week 3 only if you want the coaching-flag change applied there too).
