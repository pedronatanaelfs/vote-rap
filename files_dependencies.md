global_votes_prediction_FULL_enhanced.ipynb:

data/vote_sessions_full.csv - from: scripts/00 - Data Aquisition/data_aquisition.py
data/author_popularity.csv - from: scripts/01-feature-engineering/Author's Popularity/authors_popularity.py
data/party_popularity_best_window_last_5_sessions.csv - from: scripts/01-feature-engineering/Party Popularity/party_popularity.py
data/proposition_history_predictions_historical_probability_rule.csv - from: scripts/01-feature-engineering/Historical Approval Rate/historical_approval_rate.py


---

scripts/00 - Data Aquisition/data_aquisition.py:

data/voting/votacoes-{year}.csv (2003-2024)
data/voting/proposition/votacoesProposicoes-{year}.csv (2003-2024)
data/voting/orientations/votacoesOrientacoes-{year}.csv (2003-2024)
data/authors/proposicoesAutores-{year}.csv (2000-2024)
data/propositions/proposicoesTemas-{year}.csv (2000-2024)
data/extra/legislaturas.csv

Outputs:
data/vote_sessions_full.csv


---

scripts/01-feature-engineering/Author's Popularity/authors_popularity.py:

data/vote_sessions_full.csv
data/voting/votes/votacoesVotos-{year}.csv (2003-2024)

Outputs:
data/author_popularity.csv


---

scripts/01-feature-engineering/Party Popularity/party_popularity.py:

data/vote_sessions_full.csv
data/extra/orgaosDeputados-L{legislature}.csv (legislatures 51-57)

Outputs:
party_popularity_best_window_last_5_sessions.csv
party_popularity_auroc_comparison.png


---

scripts/01-feature-engineering/Historical Approval Rate/historical_approval_rate.py:

data/vote_sessions_full.csv

Outputs:
proposition_history_predictions_historical_probability_rule.csv
proposition_history_rules_comparison.png
