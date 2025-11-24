# Setup Notes

## Path Updates

The notebooks have been updated to work with the new repository structure. However, if you encounter path issues:

### Feature Engineering Notebooks
- Located in: `notebooks/01-feature-engineering/`
- Data files should be in: `../../data/` (relative to notebook location)

### Modeling Notebook
- Located in: `notebooks/02-modeling/`
- Data files should be in: `../../data/` (relative to notebook location)

## Data Files

All required data files are included in the `data/` directory:
- `vote_sessions_full.csv` - Main dataset
- `author_popularity.csv` - Author popularity features
- `party_popularity_best_window_last_5_sessions.csv` - Party popularity features
- `proposition_history_predictions_historical_probability_rule.csv` - Historical approval rate
- `voting_sessions_orientations_clean.csv` - Vote orientation data

## Running from Different Locations

If you need to run notebooks from a different location, you may need to update the paths. The current structure assumes:
- Repository root: `vote-rap/`
- Data directory: `vote-rap/data/`
- Notebooks: `vote-rap/notebooks/`

## Additional Data Requirements

Some feature engineering notebooks may reference additional data files that aren't included in this repository (e.g., raw API responses, intermediate processing files). In such cases:
1. Use the pre-computed feature files already in `data/`
2. Or refer to the original research repository for complete data acquisition scripts

