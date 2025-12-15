import pandas as pd
import os
import re

# 1 - Get the data from the voting directory

print("Starting to load voting data from directory...")

# Define the path pattern
base_path = "../../data/voting/"
years = range(2003, 2025)

# Initialize an empty list to store dataframes
dfs = []

# Loop through years and load each CSV
for year in years:
    file_path = os.path.join(base_path, f"votacoes-{year}.csv")
    print(f"Checking for file: {file_path}")
    if os.path.exists(file_path):
        print(f"Loading file for year {year}...")
        df = pd.read_csv(file_path, delimiter=';', quotechar='"')
        df["year"] = year  # Add a column to track the year
        dfs.append(df)
        print(f"Loaded data for year {year}, shape: {df.shape}")
    else:
        print(f"File not found: {file_path}")

if dfs:
    print("Concatenating all loaded dataframes...")
    df_sessions = pd.concat(dfs, ignore_index=True)
    print(f"Concatenation complete. Final dataframe shape: {df_sessions.shape}")
else:
    print("No dataframes were loaded. Please check the data directory and file names.")

# 2 - Select the columns that we need
print("Selecting relevant columns from session data...")
columns_to_drop = [
    'dataHoraRegistro', 'uriOrgao', 'idEvento', 'uriEvento', 'votosSim', 'votosNao', 'votosOutros',
    'ultimaAberturaVotacao_dataHoraRegistro', 'ultimaApresentacaoProposicao_dataHoraRegistro',
    'ultimaApresentacaoProposicao_idProposicao', 'ultimaApresentacaoProposicao_uriProposicao'
]
print(f"Columns to drop: {columns_to_drop}")
df_session_selected = df_sessions.drop(columns=columns_to_drop)
print(f"Columns selected. New dataframe shape: {df_session_selected.shape}")

# 3 - Get the data from the proposition directory
print("Starting to load proposition data from directory...")

# Define base path
base_path = '../../data/voting/proposition'

# Initialize an empty list to store dataframes
dfs = []

# Loop through years and load each CSV
for year in range(2003, 2025):
    file_path = os.path.join(base_path, f'votacoesProposicoes-{year}.csv')
    print(f"Checking for file: {file_path}")
    if os.path.exists(file_path):
        print(f"Loading proposition file for year {year}...")
        df = pd.read_csv(file_path, dtype={'idVotacao': str, 'proposicao_id': str}, delimiter=';', quotechar='"')
        df["year"] = year  # Add a column to track the year
        dfs.append(df)
        print(f"Loaded proposition data for year {year}, shape: {df.shape}")
    else:
        print(f"File not found: {file_path}")

if dfs:
    print("Concatenating all loaded proposition dataframes...")
    df_propositions = pd.concat(dfs, ignore_index=True)
    print(f"Concatenation complete. Final proposition dataframe shape: {df_propositions.shape}")
else:
    print("No proposition dataframes were loaded. Please check the data directory and file names.")

if 'proposicao_siglaTipo' in df_propositions.columns:
    original_shape = df_propositions.shape
    df_propositions = df_propositions[df_propositions['proposicao_siglaTipo'].astype(str).str.contains("PL", na=False)]
    print(f"Filtered df_propositions to only rows with 'PL' in 'proposicao_siglaTipo'. Shape before: {original_shape}, after: {df_propositions.shape}")

# 4 - Convert the proposicao_id column to int64
df_propositions['proposicao_id'] = df_propositions['proposicao_id'].astype('int64')
print(f"Proposicao_id column converted to int64. New dataframe shape: {df_propositions.shape}")

# 5 - Merge the two dataframes
print("Merging session and proposition dataframes...")
# Merge to add 'propositionID' column and manter 'proposicao_siglaTipo'
df_session_selected = df_session_selected.merge(
    df_propositions[['idVotacao', 'proposicao_id', 'proposicao_siglaTipo']],
    left_on='id',
    right_on='idVotacao',
    how='left'
)

print(f"Merged dataframe shape: {df_session_selected.shape}")

# 6 - Rename the preposition column
# Rename column
df_session_selected.rename(columns={'proposicao_id': 'propositionID'}, inplace=True)

# 7 - Drop redundant column
df_session_selected.drop(columns=['idVotacao'], inplace=True)

# 8 - Get the data from the orientations directory
print("Starting to load orientations data from directory...")
# Define base path
base_path = '../../data/voting/orientations'

# Initialize an empty list to store dataframes
dfs = []

# Loop through years and load each CSV
for year in range(2003, 2025):
    file_path = os.path.join(base_path, f'votacoesOrientacoes-{year}.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, dtype={'idVotacao': str}, delimiter=';', quotechar='"')
        df["year"] = year  # Add a column to track the year
        dfs.append(df)
    else:
        print(f"File not found: {file_path}")

if dfs:
    print("Concatenating all loaded orientations dataframes...")
    df_orientations = pd.concat(dfs, ignore_index=True)
    print(f"Concatenation complete. Final orientations dataframe shape: {df_orientations.shape}")
else:
    print("No orientations dataframes were loaded. Please check the data directory and file names.")

# 9 - Get unique values from 'siglaBancada'
unique_siglaBancada = df_orientations["siglaBancada"].unique()
print(f"Unique values from 'siglaBancada': {unique_siglaBancada}")

# 10 - Pivot the df_orientations to have 'siglaBancada' values as columns
df_orientations_pivot = df_orientations.pivot(index='idVotacao', columns='siglaBancada', values='orientacao')
df_orientations_pivot.reset_index(inplace=True)
print(f"Pivoted dataframe shape: {df_orientations_pivot.shape}")

# 11 - Merge with df_sessions_selected
df_session_orientation = df_session_selected.merge(df_orientations_pivot, left_on='id', right_on='idVotacao', how='left')
print(f"Merged dataframe shape: {df_session_orientation.shape}")

# 12 - Map the orientation values
print("Mapping orientation values...")
# Define mapping function
def map_orientation(value):
    if isinstance(value, str):
        value_lower = value.lower()
        if value_lower in ["sim", "yes", "y"]:
            return 1
        elif value_lower in ["não", "nao", "no", "n"]:
            return -1
    return 0

# Apply mapping to all new columns from siglaBancada
for column in unique_siglaBancada:
    df_session_orientation[column] = df_session_orientation[column].map(map_orientation)
print(f"Mapping complete. New dataframe shape: {df_session_orientation.shape}")

# 13 - Load authors data
print("Starting to load authors data from directory...")
dfs_authors = []
author_years = range(2000, 2025)

type_path = "../../data/authors/"
for year in author_years:
    file_path = os.path.join(type_path, f"proposicoesAutores-{year}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, delimiter=';', quotechar='"')
        df["year"] = year  # Add a column to track the year
        dfs_authors.append(df)
    else:
        print(f"File not found: {file_path}")

if dfs_authors:
    print("Concatenating all loaded authors dataframes...")
    df_authors = pd.concat(dfs_authors, ignore_index=True)
    print(f"Concatenation complete. Final authors dataframe shape: {df_authors.shape}")
else:
    print("No authors dataframes were loaded. Please check the data directory and file names.")

# 14 - Merge with df_session_orientation
# Perform the merge
print("Merging authors data with session orientation data...")
df_session_author = df_session_orientation.merge(
    df_authors[['idProposicao', 'tipoAutor', 'nomeAutor', 'codTipoAutor', 'idDeputadoAutor']],
    left_on='propositionID',
    right_on='idProposicao',
    how='left'
)
print(f"Merged dataframe shape: {df_session_author.shape}")

# 15 - Rename new columns
print("Renaming new columns...")
df_session_author.rename(columns={
    "tipoAutor": "author_type", 
    "nomeAutor": "author",
    "codTipoAutor": "author_type_code"
}, inplace=True)
print(f"Renaming complete. New dataframe shape: {df_session_author.shape}")

# 16 - Drop rows where 'idProposicao' or 'author' is NaN, null, or zero
print("Dropping rows where 'idProposicao' or 'author' is NaN, null, or zero...")
df_session_author = df_session_author.dropna(subset=["idProposicao", "author"])
df_session_author = df_session_author[(df_session_author["idProposicao"] != 0) & (df_session_author["author"] != 0)]
print(f"Dropping complete. New dataframe shape: {df_session_author.shape}")

# 16.1 - Group authors by proposition if there are multiple authors
print("Processing propositions with multiple authors...")
# Count authors per proposition
author_counts = df_session_author.groupby('propositionID')['author'].count().reset_index()
author_counts.rename(columns={'author': 'num_authors'}, inplace=True)

# Merge back to get author count per proposition
df_session_author = df_session_author.merge(author_counts, on='propositionID', how='left')
print(f"Found {(df_session_author['num_authors'] > 1).sum()} propositions with multiple authors")

# Keep all authors without filtering
df_session_author_filtered = df_session_author.copy()
print(f"Total author records: {df_session_author_filtered.shape}")

# 17 - Get the data from the PLEN
print("Getting data from PLEN...")
df_session_PLEN = df_session_author[(df_session_author["idOrgao"] == 180)]
print(f"Data from PLEN shape: {df_session_PLEN.shape}")

# 18 - Load propositions themes data
print("Starting to load propositions themes data from directory...")
dfs_themes = []
themes_path = "../../data/propositions/"
for year in range(2000, 2025):
    file_path = os.path.join(themes_path, f"proposicoesTemas-{year}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, delimiter=';', quotechar='"')
        dfs_themes.append(df)
    else:
        print(f"File not found: {file_path}")
print(f"Loaded {len(dfs_themes)} themes dataframes.")

# 19 - Concatenate all themes dataframes into one
print("Concatenating all themes dataframes into one...")
# Concatenate all themes dataframes into one
df_themes = pd.concat(dfs_themes, ignore_index=True)
print(f"Concatenation complete. Final themes dataframe shape: {df_themes.shape}")

# 20 - Extract proposition code from 'uriProposicao'
print("Extracting proposition code from 'uriProposicao'...")
df_themes['idProposicao'] = df_themes['uriProposicao'].apply(lambda x: int(re.search(r'\d+$', x).group()) if isinstance(x, str) and re.search(r'\d+$', x) else None)
print(f"Extraction complete. New dataframe shape: {df_themes.shape}")

# 21 - Merge themes data with df_session_PLEN
print("Merging themes data with df_session_PLEN...")
# Merge themes data with df_session_PLEN
df_session_theme = df_session_PLEN.merge(df_themes[['idProposicao', 'tema']],
                                         left_on='propositionID',
                                         right_on='idProposicao',
                                         how='left')
print(f"Merging complete. New dataframe shape: {df_session_theme.shape}")

# 22 - Rename new column
print("Renaming new column...")
df_session_theme.rename(columns={"tema": "theme"}, inplace=True)
print(f"Renaming complete. New dataframe shape: {df_session_theme.shape}")

# 23 - Add legislature information based on voting date
print("Adding legislature information based on voting date...")

# Load legislatures data
legislatures_path = "../../data/extra/legislaturas.csv"
df_legislaturas = pd.read_csv(legislatures_path, delimiter=';', quotechar='"')

# Convert date columns to datetime format
df_legislaturas['dataInicio'] = pd.to_datetime(df_legislaturas['dataInicio'])
df_legislaturas['dataFim'] = pd.to_datetime(df_legislaturas['dataFim'])

# Make sure the 'data' column in our dataset is in datetime format
df_session_theme['data'] = pd.to_datetime(df_session_theme['data'])

# Create a function to find the legislature for a given date
def find_legislature(date):
    if pd.isnull(date):
        return None
    
    for _, row in df_legislaturas.iterrows():
        if row['dataInicio'] <= date <= row['dataFim']:
            return int(row['idLegislatura'])
    
    return None

# Apply the function to create a new 'legislatura' column
df_session_theme['legislatura'] = df_session_theme['data'].apply(find_legislature)

# Report legislature mapping results
legislature_counts = df_session_theme['legislatura'].value_counts().sort_index()
print(f"Legislature mapping complete. Distribution of legislatures:")
for leg, count in legislature_counts.items():
    print(f"Legislature {leg}: {count} records")
print(f"Records without legislature assignment: {df_session_theme['legislatura'].isnull().sum()}")

# 24 - Save the dataframe
print("Saving dataframe...")
df_all_info = df_session_theme.copy()
df_all_info.to_csv("../../data/vote_sessions_full.csv", index=False)

print("DataFrame successfully exported as 'vote_sessions_full.csv'")
