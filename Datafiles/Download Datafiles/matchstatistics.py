import os
import requests
import pandas as pd
import numpy as np

# Paths
DATASET_UPDATE_DIR = "Temp"
DATASETS_DIR = "Datafiles"
OUTPUT_FILE = "Bundesliga_MatchStats.csv"

# URLs for historical data
SEASON_URLS = {
    "2016": "https://www.football-data.co.uk/mmz4281/1617/D1.csv",
    "2017": "https://www.football-data.co.uk/mmz4281/1718/D1.csv",
    "2018": "https://www.football-data.co.uk/mmz4281/1819/D1.csv",
    "2019": "https://www.football-data.co.uk/mmz4281/1920/D1.csv",
    "2020": "https://www.football-data.co.uk/mmz4281/2021/D1.csv",
    "2021": "https://www.football-data.co.uk/mmz4281/2122/D1.csv",
    "2022": "https://www.football-data.co.uk/mmz4281/2223/D1.csv",
    "2023": "https://www.football-data.co.uk/mmz4281/2324/D1.csv",
    "2024": "https://www.football-data.co.uk/mmz4281/2425/D1.csv",
}

def download_csv(url, season):
    """Download a CSV file for the given season."""
    os.makedirs(DATASET_UPDATE_DIR, exist_ok=True)
    local_path = os.path.join(DATASET_UPDATE_DIR, f'D1_{season}.csv')
    
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_path, 'wb') as f:
            f.write(response.content)
        print(f"CSV file for season {season} downloaded successfully.")
        return local_path
    else:
        print(f"Failed to download data for season {season}. Status code: {response.status_code}")
        raise Exception("Download failed!")

def process_csv(file_path, season):
    """Process a single CSV file and standardize its format."""
    df = pd.read_csv(file_path)
    
    # Get the required columns
    df_modified = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
                      'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR','HF','AF']].copy()
    
    # Rename columns to match final format
    df_modified.columns = [
        'Date', 'HomeTeam', 'AwayTeam', 'HomeTeamGoals', 'AwayTeamGoals', 'Result',
        'HomeTeamShots', 'AwayTeamShots', 'HomeTeamShotsOnTarget', 'AwayTeamShotsOnTarget',
        'HomeTeamCorners', 'AwayTeamCorners', 'HomeTeamYellow', 'AwayTeamYellow',
        'HomeTeamRed', 'AwayTeamRed','HomeTeamFouls','AwayTeamFouls'
        ]
    
    # Sort by date and calculate gamedays
    df_modified = df_modified.sort_values('Date')
    df_modified['Gameday'] = ((df_modified.index) // 9) + 1
    
    # Add season column
    df_modified.insert(0, 'Season', season)
    
    return df_modified

def create_bundesliga_dataset():
    """Create a new Bundesliga dataset from all available seasons."""
    os.makedirs(DATASETS_DIR, exist_ok=True)
    output_path = os.path.join(DATASETS_DIR, OUTPUT_FILE)
    
    all_data = []
    
    for season, url in SEASON_URLS.items():
        print(f"\nProcessing season {season}...")
        try:
            # Download and process each season
            csv_path = download_csv(url, season)
            processed_data = process_csv(csv_path, season)
            all_data.append(processed_data)
            
            # Print season statistics
            print(f"Season {season}: {len(processed_data)} matches processed")
            
        except Exception as e:
            print(f"Error processing season {season}: {e}")
            continue
    
    # Combine all seasons and sort by Season and Gameday
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values(['Season', 'Gameday'])
    
    # Add final index column
    combined_df.insert(0, 'Index', range(1, len(combined_df) + 1))
    
    # Save the complete dataset
    combined_df.to_csv(output_path, sep=';', index=False)
    print(f"\nComplete dataset created successfully: {output_path}")
    print(f"Total matches: {len(combined_df)}")
    print("\nMatches per season:")
    print(combined_df['Season'].value_counts().sort_index())
    
    # Clean up temporary files
    for file in os.listdir(DATASET_UPDATE_DIR):
        os.remove(os.path.join(DATASET_UPDATE_DIR, file))
    os.rmdir(DATASET_UPDATE_DIR)
    
    return combined_df

if __name__ == "__main__":
    df = create_bundesliga_dataset()