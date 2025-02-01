import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class FeatureSet:
    names: List[str]
    values: List[float]
    
    def validate(self):
        if len(self.names) != len(self.values):
            raise ValueError(f"Feature mismatch: {len(self.names)} names but {len(self.values)} values")
        return self

def preprocess_bundesliga_data(training_csv_path):
    df_train = pd.read_csv(training_csv_path, delimiter=';')
    df_train.columns = df_train.columns.str.strip()
    
    base_features = [
        'Goals', 'Shots', 'ShotsOnTarget', 
        'Yellow', 'Red'
    ]
    
    def calculate_team_features(team_stats: Dict[str, List], team: str, is_home: bool) -> FeatureSet:
        features = []
        feature_names = []
        role = "Home" if is_home else "Away"
        
        # Standard statistics from their respective games
        for feat in base_features:
            stat_key = f"{role}Team{feat}"
            history = team_stats[team][stat_key]
            avg = (np.mean(history[-20:]) if len(history) >= 20 
                  else np.mean(history) if history else 0)
            features.append(avg)
            feature_names.append(f'Avg_{role}_{feat}')
        
        # Separate home and away conceded goals
        if is_home:
            conceded_history = team_stats[team]['HomeConcededGoals']
        else:
            conceded_history = team_stats[team]['AwayConcededGoals']
            
        avg_conceded = (np.mean(conceded_history[-20:]) if len(conceded_history) >= 20 
                       else np.mean(conceded_history) if conceded_history else 0)
        features.append(avg_conceded)
        feature_names.append(f'Avg_{role}_ConcededGoals')
        
        # Rest of the feature calculation remains the same
        trend_history = team_stats[team]['Results'][-3:] if team_stats[team]['Results'] else []
        trend = 0
        if trend_history:
            points = [3 if result == 'W' else 1 if result == 'D' else 0 for result in trend_history]
            trend = np.mean(points)
        features.append(trend)
        feature_names.append(f'{role}_Trend')
        
        # Win statistics calculation
        if is_home:
            home_wins = team_stats[team]['HomeWins']
            total_wins = team_stats[team]['TotalWins']
        else:
            away_wins = team_stats[team]['AwayWins']
            total_wins = team_stats[team]['TotalWins']
        
        # Add win features
        if is_home:
            hw_avg = (np.mean(home_wins[-20:]) if len(home_wins) >= 20 
                     else np.mean(home_wins) if home_wins else 0)
            features.append(hw_avg)
            feature_names.append('Avg_Home_HomeWins')
        else:
            aw_avg = (np.mean(away_wins[-20:]) if len(away_wins) >= 20 
                     else np.mean(away_wins) if away_wins else 0)
            features.append(aw_avg)
            feature_names.append('Avg_Away_AwayWins')
        
        tw_avg = (np.mean(total_wins[-20:]) if len(total_wins) >= 20 
                 else np.mean(total_wins) if total_wins else 0)
        features.append(tw_avg)
        feature_names.append(f'Avg_{role}_TotalWins')
        
        return FeatureSet(names=feature_names, values=features).validate()
    
    X_processed = []
    y_processed = []
    feature_names = None
    
    for season in df_train['Season'].unique():
        season_data = df_train[df_train['Season'] == season].sort_values('Gameday')
        team_stats = {}
        
        print(f"Processing season {season}")
        
        for _, game in season_data.iterrows():
            home_team = game['HomeTeam']
            away_team = game['AwayTeam']
            gameday = game['Gameday']
            
            for team in [home_team, away_team]:
                if team not in team_stats:
                    team_stats[team] = {
                        **{f"HomeTeam{feat}": [] for feat in base_features},
                        **{f"AwayTeam{feat}": [] for feat in base_features},
                        'HomeConcededGoals': [],  # Separate home conceded goals
                        'AwayConcededGoals': [],  # Separate away conceded goals
                        'HomeWins': [], 'AwayWins': [], 'TotalWins': [],
                        'Results': []
                    }
            
            home_features = calculate_team_features(team_stats, home_team, is_home=True)
            away_features = calculate_team_features(team_stats, away_team, is_home=False)
            
            if feature_names is None:
                feature_names = home_features.names + away_features.names
                print("\nFeature structure:")
                for i, name in enumerate(feature_names):
                    print(f"{i:2d}: {name}")
            
            all_features = home_features.values + away_features.values
            
            if gameday > 1:
                X_processed.append(all_features)
                if game['Result'] == 'H':
                    y_processed.append(0)
                elif game['Result'] == 'A':
                    y_processed.append(1)
                else:
                    y_processed.append(2)
            
            if gameday > 1:
                result = game['Result']
                
                team_stats[home_team]['Results'].append('W' if result == 'H' else 'D' if result == 'D' else 'L')
                team_stats[away_team]['Results'].append('W' if result == 'A' else 'D' if result == 'D' else 'L')
                
                team_stats[home_team]['HomeWins'].append(1 if result == 'H' else 0)
                team_stats[away_team]['AwayWins'].append(1 if result == 'A' else 0)
                team_stats[home_team]['TotalWins'].append(1 if result == 'H' else 0)
                team_stats[away_team]['TotalWins'].append(1 if result == 'A' else 0)
            
            for feat in base_features:
                home_value = game[f'HomeTeam{feat}']
                away_value = game[f'AwayTeam{feat}']
                
                team_stats[home_team][f'HomeTeam{feat}'].append(home_value)
                team_stats[away_team][f'AwayTeam{feat}'].append(away_value)
            
            # Track home and away conceded goals separately
            team_stats[home_team]['HomeConcededGoals'].append(game['AwayTeamGoals'])
            team_stats[away_team]['AwayConcededGoals'].append(game['HomeTeamGoals'])
    
    X = np.array(X_processed)
    y = np.array(y_processed)
    
    return X, y, feature_names