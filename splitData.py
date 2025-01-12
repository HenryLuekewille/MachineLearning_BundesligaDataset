import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class FeatureSet:
    names: List[str]
    values: List[float]
    
    def validate(self):
        if len(self.names) != len(self.values):
            raise ValueError(f"Feature mismatch: {len(self.names)} names but {len(self.values)} values")
        return self

def preprocess_bundesliga_data(training_csv_path, prediction_csv_path=None):
    """
    Preprocesses Bundesliga match data tracking relevant team statistics and trend.
    """
    # Load training data
    df_train = pd.read_csv(training_csv_path, delimiter=';')
    df_train.columns = df_train.columns.str.strip()
    
    # Define base features (without team prefix)
    base_features = [
        'Goals', 'Shots', 'ShotsOnTarget', 
        'Yellow', 'Red'  # Removed 'Corners'
    ]
    
    def calculate_team_features(team_stats: Dict[str, List], team: str, is_home: bool) -> FeatureSet:
        """Calculate relevant features for each team based on their role"""
        features = []
        feature_names = []
        role = "Home" if is_home else "Away"
        
        # Standard statistics from their respective games
        for feat in base_features:
            stat_key = f"{role}Team{feat}"
            history = team_stats[team][stat_key]
            avg = (np.mean(history[-19:]) if len(history) >= 19 
                  else np.mean(history) if history else 0)
            features.append(avg)
            feature_names.append(f'Avg_{role}_{feat}')
        
        # Calculate trend based on last 3 games
        trend_history = team_stats[team]['Results'][-3:] if team_stats[team]['Results'] else []
        trend = 0
        if trend_history:
            # Convert results to points (win=3, draw=1, loss=0)
            points = [3 if result == 'W' else 1 if result == 'D' else 0 for result in trend_history]
            trend = np.mean(points)
        features.append(trend)
        feature_names.append(f'{role}_Trend')

        
        
        # Win statistics
        if is_home:
            home_wins = team_stats[team]['HomeWins']
            total_wins = team_stats[team]['TotalWins']
        else:
            away_wins = team_stats[team]['AwayWins']
            total_wins = team_stats[team]['TotalWins']
        
        # Add win features
        if is_home:
            hw_avg = (np.mean(home_wins[-19:]) if len(home_wins) >= 19 
                     else np.mean(home_wins) if home_wins else 0)
            features.append(hw_avg)
            feature_names.append('Avg_Home_HomeWins')
        else:
            aw_avg = (np.mean(away_wins[-19:]) if len(away_wins) >= 19 
                     else np.mean(away_wins) if away_wins else 0)
            features.append(aw_avg)
            feature_names.append('Avg_Away_AwayWins')
        
        tw_avg = (np.mean(total_wins[-19:]) if len(total_wins) >= 19 
                 else np.mean(total_wins) if total_wins else 0)
        features.append(tw_avg)
        feature_names.append(f'Avg_{role}_TotalWins')
        
        return FeatureSet(names=feature_names, values=features).validate()
    
    # Initialize processing variables
    X_processed = []
    y_processed = []
    final_team_stats = {}
    feature_names = None
    
    # Process historical data
    for season in df_train['Season'].unique():
        season_data = df_train[df_train['Season'] == season].sort_values('Gameday')
        team_stats = {}
        
        print(f"Processing season {season}")
        
        for _, game in season_data.iterrows():
            home_team = game['HomeTeam']
            away_team = game['AwayTeam']
            gameday = game['Gameday']
            
            # Initialize team stats if needed
            for team in [home_team, away_team]:
                if team not in team_stats:
                    team_stats[team] = {
                        **{f"HomeTeam{feat}": [] for feat in base_features},
                        **{f"AwayTeam{feat}": [] for feat in base_features},
                        'HomeWins': [], 'AwayWins': [], 'TotalWins': [],
                        'Results': []  # New field for tracking match results
                    }
            
            # Calculate features with validation
            home_features = calculate_team_features(team_stats, home_team, is_home=True)
            away_features = calculate_team_features(team_stats, away_team, is_home=False)
            
            # Store feature names from first iteration
            if feature_names is None:
                feature_names = home_features.names + away_features.names
                print("\nFeature structure:")
                for i, name in enumerate(feature_names):
                    print(f"{i:2d}: {name}")
            
            # Combine features
            all_features = home_features.values + away_features.values
            
            if gameday > 1:  # Don't use first game for training
                X_processed.append(all_features)
                if game['Result'] == 'H':
                    y_processed.append(0)
                elif game['Result'] == 'A':
                    y_processed.append(1)
                else:  # Draw
                    y_processed.append(2)
            
            # Update stats after feature calculation
            if gameday > 1:
                result = game['Result']
                
                # Update win stats
                is_home_win = result == 'H'
                is_away_win = result == 'A'
                
                # Update results for trend calculation
                team_stats[home_team]['Results'].append('W' if is_home_win else 'D' if result == 'D' else 'L')
                team_stats[away_team]['Results'].append('W' if is_away_win else 'D' if result == 'D' else 'L')
                
                team_stats[home_team]['HomeWins'].append(1 if is_home_win else 0)
                team_stats[away_team]['AwayWins'].append(1 if is_away_win else 0)
                team_stats[home_team]['TotalWins'].append(1 if is_home_win else 0)
                team_stats[away_team]['TotalWins'].append(1 if is_away_win else 0)
            
            # Update game stats
            for feat in base_features:
                home_value = game[f'HomeTeam{feat}']
                away_value = game[f'AwayTeam{feat}']
                
                # Store only relevant statistics
                team_stats[home_team][f'HomeTeam{feat}'].append(home_value)
                team_stats[away_team][f'AwayTeam{feat}'].append(away_value)
        
        final_team_stats.update(team_stats)
    
    X = np.array(X_processed)
    y = np.array(y_processed)
    
    # If no prediction data is needed, return historical data
    if prediction_csv_path is None:
        return X, y, feature_names
    
    # Process prediction data
    df_pred = pd.read_csv(prediction_csv_path, delimiter=';')
    df_pred.columns = df_pred.columns.str.strip()
    
    X_pred = []
    prediction_matches = []
    
    print("\nProcessing prediction data")
    
    for _, game in df_pred.iterrows():
        home_team = game['HomeTeam']
        away_team = game['AwayTeam']
        
        # Check for new teams
        for team in [home_team, away_team]:
            if team not in final_team_stats:
                print(f"Warning: No historical data for {team}")
                final_team_stats[team] = {
                    **{f"HomeTeam{feat}": [] for feat in base_features},
                    **{f"AwayTeam{feat}": [] for feat in base_features},
                    'HomeWins': [], 'AwayWins': [], 'TotalWins': [],
                    'Results': []
                }
        
        # Calculate features for prediction
        home_features = calculate_team_features(final_team_stats, home_team, is_home=True)
        away_features = calculate_team_features(final_team_stats, away_team, is_home=False)
        
        # Combine features
        all_features = home_features.values + away_features.values
        X_pred.append(all_features)
        
        prediction_matches.append({
            'Gameday': game['Gameday'],
            'HomeTeam': home_team,
            'AwayTeam': away_team
        })
    
    return X, y, np.array(X_pred), prediction_matches, feature_names