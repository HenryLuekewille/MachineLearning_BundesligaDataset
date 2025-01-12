from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
import time
from splitData import preprocess_bundesliga_data

class EnhancedFeatureOptimizer:
    def __init__(self, X, y, feature_names):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.best_score = 0
        self.best_features = []
        self.base_feature_count = len(feature_names)  # Remember number of base features
        
    def create_total_features(self):
        """Create additional total features based on home/away features"""
        X_enhanced = self.X.copy()
        new_feature_names = self.feature_names.copy()
        additional_features = []  # Track additional features separately
        
        feature_indices = {name: idx for idx, name in enumerate(self.feature_names)}
        
        # Create total features for specified metrics
        metrics = ['Wins', 'ShotsOnTarget', 'Shots', 'Red', 'Yellow']
        for metric in metrics:
            home_key = f'Avg_Home_{metric}'
            away_key = f'Avg_Away_{metric}'
            
            if home_key in feature_indices and away_key in feature_indices:
                home_idx = feature_indices[home_key]
                away_idx = feature_indices[away_key]
                
                # Calculate total (average of home and away)
                total = (X_enhanced[:, home_idx] + X_enhanced[:, away_idx]) / 2
                X_enhanced = np.column_stack([X_enhanced, total])
                new_name = f'Avg_Total_{metric}'
                new_feature_names.append(new_name)
                additional_features.append(new_name)
        
        # Add Total_Trend (average of Home_Trend and Away_Trend)
        if 'Home_Trend' in feature_indices and 'Away_Trend' in feature_indices:
            home_trend_idx = feature_indices['Home_Trend']
            away_trend_idx = feature_indices['Away_Trend']
            
            total_trend = (X_enhanced[:, home_trend_idx] + X_enhanced[:, away_trend_idx]) / 2
            X_enhanced = np.column_stack([X_enhanced, total_trend])
            new_feature_names.append('Avg_Total_Trend')
            additional_features.append('Avg_Total_Trend')
        
        return X_enhanced, new_feature_names, additional_features
    
    def evaluate_feature_set(self, X_subset):
        """Evaluate a feature subset using cross-validation"""
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )
        scores = cross_val_score(model, X_subset, self.y, cv=5, scoring='accuracy')
        return scores.mean()
    
    def optimize_features(self):
        """Optimize feature selection in two stages: first additional features, then base features"""
        print("Creating total features...")
        X_enhanced, enhanced_feature_names, additional_features = self.create_total_features()
        
        results = []
        
        print(f"\nStarting with {len(enhanced_feature_names)} features:")
        print("\nBase features:")
        for i, name in enumerate(self.feature_names):
            print(f"{i+1}. {name}")
        print("\nAdditional features:")
        for i, name in enumerate(additional_features):
            print(f"{i+1}. {name}")
        
        # Evaluate full feature set first
        full_score = self.evaluate_feature_set(X_enhanced)
        print(f"\nBaseline score with all features: {full_score:.4f}")
        
        self.best_score = full_score
        self.best_features = enhanced_feature_names
        
        results.append({
            'features': enhanced_feature_names,
            'n_features': len(enhanced_feature_names),
            'score': full_score,
            'removed_feature': None,
            'phase': 'initial'
        })
        
        # Phase 1: Remove additional features
        print("\nPhase 1: Testing removal of additional features...")
        current_features = list(range(X_enhanced.shape[1]))
        additional_indices = list(range(self.base_feature_count, len(enhanced_feature_names)))
        
        while additional_indices:
            best_score_this_round = 0
            feature_to_remove = None
            remaining_features_after_removal = None
            
            for idx in additional_indices:
                test_features = [i for i in current_features if i != idx]
                X_subset = X_enhanced[:, test_features]
                
                score = self.evaluate_feature_set(X_subset)
                
                results.append({
                    'features': [enhanced_feature_names[i] for i in test_features],
                    'n_features': len(test_features),
                    'score': score,
                    'removed_feature': enhanced_feature_names[idx],
                    'phase': 'additional'
                })
                
                if score > best_score_this_round:
                    best_score_this_round = score
                    feature_to_remove = idx
                    remaining_features_after_removal = test_features
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_features = [enhanced_feature_names[i] for i in test_features]
                    print(f"New best score: {score:.4f} with {len(test_features)} features")
                    print(f"Removed additional feature: {enhanced_feature_names[idx]}")
            
            if remaining_features_after_removal is None or best_score_this_round < self.best_score - 0.001:
                print("No improvement found in additional features. Moving to base features.")
                break
                
            print(f"\nRemoving additional feature: {enhanced_feature_names[feature_to_remove]}")
            current_features = remaining_features_after_removal
            additional_indices.remove(feature_to_remove)
        
        # Phase 2: Remove base features
        print("\nPhase 2: Testing removal of base features...")
        base_indices = [i for i in current_features if i < self.base_feature_count]
        
        while len(base_indices) > 5:  # Don't go below 5 features
            best_score_this_round = 0
            feature_to_remove = None
            remaining_features_after_removal = None
            
            for idx in base_indices:
                test_features = [i for i in current_features if i != idx]
                X_subset = X_enhanced[:, test_features]
                
                score = self.evaluate_feature_set(X_subset)
                
                results.append({
                    'features': [enhanced_feature_names[i] for i in test_features],
                    'n_features': len(test_features),
                    'score': score,
                    'removed_feature': enhanced_feature_names[idx],
                    'phase': 'base'
                })
                
                if score > best_score_this_round:
                    best_score_this_round = score
                    feature_to_remove = idx
                    remaining_features_after_removal = test_features
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_features = [enhanced_feature_names[i] for i in test_features]
                    print(f"New best score: {score:.4f} with {len(test_features)} features")
                    print(f"Removed base feature: {enhanced_feature_names[idx]}")
            
            if remaining_features_after_removal is None or best_score_this_round < self.best_score - 0.001:
                print("No improvement found. Stopping removal process.")
                break
                
            print(f"\nRemoving base feature: {enhanced_feature_names[feature_to_remove]}")
            current_features = remaining_features_after_removal
            base_indices.remove(feature_to_remove)
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('score', ascending=False)
        
        return {
            'best_features': self.best_features,
            'best_score': self.best_score,
            'all_results': results_df
        }

def run_optimization(training_csv_path):
    """Run the optimized feature selection process"""
    print("Loading and preprocessing data...")
    X, y, feature_names = preprocess_bundesliga_data(training_csv_path)
    
    print("\nStarting feature optimization...")
    optimizer = EnhancedFeatureOptimizer(X, y, feature_names)
    results = optimizer.optimize_features()
    
    print("\nOptimization Results:")
    print("-" * 80)
    print(f"Best accuracy: {results['best_score']:.4f}")
    
    print("\nBest feature combination:")
    for i, feature in enumerate(results['best_features'], 1):
        print(f"{i:2d}. {feature}")
    
    # Save detailed results
    results['all_results'].to_csv('feature_optimization_results.csv', index=False)
    print("\nDetailed results saved to 'feature_optimization_results.csv'")
    
    # Feature removal analysis by phase
    print("\nFeature Removal Analysis:")
    removal_analysis = results['all_results'][results['all_results']['removed_feature'].notna()]
    
    print("\nTop additional feature removals:")
    additional_removals = removal_analysis[removal_analysis['phase'] == 'additional'].sort_values('score', ascending=False)
    for _, row in additional_removals.head(6).iterrows():
        print(f"Score: {row['score']:.4f} - Removed: {row['removed_feature']}")
    
    print("\nTop base feature removals:")
    base_removals = removal_analysis[removal_analysis['phase'] == 'base'].sort_values('score', ascending=False)
    for _, row in base_removals.head(10).iterrows():
        print(f"Score: {row['score']:.4f} - Removed: {row['removed_feature']}")
    
    return results

if __name__ == "__main__":
    csv_path = 'Datafiles/Bundesliga_MatchStats.csv'
    results = run_optimization(csv_path)