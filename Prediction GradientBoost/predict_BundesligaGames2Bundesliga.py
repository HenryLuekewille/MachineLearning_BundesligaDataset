from itertools import cycle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import label_binarize
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from preprocess_Data_Prediction import preprocess_bundesliga_data

# Load and prepare all historical data and future matches
print("Loading and preprocessing data...")
X, y, X_pred, prediction_matches, feature_names = preprocess_bundesliga_data(
    'FootBall_Datafiles/2Bundesliga_MatchStats.csv',  # Historical data
    'FootBall_Datafiles/2Bundesliga_GamePlan.csv'         # Future matches
)

print("\nData Structure:")
print(f"Total matches in training data: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")

# Define model parameters
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.03, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 3],
    'min_samples_leaf': [1, 2]
}

# Train model on all historical data
print("\nTraining model on complete historical dataset...")
grid_search = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X, y)

best_model = grid_search.best_estimator_
print("\nBest parameters:", grid_search.best_params_)

# Make predictions for future games
y_pred = best_model.predict(X_pred)
y_pred_proba = best_model.predict_proba(X_pred)

# Create detailed predictions DataFrame
predictions_df = pd.DataFrame([
    {
        'Gameday': match['Gameday'],
        'HomeTeam': match['HomeTeam'],
        'AwayTeam': match['AwayTeam'],
        'HomeWin_Prob': probs[0],
        'Draw_Prob': probs[2],
        'AwayWin_Prob': probs[1],
        'Predicted_Result': ['H', 'A', 'D'][pred],
        'Confidence': max(probs)
    }
    for match, probs, pred in zip(prediction_matches, y_pred_proba, y_pred)
])

# Sort predictions by gameday and confidence
predictions_df = predictions_df.sort_values(['Gameday', 'Confidence'], ascending=[True, False])

# Print predictions
print("\nPredictions for future games:")
print("-" * 100)
print(f"{'Gameday':^8} {'Home Team':^25} {'Away Team':^25} {'HW':^8} {'D':^8} {'AW':^8} {'Pred':^8}")
print("-" * 100)

for _, row in predictions_df.iterrows():
    print(f"{row['Gameday']:^8} {row['HomeTeam']:^25} {row['AwayTeam']:^25} "
          f"{row['HomeWin_Prob']:^8.3f} {row['Draw_Prob']:^8.3f} "
          f"{row['AwayWin_Prob']:^8.3f} {row['Predicted_Result']:^8}")

# Save predictions to CSV
predictions_df.to_csv('predictions_2Bundesliga.csv', index=False)
print("\nPredictions saved to 'predictions_2Bundesliga.csv'")

# Feature importance analysis
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': best_model.feature_importances_
})
importance_df = importance_df.sort_values('importance', ascending=False)

# Print feature importances
print("\nTop 10 Most Important Features:")
print("-" * 60)
print(importance_df.head(10).to_string(index=False))

# Summary statistics
print("\nPrediction Summary:")
print(f"Average prediction confidence: {predictions_df['Confidence'].mean():.3f}")
print(f"Home wins predicted: {(predictions_df['Predicted_Result'] == 'H').sum()}")
print(f"Away wins predicted: {(predictions_df['Predicted_Result'] == 'A').sum()}")
print(f"Draws predicted: {(predictions_df['Predicted_Result'] == 'D').sum()}")

# Visualize feature importances (top 15)
plt.figure(figsize=(12, 8))
importance_plot_data = importance_df.head(15)
plt.barh(importance_plot_data['feature'], importance_plot_data['importance'])
plt.title("Top 15 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()