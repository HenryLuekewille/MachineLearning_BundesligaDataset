from itertools import cycle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import label_binarize
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from splitData import preprocess_bundesliga_data

# Load and prepare data
print("Loading and preprocessing data...")
X, y, X_pred, prediction_matches, feature_names = preprocess_bundesliga_data(
    'Datafiles/Bundesliga_MatchStats.csv',
    'Datafiles/gameplan_24_25.csv'
)

# Verify data structure
print("\nData Structure Verification:")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of feature names: {len(feature_names)}")

# Time-based split for historical data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print("\nSplit sizes:")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Expanded GridSearch parameters to account for new feature structure
param_grid = {
    'n_estimators': [100, 200, 300],  # Added 300
    'learning_rate': [0.03, 0.05, 0.1],  # Added 0.03
    'max_depth': [3, 4, 5],  # Added 4
    'min_samples_split': [2, 3],  # New parameter
    'min_samples_leaf': [1, 2]  # New parameter
}

print("\nTraining model...")
grid_search = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid,
    cv=5,  # Increased from 3 to 5 for more robust validation
    scoring='accuracy',
    verbose=1,
    n_jobs=-1  # Use all available cores
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("\nBest parameters:", grid_search.best_params_)

# Model evaluation
print("\nModel Evaluation:")
y_test_pred = best_model.predict(X_test)
y_test_pred_proba = best_model.predict_proba(X_test)

print("Test Set Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred,
    target_names=['Home Win', 'Away Win', 'Draw']))

# Confusion Matrix visualization
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(3)
plt.xticks(tick_marks, ['Home Win', 'Away Win', 'Draw'])
plt.yticks(tick_marks, ['Home Win', 'Away Win', 'Draw'])
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

# Add text annotations to confusion matrix
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.show()

# ROC curves
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = 3

plt.figure(figsize=(10, 8))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_test_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'ROC curve of class {i} (AUC = {roc_auc:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Multi-class Prediction')
plt.legend(loc="lower right")
plt.show()

# Predictions for future games
y_pred = best_model.predict(X_pred)
y_pred_proba = best_model.predict_proba(X_pred)

print("\nPredictions for future games:")
print("-" * 100)
print(f"{'Gameday':^8} {'Home Team':^25} {'Away Team':^25} {'HW':^8} {'D':^8} {'AW':^8}")
print("-" * 100)

for match, probs in zip(prediction_matches, y_pred_proba):
    print(f"{match['Gameday']:^8} {match['HomeTeam']:^25} {match['AwayTeam']:^25} "
          f"{probs[0]:^8.3f} {probs[2]:^8.3f} {probs[1]:^8.3f}")

# Feature importance analysis
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': best_model.feature_importances_
})
importance_df = importance_df.sort_values('importance', ascending=True)

# Visualization of feature importances
plt.figure(figsize=(12, 10))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# Detailed feature importance output
print("\nDetailed Feature Importances:")
for feature, importance in zip(importance_df['feature'], importance_df['importance']):
    print(f"{feature:<40} {importance:.4f}")

# Enhanced predictions output with confidence metrics
predictions_df = pd.DataFrame([
    {
        'Gameday': match['Gameday'],
        'HomeTeam': match['HomeTeam'],
        'AwayTeam': match['AwayTeam'],
        'HomeWin_Prob': probs[0],
        'Draw_Prob': probs[2],
        'AwayWin_Prob': probs[1],
        'Predicted_Result': ['H', 'A', 'D'][pred],
        'Confidence': max(probs)  # Added confidence metric
    }
    for match, probs, pred in zip(prediction_matches, y_pred_proba, y_pred)
])

# Sort predictions by confidence
predictions_df = predictions_df.sort_values('Confidence', ascending=False)

# Save predictions to CSV
predictions_df.to_csv('predictions.csv', index=False)
print("\nPredictions saved to 'predictions.csv'")

# Summary statistics
print("\nPrediction Summary:")
print(f"Average prediction confidence: {predictions_df['Confidence'].mean():.3f}")
print(f"Home wins predicted: {(predictions_df['Predicted_Result'] == 'H').sum()}")
print(f"Away wins predicted: {(predictions_df['Predicted_Result'] == 'A').sum()}")
print(f"Draws predicted: {(predictions_df['Predicted_Result'] == 'D').sum()}")