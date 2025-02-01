from itertools import cycle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import label_binarize
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from preprocessData_Training import preprocess_bundesliga_data

# Load and prepare data
print("Loading and preprocessing data...")
X, y, feature_names = preprocess_bundesliga_data('FootBall_Datafiles/Bundesliga_MatchStats.csv')

# Verify data structure
print("\nData Structure Verification:")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of feature names: {len(feature_names)}")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

print("\nSplit sizes:")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Define model parameters
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.02, 0.05],
    'max_depth': [3, 4, 5, 6, 7],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 4]
}


grid_search = GridSearchCV(
    GradientBoostingClassifier(),
    param_grid,
    cv=5,
    scoring=['accuracy'],  # Use "roc_auc_ovr" for multiclass
    verbose=1,
    n_jobs=-1
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