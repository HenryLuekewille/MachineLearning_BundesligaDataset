from itertools import cycle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import label_binarize
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from splitData import preprocess_bundesliga_data
from xgboost import XGBClassifier
# Load and prepare data
X, y = preprocess_bundesliga_data('Datafiles/Bundesliga_MatchStats.csv')

# Time-based split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# GridSearch for Gradient Boosting
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}

grid_search = GridSearchCV(
    GradientBoostingClassifier(random_state=42), 
    param_grid, 
    cv=3, 
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print("\nBest parameters:", grid_search.best_params_)

# Predictions
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)

# Basic metrics
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, 
      target_names=['Home Win', 'Away Win', 'Draw']))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ROC curves for multi-class
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = 3

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

plt.figure(figsize=(10, 8))
colors = cycle(['blue', 'red', 'green'])
classes = ['Home Win', 'Away Win', 'Draw']

for i, (color, class_name) in enumerate(zip(colors, classes)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'{class_name} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Multi-class Prediction')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Feature importance visualization
feature_names = ['Home_' + str(i) for i in range(12)] + ['Away_' + str(i) for i in range(12)]
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': best_model.feature_importances_
})
importance_df = importance_df.sort_values('importance', ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()