import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score

from src.preprocess import AttendancePreprocessor

model = joblib.load('models/attendance_model.pkl')
preprocessor = joblib.load('models/preprocessor.joblib')

data = np.load('data/processed_data.npz')
X_test = data['X_test']
y_test = data['y_test']

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Attend', 'Attend'],
            yticklabels=['Not Attend', 'Attend'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Student Event Attendance Prediction')
plt.show()

# 2. Feature Importance 
if hasattr(model, "coef_"):
    feature_names = preprocessor.get_feature_names()
    importance = abs(model.coef_[0])
    indices = np.argsort(importance)[::-1][:15]
    plt.figure(figsize=(14,8))
    plt.barh(np.array(feature_names)[indices], importance[indices], color="skyblue")
    plt.gca().invert_yaxis()
    plt.xlabel('Coefficient Magnitude')
    plt.title('Top 15 Features - LogisticRegression')
    plt.show()
elif hasattr(model, "feature_importances_"):
    feature_names = preprocessor.get_feature_names()
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:15]
    plt.figure(figsize=(14,8))
    plt.barh(np.array(feature_names)[indices], importance[indices], color="skyblue")
    plt.gca().invert_yaxis()
    plt.xlabel('Importance')
    plt.title('Top 15 Features - RandomForest')
    plt.show()

# 3. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(10,8))
plt.plot(recall, precision, color='blue', lw=2)
plt.axhline(y=np.mean(y_test), color='red', ls='--', label=f'Baseline ({np.mean(y_test):.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Student Event Attendance Prediction')
plt.legend()
plt.show()

# 4. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)
plt.figure(figsize=(10,8))
plt.plot(fpr, tpr, color='orange', label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Student Event Attendance Prediction')
plt.legend(loc="lower right")
plt.show()
