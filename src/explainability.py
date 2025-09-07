
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score
import warnings
from src.preprocess import AttendancePreprocessor
warnings.filterwarnings('ignore')

class ModelExplainer:
    

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None

    def load_models_and_data(self):
        
        print("Loading models and data...")

        
        self.model = joblib.load('models/attendance_model.pkl')
        self.preprocessor = joblib.load('models/preprocessor.joblib')

        
        data = np.load('data/processed_data.npz')
        self.X_test = data['X_test']
        self.y_test = data['y_test']

        
        self.feature_names = self.preprocessor.get_feature_names()

        print(f"Model type: {type(self.model).__name__}")
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Test samples: {len(self.y_test)}")

    def create_confusion_matrix_plot(self):
        
        print("Creating confusion matrix...")

        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Attend', 'Attend'],
                   yticklabels=['Not Attend', 'Attend'])
        plt.title('Confusion Matrix - Student Event Attendance Prediction')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('outputs/figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        return cm

    def create_roc_curve_plot(self):
        
        print("→ Creating ROC curve...")

        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Student Event Attendance Prediction')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/figures/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

        return roc_auc

    def create_precision_recall_plot(self):
        
        print("Creating precision-recall curve...")

        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - Student Event Attendance Prediction')
        plt.grid(True, alpha=0.3)

        
        baseline = self.y_test.mean()
        plt.axhline(y=baseline, color='red', linestyle='--', alpha=0.8, 
                   label=f'Baseline ({baseline:.3f})')
        plt.legend()
        plt.tight_layout()
        plt.savefig('outputs/figures/precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_feature_importance(self):
        
        print(" Analyzing feature importance...")

        model_type = type(self.model).__name__

        if model_type == 'LogisticRegression':
            
            feature_importance = abs(self.model.coef_[0])
            importance_type = 'Coefficient Magnitude'
        elif model_type == 'RandomForestClassifier':
            feature_importance = self.model.feature_importances_
            importance_type = 'Feature Importance'
        else:
            print(f"Feature importance not implemented for {model_type}")
            return None

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        top_features = importance_df.head(15)

        print(f"\nTop 15 Most Important Features ({importance_type}):")
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<25} {row['importance']:.4f}")

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_features['importance'], color='skyblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel(importance_type)
        plt.title(f'Top 15 Features - {model_type}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('outputs/figures/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

        importance_df.to_csv('outputs/feature_importance.csv', index=False)

        return importance_df

    def create_model_interpretation(self):
        print(" Creating model interpretation...")

        # Load feature importance
        importance_df = pd.read_csv('outputs/feature_importance.csv')
        top_5_positive = importance_df.head(5)

        interpretation = f"""
## Model Interpretation - Student Event Attendance Predictor

**Model Type**: {type(self.model).__name__}
**Performance**: ROC AUC = {roc_auc_score(self.y_test, self.model.predict_proba(self.X_test)[:, 1]):.3f}

### Top 5 Most Important Factors for Attendance Prediction:

"""

        for i, (_, row) in enumerate(top_5_positive.iterrows(), 1):
            feature = row['feature']
            importance = row['importance']

            # Interpret feature names
            if feature.startswith('notification_received'):
                description = "Whether the student received event notification"
            elif feature.startswith('tag_match_count'):
                description = "Number of matching interests between student and event"
            elif feature.startswith('past_attendance_count'):
                description = "Student's historical event attendance frequency"
            elif feature.startswith('distance_km'):
                description = "Physical distance from student location to event venue"
            elif feature.startswith('interest_') or feature.startswith('event_tag_'):
                tag_name = feature.split('_', 1)[1]
                description = f"Interest/Event tag: {tag_name}"
            elif feature.startswith('event_type_'):
                event_type = feature.split('_', 1)[1]
                description = f"Event type: {event_type}"
            else:
                description = feature.replace('_', ' ').title()

            interpretation += f"{i}. **{description}** (Importance: {importance:.4f})\n"

        interpretation += """
### Key Insights:

- **Notifications are crucial**: Students who receive notifications are much more likely to attend
- **Interest matching matters**: Events aligned with student interests have higher attendance
- **Past behavior predicts future**: Students with high past attendance tend to continue attending
- **Distance is a barrier**: Closer events have significantly higher attendance rates
- **Event type influences attendance**: Some event types (workshops, hackathons) are more popular

### Recommendations for Event Organizers:

1. **Prioritize notifications**: Ensure all eligible students receive event notifications
2. **Target relevant audiences**: Match events to student interests and departments
3. **Location planning**: Choose accessible venues to minimize travel distance
4. **Engage past attendees**: Previous attendees are your most reliable audience
5. **Event format**: Consider popular formats like workshops and hands-on sessions
"""

        with open('outputs/model_interpretation.md', 'w') as f:
            f.write(interpretation)

        print("   Saved model interpretation to outputs/model_interpretation.md")

    def generate_all_explanations(self):
        print("=== STUDENT EVENT ATTENDANCE PREDICTOR - MODEL EXPLAINABILITY ===\n")

        self.load_models_and_data()

        print("\nCreating evaluation plots...")
        self.create_confusion_matrix_plot()
        self.create_roc_curve_plot() 
        self.create_precision_recall_plot()

        print("\nAnalyzing feature importance...")
        self.analyze_feature_importance()

        self.create_model_interpretation()

        print("\n=== MODEL EXPLAINABILITY COMPLETE ===")
        print("Generated files:")
        print(" outputs/figures/confusion_matrix.png")
        print(" outputs/figures/roc_curve.png") 
        print(" outputs/figures/precision_recall_curve.png")
        print(" outputs/figures/feature_importance.png")
        print(" outputs/feature_importance.csv")
        print(" outputs/model_interpretation.md")

def main():
    explainer = ModelExplainer()
    explainer.generate_all_explanations()
    return explainer

if __name__ == "__main__":
    explainer = main()
