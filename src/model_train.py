
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve, f1_score)
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.results = {}

    def load_processed_data(self):
        print("Loading preprocessed data...")

        data = np.load('data/processed_data.npz')
        self.X_train = data['X_train']
        self.X_test = data['X_test'] 
        self.y_train = data['y_train']
        self.y_test = data['y_test']

        print(f"Train shape: {self.X_train.shape}")
        print(f"Test shape: {self.X_test.shape}")
        print(f"Train class distribution: {np.bincount(self.y_train)}")
        print(f"Test class distribution: {np.bincount(self.y_test)}")

    def train_logistic_regression(self):
        print("\n=== TRAINING LOGISTIC REGRESSION ===")

        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'class_weight': [None, 'balanced'],
            'solver': ['liblinear', 'saga']
        }

        lr = LogisticRegression(random_state=self.random_state, max_iter=1000)

        print("→ Performing grid search...")
        grid_search = GridSearchCV(
            lr, param_grid, cv=5, scoring='roc_auc', 
            n_jobs=-1
        )

        grid_search.fit(self.X_train, self.y_train)

        print(f"→ Best parameters: {grid_search.best_params_}")
        print(f"→ Best CV ROC AUC: {grid_search.best_score_:.4f}")

        self.models['logistic_regression'] = grid_search.best_estimator_

        return grid_search.best_estimator_

    def train_random_forest(self):
        print("\n=== TRAINING RANDOM FOREST ===")

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'class_weight': [None, 'balanced'],
            'min_samples_split': [2, 5]
        }

        rf = RandomForestClassifier(random_state=self.random_state)

        print(" Performing grid search...")
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='roc_auc', 
            n_jobs=-1
        )

        grid_search.fit(self.X_train, self.y_train)

        print(f" Best parameters: {grid_search.best_params_}")
        print(f" Best CV ROC AUC: {grid_search.best_score_:.4f}")

        self.models['random_forest'] = grid_search.best_estimator_

        return grid_search.best_estimator_

    def evaluate_model(self, model, model_name):
        print(f"\n Evaluating {model_name}...")

        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]

        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        f1 = f1_score(self.y_test, y_pred)

        class_report = classification_report(self.y_test, y_pred, output_dict=True)

        self.results[model_name] = {
            'roc_auc': roc_auc,
            'f1_score': f1,
            'accuracy': class_report['accuracy'],
            'precision_0': class_report['0']['precision'],
            'recall_0': class_report['0']['recall'],
            'precision_1': class_report['1']['precision'], 
            'recall_1': class_report['1']['recall'],
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

        print(f"   ROC AUC: {roc_auc:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        print(f"   Accuracy: {class_report['accuracy']:.4f}")
        print(f"   Precision (Attend): {class_report['1']['precision']:.4f}")
        print(f"   Recall (Attend): {class_report['1']['recall']:.4f}")

        return self.results[model_name]

    def select_best_model(self):
        print("\n=== MODEL SELECTION ===")

        best_score = 0
        best_model_name = None

        for model_name, results in self.results.items():
            roc_auc = results['roc_auc']
            print(f"→ {model_name}: ROC AUC = {roc_auc:.4f}")

            if roc_auc > best_score:
                best_score = roc_auc
                best_model_name = model_name

        print(f"\n Best model: {best_model_name} (ROC AUC: {best_score:.4f})")

        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name

        return self.best_model, best_model_name

    def save_results(self):
        print("\n=== SAVING RESULTS ===")

        joblib.dump(self.best_model, 'models/attendance_model.pkl')
        print(" Saved best model: models/attendance_model.pkl")

        metrics_data = []
        for model_name, results in self.results.items():
            metrics_data.append({
                'model': model_name,
                'roc_auc': results['roc_auc'],
                'f1_score': results['f1_score'],
                'accuracy': results['accuracy'],
                'precision_attend': results['precision_1'],
                'recall_attend': results['recall_1'],
                'precision_not_attend': results['precision_0'],
                'recall_not_attend': results['recall_0']
            })

        metrics_df = pd.DataFrame(metrics_data)
        metrics_df = metrics_df.sort_values('roc_auc', ascending=False)
        metrics_df.to_csv('outputs/metrics_summary.csv', index=False)
        print("→ Saved metrics: outputs/metrics_summary.csv")

        print("\nFinal Model Performance Summary:")
        print(metrics_df.round(4).to_string(index=False))

        return metrics_df

    def train_all_models(self):
        print("=== STUDENT EVENT ATTENDANCE PREDICTOR - MODEL TRAINING ===\n")

        self.load_processed_data()

        lr_model = self.train_logistic_regression()
        rf_model = self.train_random_forest()

        self.evaluate_model(lr_model, 'logistic_regression')
        self.evaluate_model(rf_model, 'random_forest')

        best_model, best_model_name = self.select_best_model()

        metrics_df = self.save_results()

        print("\n=== MODEL TRAINING COMPLETE ===")

        return best_model, best_model_name, metrics_df

def main():
    trainer = ModelTrainer(random_state=42)
    best_model, best_model_name, metrics_df = trainer.train_all_models()
    return trainer

if __name__ == "__main__":
    trainer = main()
