
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class AttendancePreprocessor:

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.tag_vocabulary = None
        self.column_transformer = None
        self.categorical_columns = ['dept', 'event_type', 'day_of_week', 'time_of_day', 'register_channel']
        self.numerical_columns = ['year', 'past_attendance_count', 'distance_km', 'notification_received']

    def extract_tags(self, tag_string):
        if pd.isna(tag_string):
            return []
        return [tag.strip() for tag in tag_string.split(',')]

    def create_tag_vocabulary(self, interests_series, event_tags_series):
        all_tags = set()

        for interests in interests_series.dropna():
            all_tags.update(self.extract_tags(interests))

        for event_tags in event_tags_series.dropna():
            all_tags.update(self.extract_tags(event_tags))

        self.tag_vocabulary = sorted(list(all_tags))
        return self.tag_vocabulary

    def create_tag_features(self, interests_series, event_tags_series):
        n_samples = len(interests_series)

        interest_features = np.zeros((n_samples, len(self.tag_vocabulary)))
        for i, interests in enumerate(interests_series):
            if pd.notna(interests):
                user_tags = self.extract_tags(interests)
                for tag in user_tags:
                    if tag in self.tag_vocabulary:
                        tag_idx = self.tag_vocabulary.index(tag)
                        interest_features[i, tag_idx] = 1

        event_tag_features = np.zeros((n_samples, len(self.tag_vocabulary)))
        for i, event_tags in enumerate(event_tags_series):
            if pd.notna(event_tags):
                event_tag_list = self.extract_tags(event_tags)
                for tag in event_tag_list:
                    if tag in self.tag_vocabulary:
                        tag_idx = self.tag_vocabulary.index(tag)
                        event_tag_features[i, tag_idx] = 1

        interest_columns = [f'interest_{tag}' for tag in self.tag_vocabulary]
        event_tag_columns = [f'event_tag_{tag}' for tag in self.tag_vocabulary]

        tag_match_count = np.sum(interest_features * event_tag_features, axis=1)

        return interest_features, event_tag_features, interest_columns, event_tag_columns, tag_match_count

    def fit_transform(self, df):
        print("Fitting preprocessor...")

        print(" Creating tag vocabulary...")
        self.create_tag_vocabulary(df['interests'], df['event_tags'])
        print(f"  Found {len(self.tag_vocabulary)} unique tags: {self.tag_vocabulary}")

        print(" Creating tag features...")
        interest_features, event_tag_features, interest_cols, event_tag_cols, tag_match_count = \
            self.create_tag_features(df['interests'], df['event_tags'])

        feature_df = df[self.categorical_columns + self.numerical_columns].copy()

        feature_df['tag_match_count'] = tag_match_count

        for i, col in enumerate(interest_cols):
            feature_df[col] = interest_features[:, i]
        for i, col in enumerate(event_tag_cols):
            feature_df[col] = event_tag_features[:, i]

        categorical_features = self.categorical_columns
        numerical_features = self.numerical_columns + ['tag_match_count']
        binary_features = interest_cols + event_tag_cols  

        self.column_transformer = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
                ('num', StandardScaler(), numerical_features),
                ('binary', 'passthrough', binary_features)
            ],
            remainder='drop'
        )

        print(" Applying categorical encoding and scaling...")
        X_processed = self.column_transformer.fit_transform(feature_df)

        cat_feature_names = self.column_transformer.named_transformers_['cat'].get_feature_names_out(categorical_features)
        num_feature_names = numerical_features
        binary_feature_names = binary_features

        self.feature_names = list(cat_feature_names) + num_feature_names + binary_feature_names

        print(f" Final feature count: {X_processed.shape[1]}")
        print(f" Sample features: {self.feature_names[:10]}...")

        return X_processed

    def transform(self, df):
        if self.column_transformer is None:
            raise ValueError("Preprocessor must be fitted before transform")

        print("Transforming new data...")

        interest_features, event_tag_features, interest_cols, event_tag_cols, tag_match_count = \
            self.create_tag_features(df['interests'], df['event_tags'])

        feature_df = df[self.categorical_columns + self.numerical_columns].copy()

        feature_df['tag_match_count'] = tag_match_count

        for i, col in enumerate(interest_cols):
            feature_df[col] = interest_features[:, i]
        for i, col in enumerate(event_tag_cols):
            feature_df[col] = event_tag_features[:, i]

        X_processed = self.column_transformer.transform(feature_df)

        return X_processed

    def get_feature_names(self):
        return self.feature_names if hasattr(self, 'feature_names') else None

def prepare_data(data_path='data/attendance.csv', test_size=0.2, random_state=42):

    print("=== STUDENT EVENT ATTENDANCE PREDICTOR - DATA PREPARATION ===\n")

    print("1. Loading data...")
    df = pd.read_csv(data_path)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Attendance rate: {df['attend'].mean():.2%}")

    print("\n2. Checking data quality...")
    missing_values = df.isnull().sum().sum()
    print(f"   Missing values: {missing_values}")

    if missing_values > 0:
        print("   Found missing values:")
        print(df.isnull().sum()[df.isnull().sum() > 0])

    print("\n3. Separating features and target...")
    X = df.drop('attend', axis=1)
    y = df['attend']

    print(f"   Features shape: {X.shape}")
    print(f"   Target distribution: {y.value_counts().to_dict()}")

    print("\n4. Creating train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"   Train shape: {X_train.shape}")
    print(f"   Test shape: {X_test.shape}")
    print(f"   Train attendance rate: {y_train.mean():.2%}")
    print(f"   Test attendance rate: {y_test.mean():.2%}")

    print("\n5. Preprocessing features...")
    preprocessor = AttendancePreprocessor(random_state=random_state)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"   Processed train shape: {X_train_processed.shape}")
    print(f"   Processed test shape: {X_test_processed.shape}")

    print("\n6. Saving preprocessor...")
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    print("    Saved models/preprocessor.joblib")

    print("\n7. Saving processed data...")
    np.savez('data/processed_data.npz', 
             X_train=X_train_processed, X_test=X_test_processed, 
             y_train=y_train, y_test=y_test)
    print("    Saved data/processed_data.npz")

    print("\n8. Sample of processed features:")
    feature_names = preprocessor.get_feature_names()
    if feature_names:
        print(f"   First 10 features: {feature_names[:10]}")
        print(f"   Last 10 features: {feature_names[-10:]}")

    print("\n=== DATA PREPARATION COMPLETE ===")

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()
