from pymongo import MongoClient
import os
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline # Added import
from sklearn.preprocessing import StandardScaler # Added import
import numpy as np # Import numpy for type checking
from typing import Any # Import Any for type hinting

def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively converts numpy types within a dictionary or list to native Python types.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64, np.generic)):
        return obj.item()
    elif isinstance(obj, dict):
        # Convert integer keys to strings for class_distribution
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(elem) for elem in obj]
    else:
        return obj

# Load environment variables from .env.local
load_dotenv(".env.local")

# MongoDB connection
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client["ml_database"]
students_collection = db["training_data"]

# Global variables for models, feature names, and feature importances
models = {"random_forest": {}, "xgboost": {}}
lstm_model = None # Placeholder for LSTM model
feature_names = None
feature_importances_dict = {"random_forest": {}, "xgboost": {}}
model_performance_metrics = {"random_forest": {}, "xgboost": {}}

# Define feature lists
dynamic_features = [
    "TIME_SPENT_ON_VIDEOS",
    "QUIZ_ATTEMPTS",
    "FORUM_PARTICIPATION_COUNT",
    "LOGIN_FREQUENCY_PER_WEEK",
    "DAYS_OLD"
]

numerical_static_features = [
    'GPA',
    'TIME_SPENT_ON_TEXT_MATERIALS',
    'TIME_SPENT_ON_INTERACTIVE_ACTIVITIES',
    'GROUP_ACTIVITY_PARTICIPATION',
    'INDIVIDUAL_ACTIVITY_PREFERENCE',
    'TIME_TO_COMPLETE_ASSIGNMENTS',
    'ACCURACY_IN_DETAIL_ORIENTED_QUESTIONS',
    'ACCURACY_IN_CONCEPTUAL_QUESTIONS',
    'VIDEO_PAUSE_AND_REPLAY_COUNT',
    'QUIZ_REVIEW_FREQUENCY',
    'SKIPPED_CONTENT_RATIO',
    'AVERAGE_STUDY_SESSION_LENGTH'
]

categorical_features = [
    'NOTE_TAKING_STYLE',
    'PROBLEM_SOLVING_PREFERENCE',
    'LEARNING_PATH_NAVIGATION',
    'RESPONSE_SPEED_IN_QUIZZES'
]

target_labels = [
    'ACTIVE_VS_REFLECTIVE',
    'SENSING_VS_INTUITIVE',
    'VISUAL_VS_VERBAL',
    'SEQUENTIAL_VS_GLOBAL'
]

boolean_cols = [
    'PREFERENCE_FOR_VISUAL_MATERIALS',
    'PREFERENCE_FOR_TEXTUAL_MATERIALS',
    'PREFERENCE_FOR_EXAMPLES',
    'SELF_REFLECTION_ACTIVITY'
]

# Define the preprocessor globally (for use in predict, etc.)
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor_obj = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_static_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop'
)