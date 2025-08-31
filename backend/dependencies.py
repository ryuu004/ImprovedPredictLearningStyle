from pymongo import MongoClient
import os
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Load environment variables from .env.local
load_dotenv(".env.local")

# MongoDB connection
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client["ml_database"]
students_collection = db["training_data"]

# Global variables for models, feature names, and feature importances
models = {}
feature_names = None
feature_importances_dict = {}
model_performance_metrics = {}
categorical_features = [
    'NOTE_TAKING_STYLE',
    'PROBLEM_SOLVING_PREFERENCE'
]
target_labels = [
    'ACTIVE_VS_REFLECTIVE',
    'SENSING_VS_INTUITIVE',
    'VISUAL_VS_VERBAL',
    'SEQUENTIAL_VS_GLOBAL'
]
boolean_cols = []
numerical_features = [
    'GPA',
    'TIME_SPENT_ON_VIDEOS',
    'TIME_SPENT_ON_TEXT_MATERIALS',
    'TIME_SPENT_ON_INTERACTIVE_ACTIVITIES',
    'FORUM_PARTICIPATION_COUNT',
    'GROUP_ACTIVITY_PARTICIPATION',
    'INDIVIDUAL_ACTIVITY_PREFERENCE',
    'QUIZ_ATTEMPTS',
    'TIME_TO_COMPLETE_ASSIGNMENTS',
    'ACCURACY_IN_DETAIL_ORIENTED_QUESTIONS',
    'ACCURACY_IN_CONCEPTUAL_QUESTIONS',
    'VIDEO_PAUSE_AND_REPLAY_COUNT',
    'QUIZ_REVIEW_FREQUENCY',
    'SKIPPED_CONTENT_RATIO',
    'LOGIN_FREQUENCY_PER_WEEK',
    'AVERAGE_STUDY_SESSION_LENGTH'
]

# Define the preprocessor globally