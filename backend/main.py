from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from dotenv import load_dotenv

app = FastAPI()

# Load environment variables from .env.local
load_dotenv(".env.local")

# MongoDB connection
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client["ml_database"]
students_collection = db["training_data"]

# Global variable for the model and features
model = None
features = None

async def train_model():
    global model, features
    # Fetch data from MongoDB
    data = list(db["training_data"].find({}))
    if not data:
        print("No data found in MongoDB to train the model.")
        return

    df = pd.DataFrame(data)

    # Ensure all required columns exist
    required_columns = [
        'GPA',
        'time_spent_on_videos',
        'time_spent_on_text_materials',
        'time_spent_on_interactive_activities',
        'forum_participation_count',
        'group_activity_participation',
        'individual_activity_preference',
        'preference_for_visual_materials',
        'preference_for_textual_materials',
        'quiz_attempts',
        'time_to_complete_assignments',
        'accuracy_in_detail_oriented_questions',
        'accuracy_in_conceptual_questions',
        'preference_for_examples',
        'self_reflection_activity',
        'video_pause_and_replay_count',
        'quiz_review_frequency',
        'skipped_content_ratio',
        'login_frequency_per_week',
        'average_study_session_length',
        'active_vs_reflective' # New target variable
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column for training: {col}")

    # Convert 'extracurriculars' to numerical (e.g., 0 for No, 1 for Yes)
    # Convert boolean fields to numerical (0 or 1)
    boolean_cols = [
        'preference_for_visual_materials',
        'preference_for_textual_materials',
        'preference_for_examples',
        'self_reflection_activity'
    ]
    for col in boolean_cols:
        df[col] = df[col].apply(lambda x: 1 if x else 0)

    # Define features (X) and target (y)
    features = [
        'GPA',
        'time_spent_on_videos',
        'time_spent_on_text_materials',
        'time_spent_on_interactive_activities',
        'forum_participation_count',
        'group_activity_participation',
        'individual_activity_preference',
        'preference_for_visual_materials',
        'preference_for_textual_materials',
        'quiz_attempts',
        'time_to_complete_assignments',
        'accuracy_in_detail_oriented_questions',
        'accuracy_in_conceptual_questions',
        'preference_for_examples',
        'self_reflection_activity',
        'video_pause_and_replay_count',
        'quiz_review_frequency',
        'skipped_content_ratio',
        'login_frequency_per_week',
        'average_study_session_length'
    ]
    target = 'active_vs_reflective'

    X = df[features]
    y = df[target]

    # Handle potential non-numeric values gracefully by converting to numeric, coercing errors
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    # Drop rows with NaN values that resulted from coercion
    X.dropna(inplace=True)
    y.dropna(inplace=True)

    # Ensure X and y have the same number of samples after dropping NaNs
    if X.shape[0] == 0 or X.shape[0] != y.shape[0]:
        print("Not enough valid data after cleaning to train the model.")
        return

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForestClassifier model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model trained successfully.")

@app.on_event("startup")
async def startup_event():
    await train_model()

@app.get("/")
async def read_root():
    return {"message": "FastAPI backend is running"}

@app.get("/db-status")
async def get_db_status():
    try:
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
        
        # Check if the students collection exists and get document count
        collection_exists = "training_data" in db.list_collection_names()
        student_count = 0
        if collection_exists:
            student_count = students_collection.count_documents({})
            
        return {
            "status": "MongoDB connection successful",
            "database": "ml_database",
            "collection": "training_data",
            "collection_exists": collection_exists,
            "student_count": student_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MongoDB connection error: {e}")

@app.post("/predict-learning-style")
async def predict_learning_style(student_data: dict):
    if model is None or features is None:
        raise HTTPException(status_code=500, detail="Model not trained yet.")

    try:
        # Prepare input data for prediction
        input_df = pd.DataFrame([student_data])

        # Ensure all required features are present and in the correct order
        for col in features:
            if col not in input_df.columns:
                raise HTTPException(status_code=400, detail=f"Missing feature: {col}")
            # Convert boolean fields to numerical if present
            if col in boolean_cols:
                input_df[col] = input_df[col].apply(lambda x: 1 if x else 0)
            # Convert to numeric, coercing errors
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        # Drop rows with NaN values that resulted from coercion
        input_df.dropna(inplace=True)

        if input_df.empty:
            raise HTTPException(status_code=400, detail="Invalid input data after cleaning.")

        # Make prediction
        prediction = model.predict(input_df[features])
        return {"active_vs_reflective_prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))