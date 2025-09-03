from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import random
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.pipeline import Pipeline # Import Pipeline
from sklearn.preprocessing import StandardScaler # Import StandardScaler
from sklearn.impute import SimpleImputer # Import SimpleImputer
from sklearn.compose import ColumnTransformer # Import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder # Import OneHotEncoder

from backend.dependencies import models, numerical_features, categorical_features, boolean_cols, db, preprocessor_obj, lstm_model
from sklearn.preprocessing import OneHotEncoder
import numpy as np # Ensure numpy is imported

router = APIRouter()

class SimulateRequest(BaseModel):
    num_students: int
    days_old: int = 0 # New field for days old

class SimulatedStudent(BaseModel):
    # This model should reflect the structure of StudentLearningData from src/lib/db.js
    # but without the target variables as they will be predicted.
    student_id: str
    age: int
    gender: str
    academic_program: str
    year_level: str
    GPA: float
    time_spent_on_videos: int
    time_spent_on_text_materials: int
    time_spent_on_interactive_activities: int
    forum_participation_count: int
    group_activity_participation: int
    individual_activity_preference: int
    note_taking_style: str
    preference_for_visual_materials: bool
    preference_for_textual_materials: bool
    quiz_attempts: int
    time_to_complete_assignments: int
    learning_path_navigation: str
    problem_solving_preference: str
    response_speed_in_quizzes: str
    accuracy_in_detail_oriented_questions: int
    accuracy_in_conceptual_questions: int
    preference_for_examples: bool
    self_reflection_activity: bool
    clickstream_sequence: List[str]
    video_pause_and_replay_count: int
    quiz_review_frequency: int
    skipped_content_ratio: float
    login_frequency_per_week: int
    average_study_session_length: int
    # Predicted learning styles will be added dynamically

def generate_realistic_student_data(num_students: int, days_old: int) -> List[Dict[str, Any]]:
    # Placeholder for realistic data generation
    # This will be replaced with more sophisticated logic later
    simulated_data = []
    for i in range(num_students):
        student = {
            "STUDENT_ID": f"sim_student_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{i}",
            "DAYS_OLD": 0, # Initialize days_old to 0
            "AGE": random.randint(18, 25),
            "GENDER": random.choice(["Male", "Female", "Other"]),
            "ACADEMIC_PROGRAM": random.choice(["IT", "Engineering", "Business", "Arts", "Science"]),
            "YEAR_LEVEL": random.choice(["1st Year", "2nd Year", "3rd Year", "4th Year"]),
            "GPA": round(random.uniform(2.0, 4.0), 2),
            "TIME_SPENT_ON_VIDEOS": random.randint(0, 20),
            "TIME_SPENT_ON_TEXT_MATERIALS": random.randint(0, 20),
            "TIME_SPENT_ON_INTERACTIVE_ACTIVITIES": random.randint(0, 15),
            "FORUM_PARTICIPATION_COUNT": random.randint(0, 10),
            "GROUP_ACTIVITY_PARTICIPATION": random.randint(0, 5),
            "INDIVIDUAL_ACTIVITY_PREFERENCE": random.randint(0, 10), # Assuming a scale
            "NOTE_TAKING_STYLE": random.choice(["typed", "handwritten", "none"]),
            "PREFERENCE_FOR_VISUAL_MATERIALS": random.choice([True, False]),
            "PREFERENCE_FOR_TEXTUAL_MATERIALS": random.choice([True, False]),
            "QUIZ_ATTEMPTS": random.randint(1, 10),
            "TIME_TO_COMPLETE_ASSIGNMENTS": random.randint(30, 180), # in minutes
            "LEARNING_PATH_NAVIGATION": random.choice(["linear", "jumping around modules"]),
            "PROBLEM_SOLVING_PREFERENCE": random.choice(["step-by-step", "holistic solution approach"]),
            "RESPONSE_SPEED_IN_QUIZZES": random.choice(["fast", "slow", "moderate"]),
            "ACCURACY_IN_DETAIL_ORIENTED_QUESTIONS": random.randint(50, 100),
            "ACCURACY_IN_CONCEPTUAL_QUESTIONS": random.randint(50, 100),
            "PREFERENCE_FOR_EXAMPLES": random.choice([True, False]),
            "SELF_REFLECTION_ACTIVITY": random.choice([True, False]),
            "CLICKSTREAM_SEQUENCE": [], # This might be complex to simulate realistically
            "VIDEO_PAUSE_AND_REPLAY_COUNT": random.randint(0, 10),
            "QUIZ_REVIEW_FREQUENCY": random.randint(0, 5),
            "SKIPPED_CONTENT_RATIO": round(random.uniform(0.0, 0.5), 2),
            "LOGIN_FREQUENCY_PER_WEEK": random.randint(1, 7),
            "AVERAGE_STUDY_SESSION_LENGTH": random.randint(30, 120), # in minutes
        }
        simulated_data.append(student)
    return simulated_data

@router.post("/simulate-students")
async def simulate_students_endpoint(request: SimulateRequest):
    if not models:
        raise HTTPException(status_code=500, detail="AI Models not loaded. Cannot simulate learning styles.")

    print(f"Simulating {request.num_students} students with {request.days_old} days old...")
    simulated_raw_data = generate_realistic_student_data(request.num_students, request.days_old)

    predictions_list = []

    # Define the exact list of features the preprocessor expects, in the correct order.
    # These lists are imported from backend.dependencies.
    all_expected_preprocessor_cols = [f.upper() for f in numerical_features] + \
                                     [f.upper() for f in categorical_features]

    # Define dynamic and static features based on the problem description
    # This is a simplified representation. In a real scenario, these would be derived from
    # the actual features used in your LSTM and RF models.
    dynamic_features = [
        "TIME_SPENT_ON_VIDEOS",
        "QUIZ_ATTEMPTS",
        "FORUM_PARTICIPATION_COUNT",
        "LOGIN_FREQUENCY_PER_WEEK",
        "DAYS_OLD" # Assuming days_old influences temporal patterns
    ]
    static_features = [
        "AGE",
        "GENDER",
        "ACADEMIC_PROGRAM",
        "YEAR_LEVEL",
        "GPA",
        "TIME_SPENT_ON_TEXT_MATERIALS",
        "TIME_SPENT_ON_INTERACTIVE_ACTIVITIES",
        "GROUP_ACTIVITY_PARTICIPATION",
        "INDIVIDUAL_ACTIVITY_PREFERENCE",
        "NOTE_TAKING_STYLE",
        "PREFERENCE_FOR_VISUAL_MATERIALS",
        "PREFERENCE_FOR_TEXTUAL_MATERIALS",
        "TIME_TO_COMPLETE_ASSIGNMENTS",
        "LEARNING_PATH_NAVIGATION",
        "PROBLEM_SOLVING_PREFERENCE",
        "RESPONSE_SPEED_IN_QUIZZES",
        "ACCURACY_IN_DETAIL_ORIENTED_QUESTIONS",
        "ACCURACY_IN_CONCEPTUAL_QUESTIONS",
        "PREFERENCE_FOR_EXAMPLES",
        "SELF_REFLECTION_ACTIVITY",
        "VIDEO_PAUSE_AND_REPLAY_COUNT",
        "QUIZ_REVIEW_FREQUENCY",
        "SKIPPED_CONTENT_RATIO",
        "AVERAGE_STUDY_SESSION_LENGTH"
    ]


    for student_data in simulated_raw_data:
        # Ensure 'DAYS_OLD' is correctly set from the request
        student_data["DAYS_OLD"] = request.days_old

        # Create a DataFrame with all expected columns and populate it with student_data.
        # This ensures all columns the preprocessor expects are present, even if
        # generate_realistic_student_data didn't explicitly provide them (they'll be NaN).
        input_df = pd.DataFrame([student_data]).reindex(columns=all_expected_preprocessor_cols, fill_value=np.nan)

        # Convert boolean fields to numerical (0 or 1)
        for col in boolean_cols:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(int)

        student_predictions = {}
        for target, model_pipeline in models.items():
            # Pass the raw input_df to the model_pipeline.predict().
            # The pipeline, now including LSTMFeatureExtractor and ColumnTransformer,
            # will handle all preprocessing and feature engineering internally.
            prediction_result = model_pipeline.predict(input_df)

            student_predictions[target] = prediction_result[0].item()

        # Add predictions and the days_old to the student data
        student_data.update(student_predictions)
        predictions_list.append({k.lower(): v for k, v in student_data.items()})

    # Save to MongoDB
    try:
        simulated_students_collection = db["simulated_students"]
        if predictions_list:
            # When saving, convert keys back to uppercase for consistency with model training data
            # and to allow predictions to be stored with original model target names.
            # No, actually, save them as lowercase so they match the frontend's expectations.
            # The model expects uppercase, but the DB can store lowercase and we'll convert on retrieve.
            # This is simpler than converting back and forth.
            result = simulated_students_collection.insert_many(predictions_list)
            print(f"Inserted {len(result.inserted_ids)} simulated students into MongoDB.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save simulated students to database: {e}")

    # Convert ObjectId to string for JSON serialization before returning
    # The keys are already lowercase now.
    for student in predictions_list:
        if "_id" in student:
            student["_id"] = str(student["_id"])
    return {"message": "Students simulated and saved successfully", "simulated_students": predictions_list}

@router.delete("/delete-simulated-students")
async def delete_simulated_students_endpoint(student_ids: List[str]):
    try:
        simulated_students_collection = db["simulated_students"]
        result = simulated_students_collection.delete_many({"student_id": {"$in": student_ids}})
        return {"message": f"Deleted {result.deleted_count} simulated students."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete simulated students: {e}")

@router.delete("/delete-all-simulated-students")
async def delete_all_simulated_students_endpoint():
    try:
        simulated_students_collection = db["simulated_students"]
        result = simulated_students_collection.delete_many({})
        return {"message": f"Deleted all {result.deleted_count} simulated students."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete all simulated students: {e}")

@router.get("/get-simulated-students")
async def get_simulated_students_endpoint():
    try:
        simulated_students_collection = db["simulated_students"]
        students = simulated_students_collection.find() # find() returns a cursor
        students_list = list(students) # Convert cursor to list
        # Convert ObjectId to string for JSON serialization
        # Convert all keys to lowercase for frontend consumption
        lowercase_students_list = []
        for student in students_list:
            lowercase_student = {k.lower(): v for k, v in student.items()}
            lowercase_student["_id_str"] = str(student["_id"]) # Add _id_str
            lowercase_students_list.append(lowercase_student)
        return lowercase_students_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve simulated students: {e}")
