from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import random
import pandas as pd
import numpy as np
from datetime import datetime

from backend.dependencies import models, numerical_features, categorical_features, boolean_cols, db, preprocessor_obj

router = APIRouter()

class SimulateRequest(BaseModel):
    num_students: int

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

def generate_realistic_student_data(num_students: int) -> List[Dict[str, Any]]:
    # Placeholder for realistic data generation
    # This will be replaced with more sophisticated logic later
    simulated_data = []
    for i in range(num_students):
        student = {
            "STUDENT_ID": f"sim_student_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{i}",
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

    print(f"Simulating {request.num_students} students...")
    simulated_raw_data = generate_realistic_student_data(request.num_students)
    
    predictions_list = []
    
    for student_data in simulated_raw_data:
        input_df = pd.DataFrame([student_data])
        # Convert boolean fields to numerical (0 or 1)
        for col in boolean_cols:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(int)

        # Ensure numerical columns are numeric before passing to preprocessor
        for col in numerical_features:
            if col in input_df.columns:
                input_df.loc[:, col] = pd.to_numeric(input_df[col], errors='coerce')
            else:
                input_df[col] = np.nan # Ensure missing numerical features are NaN for imputer

        # For categorical features, ensure they are present or default to a value
        for col in categorical_features:
            if col not in input_df.columns:
                input_df[col] = 'unknown' # Default for missing categorical features
            else:
                input_df[col] = input_df[col].astype(str).fillna('missing_category')

        student_predictions = {}
        for target, model_pipeline in models.items():
            # Select only the columns that the preprocessor expects
            # Ensure the order of columns is consistent with training data (preprocessor handles this)
            # Ensure all numerical and categorical features are present.
            # If any are missing in input_df, add them with NaN or 'unknown' as appropriate.
            all_expected_features = [f.upper() for f in numerical_features] + [f.upper() for f in categorical_features]
            for feature in all_expected_features:
                if feature not in input_df.columns:
                    if feature in [f.upper() for f in numerical_features]:
                        input_df[feature] = np.nan
                    else:
                        input_df[feature] = 'unknown'

            input_for_prediction = input_df[[f.upper() for f in numerical_features] + [f.upper() for f in categorical_features]]
            
            print(f"DEBUG: input_df columns before prediction: {input_df.columns.tolist()}")
            print(f"DEBUG: input_for_prediction shape: {input_for_prediction.shape}")
            print(f"DEBUG: input_for_prediction head:\n{input_for_prediction.head()}")

            prediction_result = model_pipeline.predict(input_for_prediction)
            
            # Assuming the target labels are in the same order as when the model was trained
            # This part needs to be improved if the original string labels are desired
            # For now, just return the numerical prediction
            student_predictions[target] = prediction_result[0].item()

        # Add predictions to the student data
        student_data.update(student_predictions)
        # Convert all keys to lowercase before appending
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
