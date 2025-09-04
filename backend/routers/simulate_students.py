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

class UpdateDaysOldRequest(BaseModel):
    student_ids: List[str]
    days_to_add: int

class SimulatedStudent(BaseModel):
    # This model should reflect the structure of StudentLearningData from src/lib/db.js
    # but without the target variables as they will be predicted.
    student_id: str
    age: int
    gender: str
    academic_program: str
    year_level: str
    GPA: float
    days_old: int # Add days_old to the Pydantic model
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
            "DAYS_OLD": days_old, # Use the passed days_old
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

    simulated_students_collection = db["simulated_students"]
    predictions_list = []

    # Create new students
    print(f"Simulating {request.num_students} new students with {request.days_old} days old...")
    simulated_raw_data = generate_realistic_student_data(request.num_students, request.days_old)

    # Define the exact list of features the preprocessor expects, in the correct order.
    all_expected_preprocessor_cols = [f.upper() for f in numerical_features] + \
                                     [f.upper() for f in categorical_features]

    for student_data in simulated_raw_data:
        input_df = pd.DataFrame([student_data]).reindex(columns=all_expected_preprocessor_cols, fill_value=np.nan)

        for col in boolean_cols:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(int)

        student_predictions = {}
        for target, model_pipeline in models.items():
            prediction_result = model_pipeline.predict(input_df)
            student_predictions[target] = prediction_result[0].item()

        student_data.update(student_predictions)
        
        # Convert keys to lowercase for database storage and frontend consumption
        processed_student_data = {k.lower(): v for k, v in student_data.items()}
        predictions_list.append(processed_student_data)

    try:
        # Insert new students
        if predictions_list:
            result = simulated_students_collection.insert_many(predictions_list)
            print(f"Inserted {len(result.inserted_ids)} simulated students into MongoDB.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save simulated students to database: {e}")

    for student in predictions_list:
        if "_id" in student:
            student["_id"] = str(student["_id"]) # Convert ObjectId to string
            # Remove the _id field from the dictionary as it's already stringified and not needed
            # The frontend should use student_id or _id_str if it needs the string representation
            student.pop("_id", None)
    return {"message": "Students simulated and saved successfully", "simulated_students": predictions_list}

@router.post("/update-days-old")
async def update_days_old_endpoint(request: UpdateDaysOldRequest):
    if not models:
        raise HTTPException(status_code=500, detail="AI Models not loaded. Cannot update learning styles.")
    
    simulated_students_collection = db["simulated_students"]
    updated_students_list = []

    all_expected_preprocessor_cols = [f.upper() for f in numerical_features] + \
                                     [f.upper() for f in categorical_features]

    for student_id in request.student_ids:
        existing_student = simulated_students_collection.find_one({"student_id": student_id})
        if not existing_student:
            print(f"Student with ID {student_id} not found, skipping update.")
            continue

        existing_student_upper = {k.upper(): v for k, v in existing_student.items()}
        
        # Update DAYS_OLD and other dynamic features
        existing_student_upper["DAYS_OLD"] += request.days_to_add
        existing_student_upper["TIME_SPENT_ON_VIDEOS"] += random.randint(1, 5)
        existing_student_upper["QUIZ_ATTEMPTS"] += random.randint(0, 2)
        existing_student_upper["FORUM_PARTICIPATION_COUNT"] += random.randint(0, 1)
        existing_student_upper["LOGIN_FREQUENCY_PER_WEEK"] += random.randint(0, 1)

        input_df = pd.DataFrame([existing_student_upper]).reindex(columns=all_expected_preprocessor_cols, fill_value=np.nan)

        for col in boolean_cols:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(int)

        student_predictions = {}
        for target, model_pipeline in models.items():
            prediction_result = model_pipeline.predict(input_df)
            student_predictions[target] = prediction_result[0].item()

        existing_student_upper.update(student_predictions)
        
        # Convert _id (ObjectId) to string if it exists before converting to lowercase
        if "_id" in existing_student_upper:
            existing_student_upper["_id"] = str(existing_student_upper["_id"])
        
        processed_student_data = {k.lower(): v for k, v in existing_student_upper.items()}
        # Remove the _id field from the dictionary as it's already stringified and not needed
        # The frontend should use student_id or _id_str if it needs the string representation
        processed_student_data.pop("_id", None)
        updated_students_list.append(processed_student_data)

        try:
            simulated_students_collection.update_one(
                {"student_id": student_id},
                {"$set": processed_student_data}
            )
            print(f"Updated student {student_id} to {processed_student_data['days_old']} days old.")
        except Exception as e:
            print(f"Failed to update student {student_id} in database: {e}")
            continue
            
    return {"message": f"Updated {len(updated_students_list)} students successfully.", "updated_students": updated_students_list}

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
