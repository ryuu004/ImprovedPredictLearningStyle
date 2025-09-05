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

from backend.dependencies import models, numerical_static_features, categorical_features, boolean_cols, db, preprocessor_obj, lstm_model, dynamic_features
from sklearn.preprocessing import OneHotEncoder
import numpy as np # Ensure numpy is imported

router = APIRouter()

class SimulateRequest(BaseModel):
    num_students: int
    days_old: int = 0
    model_type: str = "random_forest" # Add model_type with a default value

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
    active_vs_reflective: float
    sensing_vs_intuitive: float
    visual_vs_verbal: float
    sequential_vs_global: float

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
            "ACTIVE_VS_REFLECTIVE": round(random.uniform(0.0, 1.0), 2),
            "SENSING_VS_INTUITIVE": round(random.uniform(0.0, 1.0), 2),
            "VISUAL_VS_VERBAL": round(random.uniform(0.0, 1.0), 2),
            "SEQUENTIAL_VS_GLOBAL": round(random.uniform(0.0, 1.0), 2),
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
    all_expected_preprocessor_cols = [f.upper() for f in numerical_static_features] + \
                                     [f.upper() for f in categorical_features] + \
                                     [f.upper() for f in boolean_cols] + \
                                     [f.upper() for f in dynamic_features]

    for student_data in simulated_raw_data:
        input_df = pd.DataFrame([student_data]).reindex(columns=all_expected_preprocessor_cols, fill_value=np.nan)

        # Convert boolean fields to numerical (0 or 1)
        for col in boolean_cols:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(int)
            else:
                input_df[col] = 0 # Default to 0 if missing

        # Explicitly convert categorical features to string, filling NaNs with 'missing_category'
        for col in categorical_features:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(str).fillna('missing_category')
            else:
                input_df[col] = 'missing_category'

        # Ensure 'DAYS_OLD' is converted to numeric type
        if 'DAYS_OLD' in input_df.columns:
            input_df['DAYS_OLD'] = pd.to_numeric(input_df['DAYS_OLD'], errors='coerce').fillna(0).astype(int)
        else:
            input_df['DAYS_OLD'] = 0 # Default if not provided

        student_predictions = {}
        # Use the selected model_type from the request
        if request.model_type in models:
            models_to_use = {request.model_type: models[request.model_type]}
        else:
            # Fallback to a default or raise an error if the model_type is not found
            # For now, let's just use random_forest as default if not found
            print(f"Warning: Model type '{request.model_type}' not found. Using 'random_forest' as fallback.")
            models_to_use = {"random_forest": models["random_forest"]}

        for model_type, models_by_type in models_to_use.items(): # Iterate over selected model type
            for target, model_pipeline in models_by_type.items(): # Iterate over actual models
                prediction_result = model_pipeline.predict(input_df)
                student_predictions[f"{model_type}_{target}"] = prediction_result[0].item() # Prefix with model_type for clarity

        # Convert float predictions to categorical strings
        # Convert float predictions to categorical strings based on the selected model type
        # We only care about the predictions from the chosen model_type
        prefix = f"{request.model_type}_"
        if f"{prefix}ACTIVE_VS_REFLECTIVE" in student_predictions:
            student_predictions["ACTIVE_VS_REFLECTIVE"] = "Active" if student_predictions[f"{prefix}ACTIVE_VS_REFLECTIVE"] > 0.5 else "Reflective"
        else:
            # Fallback for when the model_type wasn't found and we defaulted to random_forest
            if "random_forest_ACTIVE_VS_REFLECTIVE" in student_predictions:
                student_predictions["ACTIVE_VS_REFLECTIVE"] = "Active" if student_predictions["random_forest_ACTIVE_VS_REFLECTIVE"] > 0.5 else "Reflective"

        if f"{prefix}SENSING_VS_INTUITIVE" in student_predictions:
            student_predictions["SENSING_VS_INTUITIVE"] = "Sensing" if student_predictions[f"{prefix}SENSING_VS_INTUITIVE"] > 0.5 else "Intuitive"
        else:
            if "random_forest_SENSING_VS_INTUITIVE" in student_predictions:
                student_predictions["SENSING_VS_INTUITIVE"] = "Sensing" if student_predictions["random_forest_SENSING_VS_INTUITIVE"] > 0.5 else "Intuitive"

        if f"{prefix}VISUAL_VS_VERBAL" in student_predictions:
            student_predictions["VISUAL_VS_VERBAL"] = "Visual" if student_predictions[f"{prefix}VISUAL_VS_VERBAL"] > 0.5 else "Verbal"
        else:
            if "random_forest_VISUAL_VS_VERBAL" in student_predictions:
                student_predictions["VISUAL_VS_VERBAL"] = "Visual" if student_predictions["random_forest_VISUAL_VS_VERBAL"] > 0.5 else "Verbal"

        if f"{prefix}SEQUENTIAL_VS_GLOBAL" in student_predictions:
            student_predictions["SEQUENTIAL_VS_GLOBAL"] = "Sequential" if student_predictions[f"{prefix}SEQUENTIAL_VS_GLOBAL"] > 0.5 else "Global"
        else:
            if "random_forest_SEQUENTIAL_VS_GLOBAL" in student_predictions:
                student_predictions["SEQUENTIAL_VS_GLOBAL"] = "Sequential" if student_predictions["random_forest_SEQUENTIAL_VS_GLOBAL"] > 0.5 else "Global"

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

    all_expected_preprocessor_cols = [f.upper() for f in numerical_static_features] + \
                                     [f.upper() for f in categorical_features] + \
                                     [f.upper() for f in boolean_cols] + \
                                     [f.upper() for f in dynamic_features]

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
        existing_student_upper["ACTIVE_VS_REFLECTIVE"] = round(random.uniform(0.0, 1.0), 2)
        existing_student_upper["SENSING_VS_INTUITIVE"] = round(random.uniform(0.0, 1.0), 2)
        existing_student_upper["VISUAL_VS_VERBAL"] = round(random.uniform(0.0, 1.0), 2)
        existing_student_upper["SEQUENTIAL_VS_GLOBAL"] = round(random.uniform(0.0, 1.0), 2)

        input_df = pd.DataFrame([existing_student_upper]).reindex(columns=all_expected_preprocessor_cols, fill_value=np.nan)

        # Convert boolean fields to numerical (0 or 1)
        for col in boolean_cols:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(int)
            else:
                input_df[col] = 0 # Default to 0 if missing

        # Explicitly convert categorical features to string, filling NaNs with 'missing_category'
        for col in categorical_features:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(str).fillna('missing_category')
            else:
                input_df[col] = 'missing_category'
        
        # Ensure 'DAYS_OLD' is converted to numeric type
        if 'DAYS_OLD' in input_df.columns:
            input_df['DAYS_OLD'] = pd.to_numeric(input_df['DAYS_OLD'], errors='coerce').fillna(0).astype(int)
        else:
            input_df['DAYS_OLD'] = 0 # Default if not provided

        student_predictions = {}
        # For updating students, we don't have model_type in the request. Let's use random_forest as default.
        # In a real scenario, you might want to store the model used for initial prediction or pass it here.
        models_to_use = {"random_forest": models["random_forest"]} # Default to random_forest for updates

        for model_type, models_by_type in models_to_use.items(): # Iterate over selected model type
            for target, model_pipeline in models_by_type.items(): # Iterate over actual models
                prediction_result = model_pipeline.predict(input_df)
                student_predictions[f"{model_type}_{target}"] = prediction_result[0].item() # Prefix with model_type for clarity

        # Convert float predictions to categorical strings
        # Convert float predictions to categorical strings based on the model used (random_forest for updates)
        # We only care about the predictions from the chosen model_type (random_forest here)
        prefix = "random_forest_" # Since we defaulted to random_forest for updates
        if f"{prefix}ACTIVE_VS_REFLECTIVE" in student_predictions:
            existing_student_upper["ACTIVE_VS_REFLECTIVE"] = "Active" if student_predictions[f"{prefix}ACTIVE_VS_REFLECTIVE"] > 0.5 else "Reflective"

        if f"{prefix}SENSING_VS_INTUITIVE" in student_predictions:
            existing_student_upper["SENSING_VS_INTUITIVE"] = "Sensing" if student_predictions[f"{prefix}SENSING_VS_INTUITIVE"] > 0.5 else "Intuitive"

        if f"{prefix}VISUAL_VS_VERBAL" in student_predictions:
            existing_student_upper["VISUAL_VS_VERBAL"] = "Visual" if student_predictions[f"{prefix}VISUAL_VS_VERBAL"] > 0.5 else "Verbal"

        if f"{prefix}SEQUENTIAL_VS_GLOBAL" in student_predictions:
            existing_student_upper["SEQUENTIAL_VS_GLOBAL"] = "Sequential" if student_predictions[f"{prefix}SEQUENTIAL_VS_GLOBAL"] > 0.5 else "Global"

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
