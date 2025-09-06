import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any

def generate_student_activities(student_id: str, start_date: datetime, num_days: int) -> List[Dict[str, Any]]:
    """
    Generates synthetic sequential activity data for a single student.

    Args:
        student_id (str): The unique identifier for the student.
        start_date (datetime): The starting date for activity generation.
        num_days (int): The number of days to simulate activities for.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing an activity event.
    """
    activities = []
    
    # Simulate a basic daily routine
    for day in range(num_days):
        current_date = start_date + timedelta(days=day)
        
        # Simulate login activity
        login_time = current_date.replace(hour=random.randint(8, 10), minute=random.randint(0, 59), second=random.randint(0, 59))
        activities.append({
            "student_id": student_id,
            "timestamp": login_time.isoformat(),
            "activity_type": "login"
        })

        # Simulate study sessions
        num_sessions = random.randint(1, 3)
        for _ in range(num_sessions):
            session_start_hour = random.randint(10, 22)
            session_start_minute = random.randint(0, 59)
            session_start_time = current_date.replace(hour=session_start_hour, minute=session_start_minute, second=random.randint(0, 59))
            
            session_duration_minutes = random.randint(30, 180) # 30 mins to 3 hours
            session_end_time = session_start_time + timedelta(minutes=session_duration_minutes)

            activities.append({
                "student_id": student_id,
                "timestamp": session_start_time.isoformat(),
                "activity_type": "study_session_start"
            })
            
            # During the session, simulate various activities
            sub_activities_count = random.randint(3, 10)
            for _ in range(sub_activities_count):
                sub_activity_type = random.choice([
                    "video_watch", "quiz_attempt", "forum_post", "module_completion"
                ])
                sub_activity_time = session_start_time + timedelta(minutes=random.randint(0, session_duration_minutes))
                
                activity_data = {
                    "student_id": student_id,
                    "timestamp": sub_activity_time.isoformat(),
                    "activity_type": sub_activity_type
                }
                
                if sub_activity_type == "video_watch":
                    activity_data["video_id"] = f"video_{random.randint(1, 10)}"
                    activity_data["duration_minutes"] = round(random.uniform(5, 30), 2)
                elif sub_activity_type == "quiz_attempt":
                    activity_data["quiz_id"] = f"quiz_{random.randint(1, 5)}"
                    activity_data["score"] = random.randint(50, 100)
                    activity_data["correctness"] = random.choice([True, False])
                elif sub_activity_type == "forum_post":
                    activity_data["forum_thread_id"] = f"thread_{random.randint(1, 20)}"
                elif sub_activity_type == "module_completion":
                    activity_data["module_id"] = f"module_{random.randint(1, 15)}"
                    activity_data["order_in_module"] = random.randint(1, 5) # Assuming 5 steps per module
                    
                activities.append(activity_data)

            activities.append({
                "student_id": student_id,
                "timestamp": session_end_time.isoformat(),
                "activity_type": "study_session_end",
                "duration_minutes": session_duration_minutes # Store calculated duration
            })
            
    # Sort activities by timestamp
    activities.sort(key=lambda x: x["timestamp"])
    return activities

def generate_all_students_activities(
    student_profiles: pd.DataFrame, 
    total_days: int,
    start_date: datetime = datetime(2024, 1, 1)
) -> pd.DataFrame:
    """
    Generates synthetic sequential activity data for multiple students.

    Args:
        student_profiles (pd.DataFrame): DataFrame containing student static profiles (at least 'STUDENT_ID').
        total_days (int): The total number of days to simulate activities for each student.
        start_date (datetime): The starting date for activity generation for all students.

    Returns:
        pd.DataFrame: A DataFrame containing all generated activity events.
    """
    all_activities = []
    for _, student in student_profiles.iterrows():
        student_id = student['STUDENT_ID']
        student_activities = generate_student_activities(student_id, start_date, total_days)
        all_activities.extend(student_activities)
    
    df_activities = pd.DataFrame(all_activities)
    # Ensure timestamp is datetime object for sorting and further processing
    df_activities['timestamp'] = pd.to_datetime(df_activities['timestamp'])
    df_activities = df_activities.sort_values(by=['student_id', 'timestamp']).reset_index(drop=True)
    return df_activities

if __name__ == "__main__":
    # Example Usage:
    # Create a dummy DataFrame for student profiles
    dummy_student_profiles = pd.DataFrame({
        'STUDENT_ID': [f'student_{i}' for i in range(3)],
        'GPA': [3.5, 2.8, 3.9],
        'ACADEMIC_PROGRAM': ['IT', 'Business', 'Engineering']
    })
    
    print("Generating synthetic activity data for 3 students over 7 days...")
    synthetic_activities_df = generate_all_students_activities(dummy_student_profiles, 7)
    
    print("\nGenerated Activities Sample:")
    print(synthetic_activities_df.head(10))
    print(f"\nTotal activities generated: {len(synthetic_activities_df)}")
    print(f"Columns: {synthetic_activities_df.columns.tolist()}")