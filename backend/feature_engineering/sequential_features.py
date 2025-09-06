import pandas as pd
from datetime import timedelta
from typing import List, Dict, Any
import numpy as np

class SequentialFeatureEngineer:
    """
    A class to engineer advanced sequential features from raw activity logs.
    """
    def __init__(self, sequence_length_days: int = 14):
        self.sequence_length_days = sequence_length_days

    def _calculate_engagement_ratios(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculates engagement ratios based on activity types."""
        total_activities = len(df)
        if total_activities == 0:
            return {
                "video_watch_ratio": 0.0,
                "quiz_attempt_ratio": 0.0,
                "forum_post_ratio": 0.0,
                "module_completion_ratio": 0.0,
                "avg_daily_activities": 0.0
            }

        video_watches = df[df['activity_type'] == 'video_watch'].shape[0]
        quiz_attempts = df[df['activity_type'] == 'quiz_attempt'].shape[0]
        forum_posts = df[df['activity_type'] == 'forum_post'].shape[0]
        module_completions = df[df['activity_type'] == 'module_completion'].shape[0]

        start_date = df['timestamp'].min().normalize()
        end_date = df['timestamp'].max().normalize()
        num_days = (end_date - start_date).days + 1 if total_activities > 0 else 0
        
        avg_daily_activities = total_activities / num_days if num_days > 0 else 0

        return {
            "video_watch_ratio": video_watches / total_activities,
            "quiz_attempt_ratio": quiz_attempts / total_activities,
            "forum_post_ratio": forum_posts / total_activities,
            "module_completion_ratio": module_completions / total_activities,
            "avg_daily_activities": avg_daily_activities
        }

    def _calculate_study_session_trends(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculates trends in study session lengths."""
        sessions = df[df['activity_type'].isin(['study_session_start', 'study_session_end'])].sort_values('timestamp')
        
        session_durations = []
        start_time = None
        for _, row in sessions.iterrows():
            if row['activity_type'] == 'study_session_start':
                start_time = row['timestamp']
            elif row['activity_type'] == 'study_session_end' and start_time:
                duration = (row['timestamp'] - start_time).total_seconds() / 60
                session_durations.append(duration)
                start_time = None # Reset for next session

        if not session_durations:
            return {
                "avg_session_length": 0.0,
                "std_session_length": 0.0,
                "session_length_trend": 0.0 # Placeholder for actual trend calculation
            }
        
        avg_session_length = sum(session_durations) / len(session_durations)
        std_session_length = np.std(session_durations) if len(session_durations) > 1 else 0.0
        
        # Simple trend: difference between last few sessions and first few
        if len(session_durations) >= 2:
            session_length_trend = session_durations[-1] - session_durations[0]
        else:
            session_length_trend = 0.0

        return {
            "avg_session_length": avg_session_length,
            "std_session_length": std_session_length,
            "session_length_trend": session_length_trend
        }

    def _calculate_activity_transitions(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculates common activity transitions."""
        transitions = {}
        if len(df) < 2:
            return {}

        df_sorted = df.sort_values('timestamp')
        for i in range(len(df_sorted) - 1):
            current_activity = df_sorted.iloc[i]['activity_type']
            next_activity = df_sorted.iloc[i+1]['activity_type']
            transition = f"{current_activity}_to_{next_activity}"
            transitions[transition] = transitions.get(transition, 0) + 1
        
        total_transitions = sum(transitions.values())
        if total_transitions == 0:
            return {}
            
        return {k: v / total_transitions for k, v in transitions.items()}

    def engineer_features(self, activity_logs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineers sequential features for each student from their activity logs.
        
        Args:
            activity_logs_df (pd.DataFrame): DataFrame of all student activity logs,
                                             must contain 'student_id', 'timestamp', 'activity_type'.

        Returns:
            pd.DataFrame: A DataFrame where each row is a student and columns are the engineered features.
        """
        if activity_logs_df.empty:
            return pd.DataFrame()

        # Ensure timestamp is datetime
        activity_logs_df['timestamp'] = pd.to_datetime(activity_logs_df['timestamp'])
        
        engineered_features = []
        
        for student_id, group in activity_logs_df.groupby('student_id'):
            # Filter activities for the defined sequence length (e.g., last 14 days)
            latest_timestamp = group['timestamp'].max()
            relevant_activities = group[group['timestamp'] >= (latest_timestamp - timedelta(days=self.sequence_length_days))]
            
            student_features = {"student_id": student_id}
            
            # 1. Engagement Ratios
            student_features.update(self._calculate_engagement_ratios(relevant_activities))
            
            # 2. Study Session Trends
            student_features.update(self._calculate_study_session_trends(relevant_activities))
            
            # 3. Activity Transitions (top N for example, or specific ones)
            # For simplicity, we'll include all calculated transitions and let downstream models handle feature selection
            student_features.update(self._calculate_activity_transitions(relevant_activities))
            
            # Add other potential features:
            # - Percentage of modules completed in order (requires 'module_completion' and 'order_in_module')
            # - Time since last activity
            # - Average time between logins
            
            engineered_features.append(student_features)
            
        return pd.DataFrame(engineered_features).set_index('student_id')

if __name__ == "__main__":
    # Example Usage:
    # Assuming synthetic_data_generator.py is in the same parent directory or importable
    from backend.data_generation.synthetic_data_generator import generate_all_students_activities
    
    # Create a dummy DataFrame for student profiles
    dummy_student_profiles = pd.DataFrame({
        'STUDENT_ID': [f'student_{i}' for i in range(5)],
        'GPA': [3.5, 2.8, 3.9, 3.1, 2.5],
        'ACADEMIC_PROGRAM': ['IT', 'Business', 'Engineering', 'Arts', 'Science']
    })
    
    print("Generating synthetic activity data...")
    synthetic_activities_df = generate_all_students_activities(dummy_student_profiles, 30) # Simulate 30 days of activity
    print(f"Total synthetic activities generated: {len(synthetic_activities_df)}")
    
    feature_engineer = SequentialFeatureEngineer(sequence_length_days=14)
    print("\nEngineering sequential features...")
    engineered_features_df = feature_engineer.engineer_features(synthetic_activities_df)
    
    print("\nEngineered Features Sample:")
    print(engineered_features_df.head())
    print(f"\nEngineered Features Shape: {engineered_features_df.shape}")
    print(f"Engineered Features Columns: {engineered_features_df.columns.tolist()}")