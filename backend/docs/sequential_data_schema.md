# Synthetic Sequential Activity Data Schema

This document defines the schema for the synthetic sequential activity data, which will be generated to simulate student interactions within a learning environment. This data will serve as the basis for advanced feature engineering and training the LSTM model.

Each entry in the sequential activity log will represent a single event and will adhere to the following structure:

```json
{
  "student_id": "string", // Unique identifier for the student
  "timestamp": "datetime", // ISO 8601 formatted timestamp of the activity
  "activity_type": "string", // Type of activity (e.g., "video_watch", "quiz_attempt", "forum_post", "module_completion", "login", "study_session_start", "study_session_end")
  "module_id": "string", // (Optional) Identifier of the learning module involved
  "duration_minutes": "float", // (Optional) Duration of the activity in minutes (e.g., for video watch, study sessions)
  "score": "float", // (Optional) Score obtained in an activity (e.g., quiz score, 0-100)
  "correctness": "boolean", // (Optional) Whether a specific attempt/question was correct (true/false)
  "video_id": "string", // (Optional) Identifier for video content
  "quiz_id": "string", // (Optional) Identifier for quiz content
  "forum_thread_id": "string", // (Optional) Identifier for forum thread
  "order_in_module": "integer" // (Optional) For module completion, its intended sequential order within the course
}
```

## Activity Type Descriptions:

- `video_watch`: Student watched a video. `duration_minutes` can represent watched time.
- `quiz_attempt`: Student attempted a quiz. `score` and `correctness` (per question) are relevant.
- `forum_post`: Student posted or replied in a forum. `forum_thread_id` is relevant.
- `module_completion`: Student completed a module. `module_id` and `order_in_module` are relevant for sequence tracking.
- `login`: Student logged into the platform.
- `study_session_start`: Marks the beginning of a study session.
- `study_session_end`: Marks the end of a study session. `duration_minutes` will be calculated from start/end timestamps.

This schema provides the necessary granularity to derive complex sequential features, such as:

- **Percentage of modules completed in order:** By tracking `module_completion` and `order_in_module`.
- **Trend in study session lengths:** By analyzing `study_session_start`/`end` and `duration_minutes` over time.
- **Common activity transitions:** By analyzing the sequence of `activity_type` events (e.g., `video_watch` -> `quiz_attempt` -> `forum_post`).
- **Engagement ratios:** By summing `duration_minutes` for various activity types or counting `activity_type` occurrences per day/week.
