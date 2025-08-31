from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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

# Global variables for models, feature names, and feature importances
models = {}
feature_names = None
feature_importances_dict = {} # New global variable to store feature importances
categorical_features = [
    'note_taking_style',
    'problem_solving_preference',
    'response_speed_in_quizzes',
    'year_level',
    'academic_program'
]
target_labels = [
    'active_vs_reflective',
    'sensing_vs_intuitive',
    'visual_vs_verbal',
    'sequential_vs_global'
]

async def train_model():
    global models, feature_names
    data = list(db["training_data"].find({}))
    if not data:
        print("No data found in MongoDB to train the models.")
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
        'active_vs_reflective',
        'sensing_vs_intuitive',
        'visual_vs_verbal',
        'sequential_vs_global',
        'note_taking_style',
        'problem_solving_preference',
        'response_speed_in_quizzes',
        'year_level',
        'academic_program'
    ]
    for col in required_columns:
        if col not in df.columns:
            print(f"Warning: Missing required column for training: {col}. This column will be skipped.")
            # Optionally, you might want to raise an error or handle it differently

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

    numerical_features = [
        'GPA',
        'time_spent_on_videos',
        'time_spent_on_text_materials',
        'time_spent_on_interactive_activities',
        'forum_participation_count',
        'group_activity_participation',
        'individual_activity_preference',
        'quiz_attempts',
        'time_to_complete_assignments',
        'accuracy_in_detail_oriented_questions',
        'accuracy_in_conceptual_questions',
        'video_pause_and_replay_count',
        'quiz_review_frequency',
        'skipped_content_ratio',
        'login_frequency_per_week',
        'average_study_session_length'
    ]

    # Preprocessing for categorical and numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop' # Drop other columns not specified
    )

    # Convert target labels to numerical if they are strings
    for target in target_labels:
        if df[target].dtype == 'object':
            df[target] = df[target].astype('category').cat.codes

    # Define features (X)
    X = df[numerical_features + categorical_features]
    
    # Handle potential non-numeric values gracefully by converting to numeric, coercing errors
    for col in numerical_features:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # Drop rows with NaN values that resulted from coercion
    X.dropna(inplace=True)

    # Train a model for each target label
    for target in target_labels:
        y = df.loc[X.index, target] # Align y with X after dropping NaNs

        if X.shape[0] == 0 or X.shape[0] != y.shape[0]:
            print(f"Not enough valid data after cleaning to train the model for {target}.")
            continue

        # Create a pipeline that preprocesses the data then trains the model
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        pipeline.fit(X_train, y_train)
        models[target] = pipeline
        print(f"Model for {target} trained successfully.")
        
        # Get feature names after one-hot encoding and numerical features
        # This part needs to be done carefully as one-hot encoder changes column names
        # Fit preprocessor to get feature names
        preprocessor.fit(X)
        encoded_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        
        # Combine numerical and encoded categorical feature names
        current_feature_names = list(numerical_features) + list(encoded_feature_names)
        
        # Store feature importances
        if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
            importances = pipeline.named_steps['classifier'].feature_importances_
            feature_importances_dict[target] = dict(zip(current_feature_names, importances))
            print(f"Feature importances for {target} calculated and stored.")

    # Set the global feature_names after all models are trained
    # Assuming all models use the same set of processed features,
    # we can use the last `current_feature_names` or re-derive it once.
    if models: # Ensure models were trained
        preprocessor.fit(X)
        encoded_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        feature_names = list(numerical_features) + list(encoded_feature_names)

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

@app.get("/feature-importances")
async def get_feature_importances():
    if not feature_importances_dict:
        raise HTTPException(status_code=404, detail="Feature importances not calculated yet. Models might not be trained or data is insufficient.")
    return feature_importances_dict

@app.post("/predict-learning-style")
async def predict_learning_style(student_data: dict):
    if not models or feature_names is None:
        raise HTTPException(status_code=500, detail="Models not trained yet.")

    try:
        input_df = pd.DataFrame([student_data])

        # Convert boolean fields to numerical (0 or 1)
        for col in boolean_cols:
            if col in input_df.columns:
                input_df[col] = input_df[col].apply(lambda x: 1 if x else 0)

        # Prepare numerical features by coercing to numeric
        for col in numerical_features:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
            else:
                input_df[col] = 0 # Default to 0 if numerical feature is missing

        # For categorical features, ensure they are present or default to a value that OneHotEncoder can handle
        for col in categorical_features:
            if col not in input_df.columns:
                input_df[col] = 'unknown' # Or a sensible default for your data

        # Make predictions for all models
        predictions = {}
        for target, model_pipeline in models.items():
            # Use the same preprocessor from the pipeline to transform input data
            # The pipeline itself handles the transformation
            
            # Ensure the input_df has all necessary columns for the preprocessor
            # This is crucial for ColumnTransformer to work correctly
            # We need to pass the full set of expected features to the pipeline's predict method
            
            # Reconstruct the input dataframe with all expected columns in the correct order
            # This is more robust as it ensures all expected features (numerical + categorical) are present
            # even if not provided in the student_data, they will be handled by the pipeline.
            
            # Create a dataframe with all expected input features, filling missing with None or default
            
            # It's better to let the pipeline handle missing columns if `handle_unknown='ignore'` is set for OneHotEncoder
            # and `remainder='drop'` for ColumnTransformer.
            
            # The input_df should contain the columns that the preprocessor expects from the raw data.
            # These are numerical_features + categorical_features.
            
            # Select only the columns that the preprocessor expects
            input_for_prediction = input_df[numerical_features + categorical_features]
            
            # Make prediction using the pipeline
            prediction = model_pipeline.predict(input_for_prediction)
            
            # Convert prediction back to original category if it was encoded
            if target in target_labels and df[target_labels[target_labels.index(target)]].dtype == 'object':
                # This part is tricky. We need to store the original categories during training.
                # For now, let's assume the prediction is the numerical code.
                # A more robust solution would involve storing the mapping.
                pass
            
            predictions[f"{target}_prediction"] = prediction[0].item() # .item() to get scalar from numpy array

        # Add feature importances to the prediction response
        predictions["feature_importances"] = feature_importances_dict

        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))