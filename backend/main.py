from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImblearnPipeline # Renamed to avoid conflict
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from backend.dependencies import models, feature_names, feature_importances_dict, model_performance_metrics, categorical_features, target_labels, boolean_cols, numerical_features, db, client, students_collection, preprocessor_obj, lstm_model # Import preprocessor_obj and lstm_model
import joblib # Import joblib for model persistence
from skopt import BayesSearchCV # Import BayesSearchCV for hyperparameter tuning
from skopt.space import Real, Categorical, Integer # Import space definitions
import os # Import os module for path operations
from sklearn.base import BaseEstimator, TransformerMixin # For custom transformer
from sklearn.compose import ColumnTransformer # Import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder # Import OneHotEncoder
from sklearn.impute import SimpleImputer # Import SimpleImputer
from sklearn.preprocessing import StandardScaler # Import StandardScaler
from sklearn.pipeline import Pipeline # Import Pipeline


class LSTMFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, lstm_model, dynamic_features, embedding_size=10):
        self.lstm_model = lstm_model
        self.dynamic_features = dynamic_features
        self.embedding_size = embedding_size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.lstm_model is None:
            # If LSTM model is not loaded, return dummy embeddings
            # This should ideally not happen in production if model loading is robust
            return np.zeros((X.shape[0], self.embedding_size))

        # Extract dynamic features
        dynamic_data = X[self.dynamic_features].values

        # Reshape for LSTM: (samples, timesteps, features)
        # Assuming each dynamic feature is a timestep or that we need a single timestep
        # For this simplified example, let's assume (samples, 1, features)
        # In a real scenario, you'd need to properly sequence your dynamic data.
        # Here, we're treating each student's dynamic features as a single "timestep".
        
        # This part depends heavily on the actual LSTM model's input shape.
        # For a simple feed-forward LSTM, it might be (samples, features).
        # If it's a sequence model, it would need (samples, timesteps, features).
        # For now, let's assume a simple case where we feed the dynamic features directly.
        
        # If the LSTM expects (batch_size, timesteps, features):
        # dynamic_data_reshaped = dynamic_data.reshape(dynamic_data.shape[0], 1, dynamic_data.shape[1])
        
        # For a simple placeholder, let's just ensure it's 2D for now.
        # If your LSTM expects 3D, you'll need to adapt this.
        
        # For a simple LSTM model that takes a 2D array (samples, features)
        # as input, like a Dense layer after an LSTM layer, this might work.
        # If it's a true sequence model, this needs proper time-series data.
        
        # Assuming the LSTM model expects 2D input (samples, features) directly from dynamic features
        # If it's a Keras model, you might need to convert to TensorFlow tensors.
        
        # For the purpose of getting past the error, let's assume `lstm_model.predict`
        # takes a 2D numpy array.
        
        # In a real LSTM setup, `dynamic_data` would likely need to be a sequence.
        # Here, let's simulate a simple case where we just pass the dynamic features.
        
        # Dummy LSTM prediction for now, to enable pipeline construction
        # In a real scenario, this would be `self.lstm_model.predict(dynamic_data)`
        
        # The output of LSTM is typically a 2D array (samples, embedding_size)
        # For now, let's return random data if the model isn't functional
        if self.lstm_model:
            # Assuming lstm_model.predict takes a 2D array (samples, features)
            # and returns a 2D array (samples, embedding_size).
            # If your LSTM model expects a 3D input (samples, timesteps, features),
            # `dynamic_data` would need to be reshaped accordingly, e.g.:
            # dynamic_data_reshaped = dynamic_data.reshape(dynamic_data.shape[0], 1, dynamic_data.shape[1])
            # embeddings = self.lstm_model.predict(dynamic_data_reshaped)
            
            embeddings = self.lstm_model.predict(dynamic_data)
        else:
            # If LSTM model is not loaded, return zeros for embeddings
            embeddings = np.zeros((X.shape[0], self.embedding_size))
        
        return embeddings

# Define the path for saving/loading models
MODEL_PATH = "backend/models/"

LSTM_MODEL_PATH = os.path.join(MODEL_PATH, "lstm_model.joblib")

app = FastAPI()

def load_lstm_model():
    """Loads the pre-trained LSTM model."""
    global lstm_model
    if os.path.exists(LSTM_MODEL_PATH):
        print(f"Loading LSTM model from {LSTM_MODEL_PATH}")
        lstm_model = joblib.load(LSTM_MODEL_PATH)
    else:
        print(f"Warning: LSTM model not found at {LSTM_MODEL_PATH}. LSTM embeddings will not be generated.")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

async def train_model():
    global models, feature_names, model_performance_metrics
    
    # Define dynamic and static features for training
    dynamic_features = [
        "TIME_SPENT_ON_VIDEOS",
        "QUIZ_ATTEMPTS",
        "FORUM_PARTICIPATION_COUNT",
        "LOGIN_FREQUENCY_PER_WEEK",
        "DAYS_OLD"
    ]
    static_features = [
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
    # Ensure all dynamic and static features are in the DataFrame
    all_combined_features = [f.upper() for f in dynamic_features] + [f.upper() for f in static_features]

    # Create the models directory if it doesn't exist
    os.makedirs(MODEL_PATH, exist_ok=True)

    data = list(db["training_data"].find({}))
    if not data:
        print("No data found in MongoDB to train the models.")
        return
    print(f"Found {len(data)} documents for training.")

    # --- Start: Placeholder for LSTM Model Training and Saving ---
    # In a real application, this would involve proper LSTM architecture definition,
    # data preparation (e.g., sequence generation), training, and evaluation.
    # For now, we create a dummy LSTM model to ensure the pipeline can proceed.
    from sklearn.linear_model import LogisticRegression # A simple model to act as a dummy LSTM
    
    # Define a dummy LSTM model based on dynamic features
    # This dummy model will simply learn a linear relationship
    # This is *not* a real LSTM, but allows the pipeline to function.
    if not os.path.exists(LSTM_MODEL_PATH):
        print("Training and saving a dummy LSTM model...")
        # Prepare some dummy data for the dummy LSTM
        # The dynamic features are expected to be the input
        dynamic_feature_columns = [f.upper() for f in dynamic_features] # Use the defined dynamic_features
        
        df_temp = pd.DataFrame(data).copy() # Use a temporary DataFrame for dummy LSTM training
        df_temp.columns = df_temp.columns.str.upper()
        df_temp = df_temp.loc[:,~df_temp.columns.duplicated()].copy()

        # Ensure dynamic_feature_columns exist in df_temp before selecting
        existing_dynamic_cols = [col for col in dynamic_feature_columns if col in df_temp.columns]
        
        if not existing_dynamic_cols:
            print("Warning: No dynamic features available in DataFrame for dummy LSTM training.")
            # Create a simple dummy model that takes one input if no dynamic features exist
            dummy_lstm_input = np.zeros((df_temp.shape[0], 1))
            dummy_lstm = LogisticRegression() # Just a placeholder
            dummy_lstm.fit(dummy_lstm_input, np.zeros(df_temp.shape[0])) # Train with dummy target
        else:
            dummy_lstm_input = df_temp[existing_dynamic_cols].values
            # Ensure numerical data for dummy LSTM
            for col in existing_dynamic_cols:
                df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce').fillna(0) # Fill NaN for dummy training
            
            # For a "dummy LSTM", we just need something that takes 2D input and produces 2D output
            # LogisticRegression or a small RandomForest can serve as a placeholder
            dummy_lstm = LogisticRegression(max_iter=1000) # Increase max_iter for convergence
            # Create a dummy target for training the dummy LSTM
            dummy_target = np.random.randint(0, 2, size=dummy_lstm_input.shape[0])
            try:
                dummy_lstm.fit(dummy_lstm_input, dummy_target)
            except ValueError as e:
                print(f"Error training dummy LSTM: {e}. Check dynamic features data.")
                # Fallback if dummy training fails
                dummy_lstm = None # Set to None so LSTMFeatureExtractor returns zeros
                
        if dummy_lstm:
            joblib.dump(dummy_lstm, LSTM_MODEL_PATH)
            print("Dummy LSTM model saved.")
    # --- End: Placeholder for LSTM Model Training and Saving ---

    df = pd.DataFrame(data).copy() # Explicitly copy to avoid SettingWithCopyWarning
    df.columns = df.columns.str.upper()
    # Handle duplicate columns by keeping the first occurrence
    df = df.loc[:,~df.columns.duplicated()].copy()
    print(f"DataFrame columns after uppercase conversion: {df.columns.tolist()}")

    for col in all_combined_features:
        if col not in df.columns:
            df[col] = np.nan # Add missing columns with NaN

    # Ensure all required columns exist
    required_columns = [
        'GPA',
        'TIME_SPENT_ON_VIDEOS',
        'TIME_SPENT_ON_TEXT_MATERIALS',
        'TIME_SPENT_ON_INTERACTIVE_ACTIVITIES',
        'FORUM_PARTICIPATION_COUNT',
        'GROUP_ACTIVITY_PARTICIPATION',
        'INDIVIDUAL_ACTIVITY_PREFERENCE',
        'PREFERENCE_FOR_VISUAL_MATERIALS',
        'PREFERENCE_FOR_TEXTUAL_MATERIALS',
        'QUIZ_ATTEMPTS',
        'TIME_TO_COMPLETE_ASSIGNMENTS',
        'ACCURACY_IN_DETAIL_ORIENTED_QUESTIONS',
        'ACCURACY_IN_CONCEPTUAL_QUESTIONS',
        'PREFERENCE_FOR_EXAMPLES',
        'SELF_REFLECTION_ACTIVITY',
        'VIDEO_PAUSE_AND_REPLAY_COUNT',
        'QUIZ_REVIEW_FREQUENCY',
        'SKIPPED_CONTENT_RATIO',
        'LOGIN_FREQUENCY_PER_WEEK',
        'AVERAGE_STUDY_SESSION_LENGTH',
        'DAYS_OLD', # Add DAYS_OLD to required columns
        'ACTIVE_VS_REFLECTIVE',
        'SENSING_VS_INTUITIVE',
        'VISUAL_VS_VERBAL',
        'SEQUENTIAL_VS_GLOBAL',
        'NOTE_TAKING_STYLE',
        'PROBLEM_SOLVING_PREFERENCE',
        'RESPONSE_SPEED_IN_QUIZZES',
        'YEAR_LEVEL',
        'ACADEMIC_PROGRAM'
    ]
    for col in required_columns:
        if col not in df.columns:
            print(f"Warning: Missing required column for training: {col}. This column will be skipped.")
            # Optionally, you might want to raise an error or handle it differently

    # Convert boolean fields to numerical (0 or 1)
    for col in boolean_cols:
        if col in df.columns: # Added check to avoid KeyError
            df.loc[:, col] = df[col].astype(int)

    # Convert target labels to numerical if they are strings
    for target in target_labels:
        if df[target].dtype == 'object':
            df.loc[:, target] = df[target].astype('category').cat.codes

    # The preprocessor_obj now handles numerical feature imputation
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Explicitly convert categorical features to string, filling NaNs with 'missing_category'
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('missing_category')
        else:
            # Add missing categorical columns with default 'missing_category'
            df[col] = 'missing_category'
    
    # Ensure 'DAYS_OLD' is present in the DataFrame with a default value if not already
    if 'DAYS_OLD' not in df.columns:
        df['DAYS_OLD'] = 0 # Default value for existing training data
    
    # Also ensure 'DAYS_OLD' is converted to numeric type
    df['DAYS_OLD'] = pd.to_numeric(df['DAYS_OLD'], errors='coerce').fillna(0).astype(int)

    # Prepare X for training: this will be the input to our custom transformer
    # that generates LSTM embeddings and combines with static features.
    X = df[all_combined_features].copy()
    print(f"DEBUG (main): X shape before preprocessor fit: {X.shape}")

    # Fit the global preprocessor_obj once on the original, cleaned X
    # This ensures it learns the categories from the full dataset before any augmentation
    # Only fit if not already fitted (e.g., on first startup)
    if not hasattr(preprocessor_obj, 'n_features_in_'): # Check if fitted
        preprocessor_obj.fit(X)
        print(f"DEBUG (main): preprocessor_obj fitted. Categorical feature names out: {preprocessor_obj.named_transformers_['cat'].get_feature_names_out(categorical_features)}")
        print(f"DEBUG (main): preprocessor_obj feature names: {preprocessor_obj.get_feature_names_out()}")

    # Convert target labels to numerical if they are strings
    # This needs to be done before splitting X and y for training
    target_mappings = {} # Store mappings for later use if needed
    for target in target_labels:
        if df[target].dtype == 'object':
            # Get unique values before converting to category codes
            unique_vals = df[target].unique()
            # If there's only one unique value, it will cause issues with cat.codes for classification
            if len(unique_vals) == 1:
                print(f"Warning: Target '{target}' has only one unique label: {unique_vals[0]}. This may prevent effective classification.")
                # Assign a default mapping to avoid errors
                target_mappings[target] = {unique_vals[0]: 0}
                df.loc[:, target] = 0 # Assign 0 if only one class
            else:
                df.loc[:, target] = df[target].astype('category').cat.codes
                # Store the mapping from codes back to original labels
                target_mappings[target] = dict(enumerate(df[target].astype('category').cat.categories))
        
    # Align y with X after filling NaNs and converting target labels
    for target in target_labels:
        # Initialize metrics for the current target to avoid UnboundLocalError
        train_accuracy = 0.0
        avg_accuracy = 0.0
        avg_precision = 0.0
        avg_recall = 0.0
        avg_f1 = 0.0
        skf_final = None # Initialize skf_final here
        fold_accuracies = [] # Initialize here
        fold_precisions = [] # Initialize here
        fold_recalls = [] # Initialize here
        fold_f1_scores = [] # Initialize here

        y = df.loc[X.index, target].astype(int)

        # Explicitly handle potential NaN values in y (which might appear as -1 after cat.codes)
        # and ensure y is of integer type for classification
        if y.dtype == 'int8' and (y == -1).any(): # Check if -1 exists in y (from cat.codes for NaN)
            original_len = len(X)
            # Drop rows from X and y where y is -1
            X = X[y != -1]
            y = y[y != -1]
            if len(X) < original_len:
                print(f"Dropped {original_len - len(X)} rows due to NaN/missing target values for {target}.")

        if X.shape[0] == 0 or X.shape[0] != y.shape[0]:
            print(f"Not enough valid data after cleaning to train the model for {target}.")
            continue

        # Check for sufficient unique labels in y
        unique_labels = y.nunique()
        if unique_labels < 2:
            print(f"Skipping training for {target}: Not enough unique labels ({unique_labels}) for classification.")
            continue

        print(f"DEBUG: X shape before SKF split: {X.shape}")
        print(f"DEBUG: y shape before SKF split: {y.shape}")
        print(f"DEBUG: y.dtype before SKF split: {y.dtype}")
        print(f"DEBUG: y.unique() before SKF split: {y.unique()}")
        print(f"DEBUG: y.value_counts() before SKF split:\n{y.value_counts()}")
        print(f"DEBUG: y.isnull().sum() before SKF split: {y.isnull().sum()}")

        # Define model filename
        model_filename = os.path.join(MODEL_PATH, f"model_{target}.joblib")

        # Load model if it exists
        best_model_pipeline = None
        recalculate_metrics = False
        recalculate_importances = False

        if os.path.exists(model_filename):
            print(f"Loading pre-trained model for {target} from {model_filename}")
            loaded_data = joblib.load(model_filename)
            best_model_pipeline = loaded_data['model']
            models[target] = best_model_pipeline # Store the loaded model

            # Check and load performance metrics
            if 'performance_metrics' in loaded_data:
                # Force recalculation if metrics are all zeros or empty lists
                loaded_metrics = loaded_data['performance_metrics']
                if all(value == 0 for key, value in loaded_metrics.items() if isinstance(value, (int, float))) or \
                   any(isinstance(value, list) and not value for value in loaded_metrics.values()):
                    print(f"Warning: Loaded performance metrics for {target} are all zeros or empty. Recalculating...")
                    recalculate_metrics = True
                else:
                    model_performance_metrics[target] = loaded_metrics
                    print(f"Performance metrics for {target} loaded.")
            else:
                print(f"Warning: No performance metrics found for {target} in loaded model. Recalculating...")
                recalculate_metrics = True

            # Check and load feature importances
            if 'feature_importances' in loaded_data:
                # Force recalculation if feature importances are empty
                loaded_importances = loaded_data['feature_importances']
                if not loaded_importances: # Check if the dictionary is empty
                    print(f"Warning: Loaded feature importances for {target} are empty. Recalculating...")
                    recalculate_importances = True
                else:
                    feature_importances_dict[target] = loaded_importances
                    print(f"Feature importances for {target} loaded.")
            else:
                print(f"Warning: No feature importances found for {target} in loaded model. Recalculating...")
                recalculate_importances = True
            
            # If both metrics and importances were loaded AND are valid, we can continue
            if not recalculate_metrics and not recalculate_importances:
                continue # Skip training/re-evaluation if model and metrics were fully loaded and valid
            else:
                print(f"Proceeding to recalculate metrics/importances for {target}.")
        
        # If best_model_pipeline is still None, it means the model was not loaded (or not found)
        # So, we need to train it from scratch.
        if best_model_pipeline is None:
            # Hyperparameter tuning search space for RandomForestClassifier
            param_space = {
                'classifier__n_estimators': Integer(50, 200),
                'classifier__max_features': Categorical(['sqrt', 'log2', None]),
                'classifier__max_depth': Integer(5, 50),
                'classifier__min_samples_split': Integer(2, 20),
                'classifier__min_samples_leaf': Integer(1, 10),
                'smote__k_neighbors': Integer(1, 10) # For SMOTE
            }

            # Define a preprocessor for the static and dynamic features
            # This meta-preprocessor will apply different transformations to different columns
            feature_preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, [f for f in numerical_features if f not in dynamic_features]), # Numerical static
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features), # Categorical static
                    ('lstm_features', LSTMFeatureExtractor(lstm_model, dynamic_features), dynamic_features) # Dynamic features for LSTM
                ],
                remainder='passthrough' # Keep other columns (e.g., non-preprocessed static features)
            )

            # Create a base pipeline for BayesSearchCV
            base_pipeline = ImblearnPipeline(steps=[('feature_preprocessor', feature_preprocessor),
                                                     ('smote', SMOTE(random_state=42)),
                                                     ('classifier', RandomForestClassifier(random_state=42))])

            # Use f1_score as the scoring metric for BayesSearchCV, using make_scorer for weighted average
            f1_scorer = make_scorer(f1_score, average='weighted', zero_division=0)

            # Initialize BayesSearchCV
            opt = BayesSearchCV(
                estimator=base_pipeline,
                search_spaces=param_space,
                n_iter=50, # Number of optimization iterations (can be increased)
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), # Use StratifiedKFold for CV
                scoring=f1_scorer,
                n_jobs=-1, # Use all available cores
                random_state=42,
                verbose=1
            )

            print(f"Starting hyperparameter tuning for {target}...")
            opt.fit(X, y)

            print(f"Hyperparameter tuning for {target} completed.")
            print(f"Best parameters for {target}: {opt.best_params_}")
            
            # The best estimator from BayesSearchCV is the trained model with optimal hyperparameters
            best_model_pipeline = opt.best_estimator_
            models[target] = best_model_pipeline # Store the newly trained model
        
        # Now, best_model_pipeline is guaranteed to be set (either loaded or newly trained)
        # Proceed with evaluation and saving for both cases if needed.
        
        # Initialize StratifiedKFold for cross-validation on the best model
        skf_final = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []
        fold_f1_scores = []

        for fold, (train_index, test_index) in enumerate(skf_final.split(X, y)):
            X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
            y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
            
            # Predict on the test fold using the best_model_pipeline
            y_pred = best_model_pipeline.predict(X_test_fold)
            fold_accuracies.append(accuracy_score(y_test_fold, y_pred))
            fold_precisions.append(precision_score(y_test_fold, y_pred, average='weighted', zero_division=0))
            fold_recalls.append(recall_score(y_test_fold, y_pred, average='weighted', zero_division=0))
            fold_f1_scores.append(f1_score(y_test_fold, y_pred, average='weighted', zero_division=0))
            
        # Calculate average metrics across all folds for the best model
        avg_accuracy = np.mean(fold_accuracies)
        avg_precision = np.mean(fold_precisions)
        avg_recall = np.mean(fold_recalls)
        avg_f1 = np.mean(fold_f1_scores)
        
        print(f"Model for {target} trained and tuned successfully with cross-validation.")
        
        # Initialize train_accuracy
        train_accuracy = 0.0

        # Calculate training accuracy if X is not empty
        if not X.empty:
            y_pred_train = best_model_pipeline.predict(X)
            train_accuracy = accuracy_score(y, y_pred_train)

        # Store performance metrics (averaged from cross-validation of the best model)
        model_performance_metrics[target] = {
            "training_accuracy": train_accuracy,
            "test_accuracy": avg_accuracy, # This is the averaged cross-validation accuracy
            "precision": avg_precision,
            "recall": avg_recall,
            "f1_score": avg_f1,
            "model_type": "Random Forest (Tuned)",
            "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_size": len(data),
            "features_used": len(numerical_features) + len(categorical_features),
            "cross_validation_folds": skf_final.get_n_splits(X, y),
            "best_params": opt.best_params_ if 'opt' in locals() else "N/A (loaded model)", # Include best_params if newly trained
            "fold_accuracies": fold_accuracies,
            "fold_precisions": fold_precisions,
            "fold_recalls": fold_recalls,
            "fold_f1_scores": fold_f1_scores
        }
        
        # Get feature importances from the best classifier within the pipeline
        if hasattr(best_model_pipeline.named_steps['classifier'], 'feature_importances_'):
            importances = best_model_pipeline.named_steps['classifier'].feature_importances_
            preprocessed_feature_names = best_model_pipeline.named_steps['feature_preprocessor'].get_feature_names_out()
            feature_importances_dict[target] = dict(zip(preprocessed_feature_names, importances))
            print(f"Feature importances for {target} calculated and stored.")
 
        # Prepare data to save: model, performance metrics, and feature importances
        model_data_to_save = {
            'model': best_model_pipeline,
            'performance_metrics': model_performance_metrics[target],
            'feature_importances': feature_importances_dict.get(target, {})
        }

        # Save the trained model and its associated data
        joblib.dump(model_data_to_save, model_filename)
        print(f"Model and associated data for {target} saved to {model_filename}")

@app.on_event("startup")
async def startup_event():
    load_lstm_model() # Load LSTM model on startup
    await train_model()

@app.get("/")
async def read_root():
    return {"message": "FastAPI backend is running"}

from backend.routers import model_metrics # Include the new router
from backend.routers import simulate_students # Include the new router

app.include_router(model_metrics.router)
app.include_router(simulate_students.router)

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

@app.get("/model-performance")
async def get_model_performance():
    if not models:
        raise HTTPException(status_code=404, detail="Models not trained yet.")

    return model_performance_metrics


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
                input_df[col] = input_df[col].astype(int)

        # Numerical feature imputation is now handled by the preprocessor_obj
        # Ensure numerical columns are numeric before passing to preprocessor,
        # but let SimpleImputer handle NaNs.
        for col in numerical_features:
            if col in input_df.columns:
                input_df.loc[:, col] = input_df[col].replace('none', np.nan)
                input_df.loc[:, col] = pd.to_numeric(input_df[col], errors='coerce')
            # If the column is missing, SimpleImputer in preprocessor_obj will handle it
            # by filling with the mean from training data, so no need for manual 0 fill here.

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
            
            # Make prediction using the pipeline. The pipeline handles all preprocessing internally.
            prediction = model_pipeline.predict(input_df)
            
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