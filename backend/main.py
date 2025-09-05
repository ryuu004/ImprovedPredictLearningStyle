from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImblearnPipeline # Renamed to avoid conflict
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, confusion_matrix
from imblearn.over_sampling import SMOTE, RandomOverSampler
from fastapi.encoders import jsonable_encoder
from json import JSONEncoder
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from backend.dependencies import models, feature_names, feature_importances_dict, model_performance_metrics, categorical_features, target_labels, boolean_cols, numerical_features, db, client, students_collection, preprocessor_obj, lstm_model, convert_numpy_types # Import preprocessor_obj, lstm_model, and convert_numpy_types
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

import tensorflow as tf
from tensorflow import keras

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
            return np.zeros((X.shape[0], self.embedding_size))

    def get_feature_names_out(self, input_features=None):
        # Return names for the generated LSTM features
        return [f"lstm_feature_{i}" for i in range(self.embedding_size)]

        # Ensure dynamic_data is numeric before passing to LSTM
        dynamic_data_df = X[self.dynamic_features].copy()
        for col in self.dynamic_features:
            dynamic_data_df[col] = pd.to_numeric(dynamic_data_df[col], errors='coerce').fillna(0) # Fill NaN with 0 for LSTM input

        dynamic_data = dynamic_data_df.values
        
        # Reshape for LSTM: (samples, timesteps, features)
        # Assuming each dynamic feature is a timestep or that we need a single timestep
        # For simplicity, treat each set of dynamic features as a single timestep
        dynamic_data_reshaped = dynamic_data.reshape(dynamic_data.shape[0], 1, dynamic_data.shape[1])
        
        # Predict embeddings using the LSTM model
        embeddings = self.lstm_model.predict(dynamic_data_reshaped)
        
        return embeddings

# Define the path for saving/loading models
MODEL_PATH = "backend/models/"

LSTM_MODEL_PATH = os.path.join(MODEL_PATH, "lstm_model.joblib") # Will save Keras model here

class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic): # Catch all other numpy scalars
            return obj.item()
        return JSONEncoder.default(self, obj)

app = FastAPI()
app.json_encoder = NumpyEncoder # Set the custom encoder

def load_lstm_model():
    """Loads the pre-trained LSTM model."""
    global lstm_model
    if os.path.exists(LSTM_MODEL_PATH):
        print(f"Loading LSTM model from {LSTM_MODEL_PATH}")
        try:
            # Load the Keras model directly
            lstm_model = keras.models.load_model(LSTM_MODEL_PATH)
            print("LSTM model loaded successfully.")
        except Exception as e:
            print(f"Error loading LSTM model from {LSTM_MODEL_PATH}: {e}. LSTM embeddings will not be generated.")
            lstm_model = None
    else:
        print(f"Warning: LSTM model not found at {LSTM_MODEL_PATH}. LSTM embeddings will not be generated.")
        lstm_model = None

async def train_lstm_model():
    """Trains a simple LSTM model and saves it."""
    print("Starting LSTM model training...")
    data = list(db["training_data"].find({}))
    if not data:
        print("No data found in MongoDB for LSTM training. Skipping LSTM training.")
        return

    df = pd.DataFrame(data).copy()
    df.columns = df.columns.str.upper()
    df = df.loc[:,~df.columns.duplicated()].copy()

    # Define dynamic features for LSTM training
    dynamic_features = [
        "TIME_SPENT_ON_VIDEOS",
        "QUIZ_ATTEMPTS",
        "FORUM_PARTICIPATION_COUNT",
        "LOGIN_FREQUENCY_PER_WEEK",
        "DAYS_OLD"
    ]

    # Ensure dynamic features are numeric and fill NaNs
    dynamic_data_df = df[dynamic_features].copy()
    for col in dynamic_features:
        dynamic_data_df[col] = pd.to_numeric(dynamic_data_df[col], errors='coerce').fillna(0)
    
    X_lstm = dynamic_data_df.values
    # Reshape to (samples, timesteps, features)
    X_lstm = X_lstm.reshape(X_lstm.shape[0], 1, X_lstm.shape[1])

    # Define a dummy target for LSTM training, as it's primarily for feature extraction
    # In a real scenario, y_lstm would be derived from your sequential data's labels
    # For now, we'll use a simple binary target for demonstration
    y_lstm = np.random.randint(0, 2, size=X_lstm.shape[0])

    # Define LSTM model architecture
    model = keras.Sequential([
        keras.layers.LSTM(64, activation='relu', input_shape=(X_lstm.shape[1], X_lstm.shape[2])),
        keras.layers.Dense(1, activation='sigmoid') # Binary classification for dummy target
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Fitting LSTM model...")
    model.fit(X_lstm, y_lstm, epochs=5, batch_size=32, verbose=0) # Train silently
    
    # Save the LSTM model
    os.makedirs(MODEL_PATH, exist_ok=True)
    model.save(LSTM_MODEL_PATH)
    print(f"LSTM model trained and saved to {LSTM_MODEL_PATH}")

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
        "PREFERENCE_FOR_EXAMPLES",
        "SELF_REFLECTION_ACTIVITY",
        "VIDEO_PAUSE_AND_REPLAY_COUNT",
        "QUIZ_REVIEW_FREQUENCY",
        "SKIPPED_CONTENT_RATIO",
        "LOGIN_FREQUENCY_PER_WEEK",
        "AVERAGE_STUDY_SESSION_LENGTH",
        "DAYS_OLD", # Add DAYS_OLD to required columns
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
            
            # Clear metrics and delete model file for this target if it exists
            if target in model_performance_metrics["random_forest"]:
                del model_performance_metrics["random_forest"][target]
            if target in feature_importances_dict["random_forest"]:
                del feature_importances_dict["random_forest"][target]
            
            # Also remove from `models` dictionary if it exists
            if target in models["random_forest"]:
                del models["random_forest"][target]
            
            model_filename = os.path.join(MODEL_PATH, f"model_RF_{target}.joblib")
            if os.path.exists(model_filename):
                os.remove(model_filename)
                print(f"Deleted existing model file: {model_filename}")
            continue

        print(f"DEBUG: X shape before SKF split: {X.shape}")
        print(f"DEBUG: y shape before SKF split: {y.shape}")
        print(f"DEBUG: y.dtype before SKF split: {y.dtype}")
        print(f"DEBUG: y.unique() before SKF split: {y.unique()}")
        print(f"DEBUG: y.value_counts() before SKF split:\n{y.value_counts()}")
        print(f"DEBUG: y.isnull().sum() before SKF split: {y.isnull().sum()}")
        print(f"DEBUG: Class distribution for {target}:")
        print(y.value_counts())

        # Define model filename
        model_filename = os.path.join(MODEL_PATH, f"model_RF_{target}.joblib")

        # Load model if it exists
        best_model_pipeline = None
        recalculate_metrics = False
        recalculate_importances = False

        if os.path.exists(model_filename):
            print(f"Loading pre-trained model for {target} from {model_filename}")
            loaded_data = joblib.load(model_filename)
            best_model_pipeline = loaded_data['model']
            models["random_forest"][target] = best_model_pipeline # Store the loaded model

            # Check and load performance metrics
            if 'performance_metrics' in loaded_data:
                # Force recalculation if metrics are all zeros or empty lists
                loaded_metrics = loaded_data['performance_metrics']
                if all(value == 0 for key, value in loaded_metrics.items() if isinstance(value, (int, float))) or \
                   any(isinstance(value, list) and not value for value in loaded_metrics.values()):
                    print(f"Warning: Loaded performance metrics for {target} are all zeros or empty. Recalculating...")
                    recalculate_metrics = True
                else:
                    model_performance_metrics["random_forest"][target] = loaded_metrics
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
                    feature_importances_dict["random_forest"][target] = loaded_importances
                    print(f"Feature importances for {target} loaded.")
            else:
                print(f"Warning: No feature importances found for {target} in loaded model. Recalculating...")
                recalculate_importances = True
            
            # If both metrics and importances were loaded AND are valid, we can continue
            if not recalculate_metrics and not recalculate_importances:
                # If everything is valid and loaded, we still need to ensure the best_model_pipeline
                # is set for the current target, as `models` is global.
                models["random_forest"][target] = best_model_pipeline
                # Skip training if not needed, but proceed to metric calculation below.
                # Do not `continue` here, as we always want to calculate/re-calculate metrics.
                pass
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
                    ('num', numerical_transformer, numerical_features + dynamic_features), # Combine numerical static and dynamic
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features) # Categorical static
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
            models["random_forest"][target] = best_model_pipeline # Store the newly trained model
        
        # Now, best_model_pipeline is guaranteed to be set (either loaded or newly trained)
        # Proceed with evaluation and saving for both cases if needed.
        
        # Initialize StratifiedKFold for cross-validation on the best model
        skf_final = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []
        fold_f1_scores = []
        # Initialize lists to store true and predicted labels for confusion matrix calculation
        all_y_test_folds = []
        all_y_preds = []
        
        for fold, (train_index, test_index) in enumerate(skf_final.split(X, y)):
            X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
            y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
            
            # Predict on the test fold using the best_model_pipeline
            y_pred = best_model_pipeline.predict(X_test_fold)
            fold_accuracies.append(accuracy_score(y_test_fold, y_pred))
            fold_precisions.append(precision_score(y_test_fold, y_pred, average='weighted', zero_division=0))
            fold_recalls.append(recall_score(y_test_fold, y_pred, average='weighted', zero_division=0))
            fold_f1_scores.append(f1_score(y_test_fold, y_pred, average='weighted', zero_division=0))

            # Store true and predicted labels for confusion matrix
            all_y_test_folds.extend(y_test_fold.tolist())
            all_y_preds.extend(y_pred.tolist())
            
        # Calculate average metrics across all folds for the best model
        avg_accuracy = float(np.mean(fold_accuracies))
        avg_precision = float(np.mean(fold_precisions))
        avg_recall = float(np.mean(fold_recalls))
        avg_f1 = float(np.mean(fold_f1_scores))
        
        print(f"Model for {target} trained and tuned successfully with cross-validation.")
        
        # Initialize train_accuracy
        train_accuracy = 0.0

        # Calculate training accuracy if X.empty:
        if not X.empty:
            y_pred_train = best_model_pipeline.predict(X)
            train_accuracy = float(accuracy_score(y, y_pred_train))

        # Calculate overall confusion matrix for the target
        overall_confusion_matrix = confusion_matrix(all_y_test_folds, all_y_preds).tolist()
        print(f"DEBUG (main.py - RF): Overall Confusion Matrix for {target}: {overall_confusion_matrix}")
        
        # Store performance metrics (averaged from cross-validation of the best model)
        model_performance_metrics["random_forest"][target] = {
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
            "best_params": convert_numpy_types(opt.best_params_) if 'opt' in locals() else {}, # Changed 'N/A (loaded model)' to {}
            "fold_accuracies": convert_numpy_types(fold_accuracies),
            "fold_precisions": convert_numpy_types(fold_precisions),
            "fold_recalls": convert_numpy_types(fold_recalls),
            "fold_f1_scores": convert_numpy_types(fold_f1_scores),
            "confusion_matrices_per_fold": [overall_confusion_matrix], # Store the overall confusion matrix
            "class_distribution": convert_numpy_types(y.value_counts().to_dict()) # Add class distribution
        }
        print(f"DEBUG (main.py - RF): Stored metrics for {target}: {model_performance_metrics['random_forest'][target]}")
        
        # Get feature importances from the best classifier within the pipeline
        if hasattr(best_model_pipeline.named_steps['classifier'], 'feature_importances_'):
            importances = best_model_pipeline.named_steps['classifier'].feature_importances_
            preprocessed_feature_names = best_model_pipeline.named_steps['feature_preprocessor'].get_feature_names_out()
            feature_importances_dict["random_forest"][target] = convert_numpy_types(dict(zip(preprocessed_feature_names, importances)))
            print(f"Feature importances for {target} calculated and stored.")
 
        # Prepare data to save: model, performance metrics, and feature importances
        model_data_to_save = {
            'model': best_model_pipeline,
            'performance_metrics': model_performance_metrics["random_forest"][target],
            'feature_importances': feature_importances_dict.get("random_forest", {}).get(target, {})
        }

        # Save the trained model and its associated data
        joblib.dump(model_data_to_save, model_filename)
        print(f"Model and associated data for {target} saved to {model_filename}")

async def train_xgboost_model():
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
        "PREFERENCE_FOR_EXAMPLES",
        "SELF_REFLECTION_ACTIVITY",
        "VIDEO_PAUSE_AND_REPLAY_COUNT",
        "QUIZ_REVIEW_FREQUENCY",
        "SKIPPED_CONTENT_RATIO",
        "LOGIN_FREQUENCY_PER_WEEK",
        "AVERAGE_STUDY_SESSION_LENGTH",
        "DAYS_OLD", # Add DAYS_OLD to required columns
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
            
            # Clear metrics and delete model file for this target if it exists
            if target in model_performance_metrics["xgboost"]:
                del model_performance_metrics["xgboost"][target]
            if target in feature_importances_dict["xgboost"]:
                del feature_importances_dict["xgboost"][target]
            
            # Also remove from `models` dictionary if it exists
            if target in models["xgboost"]:
                del models["xgboost"][target]
            
            model_filename = os.path.join(MODEL_PATH, f"model_XGB_{target}.joblib")
            if os.path.exists(model_filename):
                os.remove(model_filename)
                print(f"Deleted existing model file: {model_filename}")
            continue

        print(f"DEBUG: X shape before SKF split: {X.shape}")
        print(f"DEBUG: y shape before SKF split: {y.shape}")
        print(f"DEBUG: y.dtype before SKF split: {y.dtype}")
        print(f"DEBUG: y.unique() before SKF split: {y.unique()}")
        print(f"DEBUG: y.value_counts() before SKF split:\n{y.value_counts()}")
        print(f"DEBUG: y.isnull().sum() before SKF split: {y.isnull().sum()}")
        print(f"DEBUG: Class distribution for {target}:")
        print(y.value_counts())

        # Define model filename
        model_filename = os.path.join(MODEL_PATH, f"model_XGB_{target}.joblib") # Changed for XGBoost

        # Load model if it exists
        best_model_pipeline = None
        recalculate_metrics = False
        recalculate_importances = False

        if os.path.exists(model_filename):
            print(f"Loading pre-trained model for {target} from {model_filename}")
            loaded_data = joblib.load(model_filename)
            best_model_pipeline = loaded_data['model']
            models["xgboost"][target] = best_model_pipeline # Store the loaded model

            # Check and load performance metrics
            if 'performance_metrics' in loaded_data:
                # Force recalculation if metrics are all zeros or empty lists
                loaded_metrics = loaded_data['performance_metrics']
                if all(value == 0 for key, value in loaded_metrics.items() if isinstance(value, (int, float))) or \
                   any(isinstance(value, list) and not value for value in loaded_metrics.values()):
                    print(f"Warning: Loaded performance metrics for {target} are all zeros or empty. Recalculating...")
                    recalculate_metrics = True
                else:
                    model_performance_metrics["xgboost"][target] = loaded_metrics
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
                    feature_importances_dict["xgboost"][target] = loaded_importances
                    print(f"Feature importances for {target} loaded.")
            else:
                print(f"Warning: No feature importances found for {target} in loaded model. Recalculating...")
                recalculate_importances = True
            
            # If both metrics and importances were loaded AND are valid, we can continue
            if not recalculate_metrics and not recalculate_importances:
                # If everything is valid and loaded, we still need to ensure the best_model_pipeline
                # is set for the current target, as `models` is global.
                models["xgboost"][target] = best_model_pipeline
                # Skip training if not needed, but proceed to metric calculation below.
                # Do not `continue` here, as we always want to calculate/re-calculate metrics.
                pass
            else:
                print(f"Proceeding to recalculate metrics/importances for {target}.")
        
        # If best_model_pipeline is still None, it means the model was not loaded (or not found)
        # So, we need to train it from scratch.
        if best_model_pipeline is None:
            # Hyperparameter tuning search space for RandomForestClassifier
            param_space = {
                'classifier__n_estimators': Integer(100, 1000), # Changed for XGBoost
                'classifier__max_depth': Integer(3, 10), # Changed for XGBoost
                'classifier__learning_rate': Real(0.01, 0.3, prior='log-uniform'), # Changed for XGBoost
                'classifier__subsample': Real(0.5, 1.0, prior='uniform'), # Changed for XGBoost
                'classifier__colsample_bytree': Real(0.5, 1.0, prior='uniform'), # Changed for XGBoost
                'classifier__gamma': Real(0, 0.5, prior='uniform'), # Changed for XGBoost
                'classifier__reg_alpha': Real(0, 0.5, prior='uniform'), # Changed for XGBoost
                'smote__k_neighbors': Integer(1, 10) # For SMOTE
            }

            # Define a preprocessor for the static and dynamic features
            # This meta-preprocessor will apply different transformations to different columns
            feature_preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features + dynamic_features), # Combine numerical static and dynamic
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features) # Categorical static
                ],
                remainder='passthrough' # Keep other columns (e.g., non-preprocessed static features)
            )

            # Create a base pipeline for BayesSearchCV
            base_pipeline = ImblearnPipeline(steps=[('feature_preprocessor', feature_preprocessor),
                                                     ('smote', SMOTE(random_state=42)),
                                                     ('classifier', XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False))]) # Changed for XGBoost

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
            models["xgboost"][target] = best_model_pipeline # Store the newly trained model
        
        # Now, best_model_pipeline is guaranteed to be set (either loaded or newly trained)
        # Proceed with evaluation and saving for both cases if needed.
        
        # Initialize StratifiedKFold for cross-validation on the best model
        skf_final = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []
        fold_f1_scores = []
        all_y_test_folds = [] # To store y_test for all folds
        all_y_preds = [] # To store y_pred for all folds
        
        for fold, (train_index, test_index) in enumerate(skf_final.split(X, y)):
            X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
            y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
            
            # Predict on the test fold using the best_model_pipeline
            y_pred = best_model_pipeline.predict(X_test_fold)
            fold_accuracies.append(accuracy_score(y_test_fold, y_pred))
            fold_precisions.append(precision_score(y_test_fold, y_pred, average='weighted', zero_division=0))
            fold_recalls.append(recall_score(y_test_fold, y_pred, average='weighted', zero_division=0))
            fold_f1_scores.append(f1_score(y_test_fold, y_pred, average='weighted', zero_division=0))
            
            all_y_test_folds.append(y_test_fold)
            all_y_preds.append(y_pred)
            
        # Calculate average metrics across all folds for the best model
        avg_accuracy = float(np.mean(fold_accuracies))
        avg_precision = float(np.mean(fold_precisions))
        avg_recall = float(np.mean(fold_recalls))
        avg_f1 = float(np.mean(fold_f1_scores))
        
        print(f"Model for {target} trained and tuned successfully with cross-validation.")
        
        # Initialize train_accuracy
        train_accuracy = 0.0

        # Calculate training accuracy if X is not empty
        if not X.empty:
            y_pred_train = best_model_pipeline.predict(X)
            train_accuracy = float(accuracy_score(y, y_pred_train))

        # Store performance metrics (averaged from cross-validation of the best model)
        model_performance_metrics["xgboost"][target] = {
            "training_accuracy": train_accuracy,
            "test_accuracy": avg_accuracy, # This is the averaged cross-validation accuracy
            "precision": avg_precision,
            "recall": avg_recall,
            "f1_score": avg_f1,
            "model_type": "XGBoost (Tuned)", # Changed for XGBoost
            "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_size": len(data),
            "features_used": len(numerical_features) + len(categorical_features),
            "cross_validation_folds": skf_final.get_n_splits(X, y),
            "best_params": convert_numpy_types(opt.best_params_) if 'opt' in locals() else "N/A (loaded model)", # Include best_params if newly trained
            "fold_accuracies": convert_numpy_types(fold_accuracies),
            "fold_precisions": convert_numpy_types(fold_precisions),
            "fold_recalls": convert_numpy_types(fold_recalls),
            "fold_f1_scores": convert_numpy_types(fold_f1_scores),
            "confusion_matrices_per_fold": [], # Initialize as empty, will be populated below
            "class_distribution": convert_numpy_types(y.value_counts().to_dict()) # Add class distribution
        }
        
        # Initialize lists to store true and predicted labels for confusion matrix calculation
        all_y_test_folds = []
        all_y_preds = []

        for fold, (train_index, test_index) in enumerate(skf_final.split(X, y)):
            X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
            y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
            
            # Predict on the test fold using the best_model_pipeline
            y_pred = best_model_pipeline.predict(X_test_fold)
            fold_accuracies.append(accuracy_score(y_test_fold, y_pred))
            fold_precisions.append(precision_score(y_test_fold, y_pred, average='weighted', zero_division=0))
            fold_recalls.append(recall_score(y_test_fold, y_pred, average='weighted', zero_division=0))
            fold_f1_scores.append(f1_score(y_test_fold, y_pred, average='weighted', zero_division=0))

            # Store true and predicted labels for confusion matrix
            all_y_test_folds.extend(y_test_fold.tolist())
            all_y_preds.extend(y_pred.tolist())
            
        # Calculate average metrics across all folds for the best model
        avg_accuracy = float(np.mean(fold_accuracies))
        avg_precision = float(np.mean(fold_precisions))
        avg_recall = float(np.mean(fold_recalls))
        avg_f1 = float(np.mean(fold_f1_scores))
        
        print(f"Model for {target} trained and tuned successfully with cross-validation.")
        
        # Initialize train_accuracy
        train_accuracy = 0.0

        # Calculate training accuracy if X is not empty
        if not X.empty:
            y_pred_train = best_model_pipeline.predict(X)
            train_accuracy = float(accuracy_score(y, y_pred_train))

        # Calculate overall confusion matrix for the target
        overall_confusion_matrix = confusion_matrix(all_y_test_folds, all_y_preds).tolist()
        print(f"DEBUG (main.py - XGB): Overall Confusion Matrix for {target}: {overall_confusion_matrix}")

        # Store performance metrics (averaged from cross-validation of the best model)
        model_performance_metrics["xgboost"][target] = {
            "training_accuracy": train_accuracy,
            "test_accuracy": avg_accuracy, # This is the averaged cross-validation accuracy
            "precision": avg_precision,
            "recall": avg_recall,
            "f1_score": avg_f1,
            "model_type": "XGBoost (Tuned)", # Changed for XGBoost
            "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_size": len(data),
            "features_used": len(numerical_features) + len(categorical_features),
            "cross_validation_folds": skf_final.get_n_splits(X, y),
            "best_params": convert_numpy_types(opt.best_params_) if 'opt' in locals() else {}, # Changed 'N/A (loaded model)' to {}
            "fold_accuracies": convert_numpy_types(fold_accuracies),
            "fold_precisions": convert_numpy_types(fold_precisions),
            "fold_recalls": convert_numpy_types(fold_recalls),
            "fold_f1_scores": convert_numpy_types(fold_f1_scores),
            "confusion_matrices_per_fold": [overall_confusion_matrix], # Store the overall confusion matrix
            "class_distribution": convert_numpy_types(y.value_counts().to_dict()) # Add class distribution
        }
        print(f"DEBUG (main.py - XGB): Stored metrics for {target}: {model_performance_metrics['xgboost'][target]}")
        
        # Get feature importances from the best classifier within the pipeline
        if hasattr(best_model_pipeline.named_steps['classifier'], 'feature_importances_'):
            importances = best_model_pipeline.named_steps['classifier'].feature_importances_
            preprocessed_feature_names = best_model_pipeline.named_steps['feature_preprocessor'].get_feature_names_out()
            feature_importances_dict["xgboost"][target] = convert_numpy_types(dict(zip(preprocessed_feature_names, importances)))
            print(f"Feature importances for {target} calculated and stored.")
 
        # Prepare data to save: model, performance metrics, and feature importances
        model_data_to_save = {
            'model': best_model_pipeline,
            'performance_metrics': model_performance_metrics["xgboost"][target],
            'feature_importances': feature_importances_dict.get("xgboost", {}).get(target, {})
        }

        # Save the trained model and its associated data
        joblib.dump(model_data_to_save, model_filename)
        print(f"Model and associated data for {target} saved to {model_filename}")

@app.on_event("startup")
async def startup_event():
    load_lstm_model() # Load LSTM model on startup
    await train_model() # Train Random Forest models
    await train_xgboost_model() # Train XGBoost models

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
        raise HTTPException(status_code=500, detail=str(e))

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
    if not models:
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
        for model_type, models_by_type in models.items():
            for target, model_pipeline in models_by_type.items():
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
                # This logic is simplified for now. A robust solution would involve storing category mappings.
                predictions[f"{model_type}_{target}_prediction"] = float(prediction[0].item()) # Ensure native Python float
                try:
                    proba = model_pipeline.predict_proba(input_df)[0].tolist()
                    predictions[f"{model_type}_{target}_proba"] = proba
                except AttributeError:
                    predictions[f"{model_type}_{target}_proba"] = "N/A" # Model does not support predict_proba

        # Add feature importances to the prediction response
        # This will return importances for both models, nested under their types
        predictions["feature_importances"] = feature_importances_dict

        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))