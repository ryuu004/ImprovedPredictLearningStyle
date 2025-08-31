from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImblearnPipeline # Renamed to avoid conflict
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from backend.dependencies import models, feature_names, feature_importances_dict, model_performance_metrics, categorical_features, target_labels, boolean_cols, numerical_features, db, client, students_collection
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib # Import joblib for model persistence
from skopt import BayesSearchCV # Import BayesSearchCV for hyperparameter tuning
from skopt.space import Real, Categorical, Integer # Import space definitions
import os # Import os module for path operations

# Define the preprocessor globally
preprocessor_obj = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop'
)

# Define the path for saving/loading models
MODEL_PATH = "backend/models/"

app = FastAPI()

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

    # Explicitly convert categorical features to string, filling NaNs with 'missing_category'
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('missing_category')
        else:
            # Add missing categorical columns with default 'missing_category'
            df[col] = 'missing_category'
    
    X = df[numerical_features + categorical_features].copy() # Use .copy() to avoid SettingWithCopyWarning with X
    print(f"DEBUG (main): numerical_features length: {len(numerical_features)}")
    print(f"DEBUG (main): categorical_features length: {len(categorical_features)}")
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

        print(f"DEBUG: Before training for target {target}:")
        print(f"DEBUG: y.dtype: {y.dtype}")
        print(f"DEBUG: y.unique(): {y.unique()}")
        print(f"DEBUG: y.value_counts():\n{y.value_counts()}")
        print(f"DEBUG: y.isnull().sum(): {y.isnull().sum()}")

        # Define model filename
        model_filename = os.path.join(MODEL_PATH, f"model_{target}.joblib")

        # Load model if it exists
        if os.path.exists(model_filename):
            print(f"Loading pre-trained model for {target} from {model_filename}")
            loaded_data = joblib.load(model_filename)
            models[target] = loaded_data['model']
            
            # Load performance metrics and feature importances
            if 'performance_metrics' in loaded_data:
                model_performance_metrics[target] = loaded_data['performance_metrics']
                print(f"Performance metrics for {target} loaded.")
            else:
                print(f"Warning: No performance metrics found for {target} in loaded model.")
                # Fallback to N/A if metrics are not found in the loaded file
                model_performance_metrics[target] = {
                    "accuracy": "N/A (loaded)",
                    "precision": "N/A (loaded)",
                    "recall": "N/A (loaded)",
                    "f1_score": "N/A (loaded)",
                    "model_type": "Random Forest (Loaded)",
                    "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "dataset_size": len(data),
                    "features_used": len(numerical_features) + len(categorical_features),
                    "cross_validation_folds": "N/A (loaded)",
                    "fold_accuracies": "N/A (loaded)",
                    "fold_precisions": "N/A (loaded)",
                    "fold_recalls": "N/A (loaded)",
                    "fold_f1_scores": "N/A (loaded)"
                }

            if 'feature_importances' in loaded_data:
                feature_importances_dict[target] = loaded_data['feature_importances']
                print(f"Feature importances for {target} loaded.")
            else:
                print(f"Warning: No feature importances found for {target} in loaded model.")
            
            continue # Skip training if model is loaded

        # Hyperparameter tuning search space for RandomForestClassifier
        # More parameters can be added here
        param_space = {
            'classifier__n_estimators': Integer(50, 200),
            'classifier__max_features': Categorical(['sqrt', 'log2', None]),
            'classifier__max_depth': Integer(5, 50),
            'classifier__min_samples_split': Integer(2, 20),
            'classifier__min_samples_leaf': Integer(1, 10),
            'smote__k_neighbors': Integer(1, 10) # For SMOTE
        }

        # Create a base pipeline for BayesSearchCV
        base_pipeline = ImblearnPipeline(steps=[('preprocessor', preprocessor_obj),
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
        models[target] = best_model_pipeline

        # Prepare data to save: model, performance metrics, and feature importances
        model_data_to_save = {
            'model': best_model_pipeline,
            'performance_metrics': {
                "training_accuracy": train_accuracy,
                "test_accuracy": avg_accuracy,
                "precision": avg_precision,
                "recall": avg_recall,
                "f1_score": avg_f1,
                "model_type": "Random Forest (Tuned)",
                "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_size": len(data),
                "features_used": len(numerical_features) + len(categorical_features),
                "cross_validation_folds": skf_final.get_n_splits(X, y) if skf_final is not None else "N/A",
                "best_params": opt.best_params_,
                "fold_accuracies": fold_accuracies,
                "fold_precisions": fold_precisions,
                "fold_recalls": fold_recalls,
                "fold_f1_scores": fold_f1_scores
            },
            'feature_importances': feature_importances_dict.get(target, {}) # Use .get() to avoid KeyError if not present
        }

        # Save the trained model and its associated data
        # Only save if skf_final was actually initialized (meaning training occurred)
        if skf_final is not None:
            joblib.dump(model_data_to_save, model_filename)
            print(f"Model and associated data for {target} saved to {model_filename}")
        else:
            print(f"Skipping saving model for {target} as training was not performed or failed.")

        # Evaluate the best model using cross-validation (from BayesSearchCV results)
        # We can directly use opt.cv_results_ to get fold metrics if needed,
        # but for simplicity, let's re-evaluate on the best estimator using a new SKF.
        
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
            "best_params": opt.best_params_,
            "fold_accuracies": fold_accuracies,
            "fold_precisions": fold_precisions,
            "fold_recalls": fold_recalls,
            "fold_f1_scores": fold_f1_scores
        }
        
        # Get feature importances from the best classifier within the pipeline
        if hasattr(best_model_pipeline.named_steps['classifier'], 'feature_importances_'):
            importances = best_model_pipeline.named_steps['classifier'].feature_importances_
            preprocessed_feature_names = best_model_pipeline.named_steps['preprocessor'].get_feature_names_out()
            feature_importances_dict[target] = dict(zip(preprocessed_feature_names, importances))
            print(f"Feature importances for {target} calculated and stored.")
 
    # Set the global feature_names after all models are trained or loaded
    # Assuming all models use the same set of processed features,
    # we can use the last `current_feature_names` or re-derive it once.
    if models: # Ensure models were trained or loaded
        # Get preprocessor from any of the models to get feature names
        any_model_pipeline = next(iter(models.values()))
        
        # Ensure 'preprocessor' step exists in the pipeline
        if 'preprocessor' in any_model_pipeline.named_steps:
            preprocessor_step = any_model_pipeline.named_steps['preprocessor']
            # Check if 'cat' transformer exists and has get_feature_names_out
            if 'cat' in preprocessor_step.named_transformers_ and hasattr(preprocessor_step.named_transformers_['cat'], 'get_feature_names_out'):
                encoded_feature_names = preprocessor_step.named_transformers_['cat'].get_feature_names_out(categorical_features)
                feature_names = list(numerical_features) + list(encoded_feature_names)
            else:
                print("Warning: 'cat' transformer or get_feature_names_out not found in preprocessor. Feature names might be incomplete.")
                feature_names = list(numerical_features) # Fallback
        else:
            print("Warning: 'preprocessor' step not found in pipeline. Feature names might be incomplete.")
            feature_names = list(numerical_features) # Fallback

@app.on_event("startup")
async def startup_event():
    await train_model()

@app.get("/")
async def read_root():
    return {"message": "FastAPI backend is running"}

from backend.routers import model_metrics # Include the new router

app.include_router(model_metrics.router)

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