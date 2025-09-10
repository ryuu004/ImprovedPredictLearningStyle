import warnings

# Suppress the specific UserWarning from XGBoost
warnings.filterwarnings("ignore", message=".*use_label_encoder.*", category=UserWarning)

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
from datetime import datetime, timedelta
from backend.dependencies import models, feature_names, feature_importances_dict, model_performance_metrics, categorical_features, target_labels, boolean_cols, numerical_static_features, dynamic_features, db, client, students_collection, preprocessor_obj, lstm_model, convert_numpy_types # Import preprocessor_obj, lstm_model, and convert_numpy_types
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
from backend.data_generation.synthetic_data_generator import generate_student_activities, generate_all_students_activities # Import the data generator
from backend.feature_engineering.sequential_features import SequentialFeatureEngineer # Import the feature engineer

# Define numerical transformer
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

class LSTMFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, lstm_model, embedding_size=64):
        self.lstm_model = lstm_model
        self.embedding_size = embedding_size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.lstm_model is None:
            return np.zeros((X.shape[0], self.embedding_size))

        # X is expected to already contain the pre-computed LSTM input features
        # These features should be a single row per student, representing the sequence
        
        # Ensure the input X for LSTM has the correct shape and is numeric
        if not isinstance(X, np.ndarray):
            X_np = X.to_numpy() # Convert DataFrame/Series to numpy array
        else:
            X_np = X
        
        # Reshape for LSTM: (samples, timesteps, features)
        # Assuming X_np is (samples, features) and timesteps is 1 for this context
        X_lstm_input = X_np.reshape(X_np.shape[0], 1, X_np.shape[1])
        
        # Make prediction with the LSTM model
        embeddings = self.lstm_model.predict(X_lstm_input)
        
        return embeddings

    def get_feature_names_out(self, input_features=None):
        return [f"lstm_feature_{i}" for i in range(self.embedding_size)]


# Define the path for saving/loading models
MODEL_PATH = "backend/models/"

LSTM_MODEL_PATH = os.path.join(MODEL_PATH, "lstm_model.keras") # Will save Keras model here

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
        print(f"Attempting to load LSTM model from {LSTM_MODEL_PATH}...")
        try:
            lstm_model = keras.models.load_model(LSTM_MODEL_PATH)
            print("LSTM model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading LSTM model from {LSTM_MODEL_PATH}: {e}. This may indicate a corrupted file or an incompatible Keras version. LSTM embeddings will not be generated.")
            lstm_model = None
            return False
    else:
        print(f"LSTM model not found at {LSTM_MODEL_PATH}. It will be trained if data is available.")
        lstm_model = None
        return False

async def train_lstm_model(df_full_training_data, sequential_feature_engineer_instance, save_model=True):
    """
    Trains a simple LSTM model and returns it, along with the engineered sequential features
    for the entire dataset.
    """
    print("Starting LSTM model training and sequential feature generation...")
    if df_full_training_data.empty:
        print("No data found for LSTM training. Skipping LSTM training.")
        return None, None

    # Ensure 'STUDENT_ID' is available in the DataFrame for feature engineering
    if 'STUDENT_ID' not in df_full_training_data.columns:
        # If student_id is missing, generate dummy ones for training
        df_full_training_data['STUDENT_ID'] = [f"temp_student_{i}" for i in range(len(df_full_training_data))]
    
    # Ensure 'DAYS_OLD' is present, it's used in data generation
    if 'DAYS_OLD' not in df_full_training_data.columns:
        df_full_training_data['DAYS_OLD'] = 0 # Default if not present

    sim_days_for_lstm_training = 7 # Simulate 7 days of activity for LSTM training (reduced for faster training)
    
    # Generate synthetic activities for all students in df_full_training_data
    synthetic_activities_for_lstm_training = generate_all_students_activities(
        df_full_training_data[['STUDENT_ID', 'DAYS_OLD']], # Pass relevant columns for activity generation
        sim_days_for_lstm_training
    )
    
    # Engineer sequential features from the generated activities
    engineered_lstm_features_df = sequential_feature_engineer_instance.engineer_features(
        synthetic_activities_for_lstm_training
    )
    
    if engineered_lstm_features_df.empty:
        print("No engineered LSTM features found for training. Skipping LSTM training.")
        return None, None

    # Align features with df_full_training_data's student IDs
    # This is crucial if some students did not generate activities or if order is different
    # Make sure to reindex with the index of df_full_training_data to maintain sample order
    engineered_lstm_features_df = engineered_lstm_features_df.reindex(df_full_training_data['STUDENT_ID']).fillna(0)

    X_lstm = engineered_lstm_features_df.values
    
    # Reshape for LSTM: (samples, timesteps, features)
    # Since engineered_lstm_features_df already represents a single feature vector per student
    # for a given sequence length, the timesteps dimension will be 1.
    X_lstm = X_lstm.reshape(X_lstm.shape[0], 1, X_lstm.shape[1])

    # Define a dummy target for LSTM training, as it's primarily for feature extraction
    y_lstm = np.random.randint(0, 2, size=X_lstm.shape[0])

    lstm_feature_count = X_lstm.shape[2] # Store the actual feature count

    # Define LSTM model architecture with Bidirectional, Dropout, Batch Normalization
    model = keras.Sequential([
        keras.layers.Input(shape=(X_lstm.shape[1], lstm_feature_count)),
        keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True, activation='relu')),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Bidirectional(keras.layers.LSTM(32, activation='relu')),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid') # Binary classification for dummy target
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Fitting LSTM model...")
    # Increased epochs for better training
    model.fit(X_lstm, y_lstm, epochs=10, batch_size=32, verbose=0)
    
    print(f"LSTM model trained with input shape {X_lstm.shape}.")
    if save_model:
        try:
            model.save(LSTM_MODEL_PATH)
            print(f"LSTM model saved successfully to {LSTM_MODEL_PATH}")
        except Exception as e:
            print(f"Error saving LSTM model to {LSTM_MODEL_PATH}: {e}")
    # Get the embedding layer output (output of the second Bidirectional LSTM layer)
    embedding_model = keras.Model(inputs=model.input, outputs=model.layers[4].output)
    embeddings = embedding_model.predict(X_lstm)
    
    # Create a DataFrame from the embeddings
    precomputed_lstm_features_df = pd.DataFrame(embeddings, index=engineered_lstm_features_df.index,
                                                columns=[f"lstm_feature_{i}" for i in range(embeddings.shape[1])])
    
    return model, precomputed_lstm_features_df # Return the trained model and the precomputed embeddings

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

async def train_model(lstm_model_already_loaded: bool = False):
    global models, feature_names, model_performance_metrics, lstm_model
    
    # Initialize SequentialFeatureEngineer
    sequential_feature_engineer = SequentialFeatureEngineer(sequence_length_days=14)

    current_lstm_model = lstm_model # Initialize with the globally loaded LSTM model
    precomputed_lstm_features_df = pd.DataFrame() # Initialize an empty DataFrame
    lstm_output_features = 0 # Initialize

    # Define dynamic and static features for training - imported from dependencies
    # Ensure all dynamic and static features are in the DataFrame
    # Note: dynamic_features are now handled by LSTM and should not be duplicated here
    all_combined_features = ['STUDENT_ID', 'DAYS_OLD'] + [f.upper() for f in numerical_static_features] + \
                            [f.upper() for f in categorical_features] + [f.upper() for f in boolean_cols]

    # Create the models directory if it doesn't exist
    os.makedirs(MODEL_PATH, exist_ok=True)

    data = list(db["training_data"].find({}))
    if not data:
        print("No data found in MongoDB to train the models.")
        return
    print(f"Found {len(data)} documents for training.")


    df_full = pd.DataFrame(data).copy() # Explicitly copy to avoid SettingWithCopyWarning
    df_full.columns = df_full.columns.str.upper()
    # Handle duplicate columns by keeping the first occurrence
    df_full = df_full.loc[:,~df_full.columns.duplicated()].copy()
    
    # Ensure 'DAYS_OLD' column exists and is numeric in df_full
    if 'DAYS_OLD' not in df_full.columns:
        df_full['DAYS_OLD'] = 0 # Default value if not present in training data
    df_full['DAYS_OLD'] = pd.to_numeric(df_full['DAYS_OLD'], errors='coerce').fillna(0).astype(int)
    
    # Ensure 'STUDENT_ID' exists for LSTMFeatureExtractor
    if 'STUDENT_ID' not in df_full.columns:
        df_full['STUDENT_ID'] = [f"student_{i}" for i in range(len(df_full))]

    print(f"DataFrame columns after uppercase conversion: {df_full.columns.tolist()}")

    # Store overall metrics for nested CV
    overall_rf_metrics = {target: {"fold_metrics": []} for target in target_labels}

    # Outer loop for overall evaluation
    outer_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for outer_fold, (outer_train_index, outer_test_index) in enumerate(outer_skf.split(df_full, df_full[target_labels[0]])): # Assuming first target for splitting
        print(f"\n--- Starting Outer Fold {outer_fold + 1} ---")
        df_outer_train = df_full.iloc[outer_train_index].copy()
        df_final_test = df_full.iloc[outer_test_index].copy()

        # Pre-compute LSTM features for the entire dataset once
        # Pre-compute LSTM features for the entire dataset once, if not already loaded
        if not lstm_model_already_loaded:
            if outer_fold == 0: # Only do this once before the outer loop starts
                current_lstm_model, precomputed_lstm_features_df = await train_lstm_model(df_full, sequential_feature_engineer)
                if precomputed_lstm_features_df is None:
                    print("Warning: No precomputed LSTM features. Proceeding without LSTM features.")
                    lstm_output_features = 0
                else:
                    lstm_output_features = precomputed_lstm_features_df.shape[1]
                    # Merge precomputed LSTM features into the full DataFrame
                    df_full = df_full.set_index('STUDENT_ID').join(precomputed_lstm_features_df, how='left', rsuffix='_lstm').reset_index()
                    # Fill any NaN values that might result from the join (e.g., if a student had no activities)
                    for col in precomputed_lstm_features_df.columns:
                        df_full[col] = df_full[col].fillna(0)
                    print(f"Precomputed {lstm_output_features} LSTM features for {len(df_full)} students.")
        else:
            # If LSTM model is already loaded, we assume its features are integrated or not needed for this path
            # This part might need more sophisticated handling if LSTM features are dynamically generated
            # and not precomputed globally. For now, we'll assume they are handled.
            print("LSTM model already loaded, skipping re-training and re-computation of its features.")
            lstm_output_features = lstm_model.output_shape[-1] if lstm_model else 0 # Get output features from loaded model
            # For now, we'll assume precomputed_lstm_features_df is populated elsewhere if needed, or that
            # the feature_preprocessor handles the already-loaded LSTM model directly.
            # This part needs careful consideration based on the exact flow of data.
            # For now, we'll proceed assuming precomputed_lstm_features_df is handled.
            # If df_full needs to be updated with LSTM features, that logic needs to be here.
            # For simplicity, we'll assume the feature_preprocessor can work with the global lstm_model.
            precomputed_lstm_features_df = pd.DataFrame(index=df_full['STUDENT_ID'].unique(), columns=[f"lstm_feature_{i}" for i in range(lstm_output_features)])
            df_full = df_full.set_index('STUDENT_ID').join(precomputed_lstm_features_df, how='left', rsuffix='_lstm').reset_index()
            for col in precomputed_lstm_features_df.columns:
                df_full[col] = df_full[col].fillna(0)

        df_outer_train = df_full.iloc[outer_train_index].copy()
        df_final_test = df_full.iloc[outer_test_index].copy()
        
        for target in target_labels:
            print(f"Processing target: {target}")
            
            # Prepare X and y for the current outer training fold
            # X_outer_train now includes the precomputed LSTM features
            # Combine static, categorical, boolean, and LSTM features for training
            features_for_training = numerical_static_features + categorical_features + boolean_cols + list(precomputed_lstm_features_df.columns)
            X_outer_train = df_outer_train[features_for_training].copy()
            # Convert target labels to numerical (0 or 1)
            # Ensure robustness by converting to string, stripping whitespace, and lowercasing
            y_outer_train = df_outer_train.loc[X_outer_train.index, target].astype(str).str.strip().str.lower().apply(
                lambda x: 1 if x == target.split('_VS_')[0].lower() else 0
            ).astype(int)

            # Handle missing target values in y_outer_train
            if y_outer_train.dtype == 'int8' and (y_outer_train == -1).any():
                original_len = len(X_outer_train)
                X_outer_train = X_outer_train[y_outer_train != -1]
                y_outer_train = y_outer_train[y_outer_train != -1]
                if len(X_outer_train) < original_len:
                    print(f"Dropped {original_len - len(X_outer_train)} rows due to NaN/missing target values for {target} in outer training set.")

            if X_outer_train.shape[0] == 0 or X_outer_train.shape[0] != y_outer_train.shape[0]:
                print(f"Not enough valid data in outer training set for {target}. Skipping.")
                continue

            unique_labels_outer_train = y_outer_train.nunique()
            if unique_labels_outer_train < 2:
                print(f"Skipping training for {target} in outer fold {outer_fold + 1}: Not enough unique labels ({unique_labels_outer_train}) for classification.")
                continue

            # Inner loop for hyperparameter tuning and model training
            param_space = {
                'classifier__n_estimators': Integer(50, 200),
                'classifier__max_features': Categorical(['sqrt', 'log2', None]),
                'classifier__max_depth': Integer(5, 50),
                'classifier__min_samples_split': Integer(2, 20),
                'classifier__min_samples_leaf': Integer(1, 10),
                'smote__k_neighbors': Integer(1, 10) # For SMOTE
            }
 
            # Define a preprocessor for the static and dynamic features, and now including precomputed LSTM features
            feature_preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_static_features), # Numerical static features
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features), # Categorical static features
                    ('boolean', 'passthrough', boolean_cols), # Pass through boolean columns
                    # Include precomputed LSTM features directly as numerical features if they exist
                    ('lstm_features', numerical_transformer, list(precomputed_lstm_features_df.columns))
                ],
                remainder='drop' # Drop any other columns not explicitly handled
            )
 
            # Create a base pipeline for BayesSearchCV
            base_pipeline = ImblearnPipeline(steps=[('feature_preprocessor', feature_preprocessor),\
                                                     ('smote', SMOTE(random_state=42)),\
                                                     ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))])
 
            f1_scorer = make_scorer(f1_score, average='weighted', zero_division=0)
 
            opt = BayesSearchCV(
                estimator=base_pipeline,
                search_spaces=param_space,
                n_iter=10, # Further reduced n_iter for faster execution during nested CV
                cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42), # Further reduced n_splits for faster inner CV
                scoring=f1_scorer,
                n_jobs=-1, # Use all available cores
                random_state=42,
                verbose=0 # Suppress verbose output for inner loop
            )
 
            print(f"Starting inner loop hyperparameter tuning for {target} in outer fold {outer_fold + 1}...")
            opt.fit(X_outer_train, y_outer_train)
 
            print(f"Inner loop tuning for {target} completed. Best parameters: {opt.best_params_}")
            
            # Retrain final model on the entire outer_training_set with best hyperparameters
            final_model_pipeline = opt.best_estimator_
            
            # Evaluate on the final_test_set (from the outer loop)
            # X_final_test now includes the precomputed LSTM features
            # Combine static, categorical, boolean, and LSTM features for testing
            X_final_test = df_final_test[features_for_training].copy() # Use the same feature list as training
            # Convert target labels to numerical (0 or 1)
            # Ensure robustness by converting to string, stripping whitespace, and lowercasing
            y_final_test = df_final_test.loc[X_final_test.index, target].astype(str).str.strip().str.lower().apply(
                lambda x: 1 if x == target.split('_VS_')[0].lower() else 0
            ).astype(int)

            # Handle missing target values in y_final_test
            if y_final_test.dtype == 'int8' and (y_final_test == -1).any():
                original_len = len(X_final_test)
                X_final_test = X_final_test[y_final_test != -1]
                y_final_test = y_final_test[y_final_test != -1]
                if len(X_final_test) < original_len:
                    print(f"Dropped {original_len - len(X_final_test)} rows due to NaN/missing target values for {target} in final test set.")

            if X_final_test.shape[0] == 0 or X_final_test.shape[0] != y_final_test.shape[0]:
                print(f"Not enough valid data in final test set for {target}. Skipping evaluation for this target in outer fold {outer_fold + 1}.")
                continue
            
            y_pred_final = final_model_pipeline.predict(X_final_test)
            
            accuracy = accuracy_score(y_final_test, y_pred_final)
            precision = precision_score(y_final_test, y_pred_final, average='weighted', zero_division=0)
            recall = recall_score(y_final_test, y_pred_final, average='weighted', zero_division=0)
            f1 = f1_score(y_final_test, y_pred_final, average='weighted', zero_division=0)
            f1_per_class = f1_score(y_final_test, y_pred_final, average=None, zero_division=0)
            
            print(f"Outer Fold {outer_fold + 1} - {target} - Test Metrics: Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}")
            
            # Store metrics for aggregation
            overall_rf_metrics[target]["fold_metrics"].append({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "best_params": convert_numpy_types(opt.best_params_),
                "confusion_matrix": confusion_matrix(y_final_test, y_pred_final).tolist(),
                "class_distribution": convert_numpy_types(y_final_test.value_counts().to_dict()),
                "f1_per_class": f1_per_class.tolist()
            })
            
            # Save the trained model for this outer fold (optional, could save only the best overall)
            # For simplicity, we'll just keep the last trained model in the global `models` dict
            models["random_forest"][target] = final_model_pipeline
            
            # Get feature importances for Random Forest
            classifier_rf = final_model_pipeline.named_steps['classifier']
            if hasattr(classifier_rf, 'feature_importances_'):
                importances_rf = classifier_rf.feature_importances_
                fitted_preprocessor_rf = final_model_pipeline.named_steps['feature_preprocessor']
                all_processed_feature_names_rf = fitted_preprocessor_rf.get_feature_names_out()
                feature_importances_dict["random_forest"][target] = dict(zip(all_processed_feature_names_rf, importances_rf))
                print(f"Feature importances for Random Forest - {target} calculated and stored.")
            else:
                print(f"Random Forest classifier does not have feature_importances_ attribute.")

    # Aggregate metrics from all outer folds
    for target in target_labels:
        if overall_rf_metrics[target]["fold_metrics"]:
            avg_accuracy = np.mean([m["accuracy"] for m in overall_rf_metrics[target]["fold_metrics"]])
            avg_precision = np.mean([m["precision"] for m in overall_rf_metrics[target]["fold_metrics"]])
            avg_recall = np.mean([m["recall"] for m in overall_rf_metrics[target]["fold_metrics"]])
            avg_f1 = np.mean([m["f1_score"] for m in overall_rf_metrics[target]["fold_metrics"]])
            
            # For overall confusion matrix, sum up individual confusion matrices
            summed_cm = np.sum([m["confusion_matrix"] for m in overall_rf_metrics[target]["fold_metrics"]], axis=0).tolist()
            
            # For class distribution, sum up individual class distributions
            summed_class_dist = {}
            for m in overall_rf_metrics[target]["fold_metrics"]:
                for cls, count in m["class_distribution"].items():
                    summed_class_dist[cls] = summed_class_dist.get(cls, 0) + count

            model_performance_metrics["random_forest"][target] = {
                "training_accuracy": "N/A (Nested CV)", # Training accuracy is for inner loop, not overall
                "test_accuracy": float(avg_accuracy),
                "precision": float(avg_precision),
                "recall": float(avg_recall),
                "f1_score": float(avg_f1),
                "model_type": "Random Forest (Nested CV Tuned)",
                "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_size": len(df_full),
                "features_used": len(numerical_static_features) + len(categorical_features) + len(boolean_cols) + 64, # 64 for LSTM features
                "cross_validation_folds": outer_skf.get_n_splits(df_full, df_full[target_labels[0]]),
                "best_params": {f"Fold {i+1}": m["best_params"] for i, m in enumerate(overall_rf_metrics[target]["fold_metrics"])},
                "fold_accuracies": [m["accuracy"] for m in overall_rf_metrics[target]["fold_metrics"]],
                "fold_precisions": [m["precision"] for m in overall_rf_metrics[target]["fold_metrics"]],
                "fold_recalls": [m["recall"] for m in overall_rf_metrics[target]["fold_metrics"]],
                "fold_f1_scores": [m["f1_score"] for m in overall_rf_metrics[target]["fold_metrics"]],
                "confusion_matrices_per_fold": [summed_cm], # Store the summed confusion matrix
                "class_distribution": convert_numpy_types(summed_class_dist)
            }
            print(f"Overall RF Metrics for {target}: Acc: {avg_accuracy:.4f}, Prec: {avg_precision:.4f}, Rec: {avg_recall:.4f}, F1: {avg_f1:.4f}")
            
            # Save the final aggregated model and its associated metrics and feature importances
            model_filename = os.path.join(MODEL_PATH, f"model_RF_{target}.joblib")
            model_data_to_save = {
                'model': models["random_forest"][target],
                'performance_metrics': model_performance_metrics["random_forest"][target],
                'feature_importances': feature_importances_dict["random_forest"][target]
            }
            joblib.dump(model_data_to_save, model_filename)
            print(f"Final RF Model and aggregated data for {target} saved to {model_filename}")
        else:
            print(f"No sufficient data to calculate overall RF metrics for {target}.")
 
async def train_xgboost_model(lstm_model_already_loaded: bool = False):
    global models, feature_names, model_performance_metrics, lstm_model
    
    # Initialize SequentialFeatureEngineer
    sequential_feature_engineer = SequentialFeatureEngineer(sequence_length_days=14)



    # Define dynamic and static features for training - imported from dependencies
    # Ensure all dynamic and static features are in the DataFrame
    all_combined_features = ['STUDENT_ID', 'DAYS_OLD'] + [f.upper() for f in numerical_static_features] + \
                            [f.upper() for f in categorical_features] + [f.upper() for f in boolean_cols]
 
    # Create the models directory if it doesn't exist
    os.makedirs(MODEL_PATH, exist_ok=True)
 
    data = list(db["training_data"].find({}))
    if not data:
        print("No data found in MongoDB to train the models.")
        return
    print(f"Found {len(data)} documents for training.")
 
 
    df_full = pd.DataFrame(data).copy() # Explicitly copy to avoid SettingWithCopyWarning
    df_full.columns = df_full.columns.str.upper()
    # Handle duplicate columns by keeping the first occurrence
    df_full = df_full.loc[:,~df_full.columns.duplicated()].copy()
    
    # Ensure 'DAYS_OLD' column exists and is numeric in df_full
    if 'DAYS_OLD' not in df_full.columns:
        df_full['DAYS_OLD'] = 0 # Default value if not present in training data
    df_full['DAYS_OLD'] = pd.to_numeric(df_full['DAYS_OLD'], errors='coerce').fillna(0).astype(int)
    
    # Ensure 'STUDENT_ID' exists for LSTMFeatureExtractor
    if 'STUDENT_ID' not in df_full.columns:
        df_full['STUDENT_ID'] = [f"student_{i}" for i in range(len(df_full))]

    print(f"DataFrame columns after uppercase conversion: {df_full.columns.tolist()}")
 
    # Store overall metrics for nested CV
    overall_xgb_metrics = {target: {"fold_metrics": []} for target in target_labels}
 
    # Outer loop for overall evaluation
    outer_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
 
    for outer_fold, (outer_train_index, outer_test_index) in enumerate(outer_skf.split(df_full, df_full[target_labels[0]])): # Assuming first target for splitting
        print(f"\n--- Starting Outer Fold {outer_fold + 1} ---")
        df_outer_train = df_full.iloc[outer_train_index].copy()
        df_final_test = df_full.iloc[outer_test_index].copy()
 
        # Pre-compute LSTM features for the entire dataset once
        # Pre-compute LSTM features for the entire dataset once, if not already loaded
        if not lstm_model_already_loaded:
            if outer_fold == 0: # Only do this once before the outer loop starts
                current_lstm_model, precomputed_lstm_features_df = await train_lstm_model(df_full, sequential_feature_engineer)
                if precomputed_lstm_features_df is None:
                    print("Warning: No precomputed LSTM features. Proceeding without LSTM features.")
                    lstm_output_features = 0
                else:
                    lstm_output_features = precomputed_lstm_features_df.shape[1]
                    # Merge precomputed LSTM features into the full DataFrame
                    df_full = df_full.set_index('STUDENT_ID').join(precomputed_lstm_features_df, how='left', rsuffix='_lstm').reset_index()
                    # Fill any NaN values that might result from the join (e.g., if a student had no activities)
                    for col in precomputed_lstm_features_df.columns:
                        df_full[col] = df_full[col].fillna(0)
                    print(f"Precomputed {lstm_output_features} LSTM features for {len(df_full)} students.")
        else:
            # If LSTM model is already loaded, we assume its features are integrated or not needed for this path
            print("LSTM model already loaded, skipping re-training and re-computation of its features.")
            lstm_output_features = lstm_model.output_shape[-1] if lstm_model else 0
            precomputed_lstm_features_df = pd.DataFrame(index=df_full['STUDENT_ID'].unique(), columns=[f"lstm_feature_{i}" for i in range(lstm_output_features)])
            df_full = df_full.set_index('STUDENT_ID').join(precomputed_lstm_features_df, how='left', rsuffix='_lstm').reset_index()
            for col in precomputed_lstm_features_df.columns:
                df_full[col] = df_full[col].fillna(0)

        df_outer_train = df_full.iloc[outer_train_index].copy()
        df_final_test = df_full.iloc[outer_test_index].copy()
        
        for target in target_labels:
            print(f"Processing target: {target}")
            
            # Prepare X and y for the current outer training fold
            # X_outer_train now includes the precomputed LSTM features
            # Combine static, categorical, boolean, and LSTM features for training
            features_for_training = numerical_static_features + categorical_features + boolean_cols + list(precomputed_lstm_features_df.columns)
            X_outer_train = df_outer_train[features_for_training].copy()
            # Convert target labels to numerical (0 or 1)
            # Ensure robustness by converting to string, stripping whitespace, and lowercasing
            y_outer_train = df_outer_train.loc[X_outer_train.index, target].astype(str).str.strip().str.lower().apply(
                lambda x: 1 if x == target.split('_VS_')[0].lower() else 0
            ).astype(int)

            # Handle missing target values in y_outer_train
            if y_outer_train.dtype == 'int8' and (y_outer_train == -1).any():
                original_len = len(X_outer_train)
                X_outer_train = X_outer_train[y_outer_train != -1]
                y_outer_train = y_outer_train[y_outer_train != -1]
                if len(X_outer_train) < original_len:
                    print(f"Dropped {original_len - len(X_outer_train)} rows due to NaN/missing target values for {target} in outer training set.")

            if X_outer_train.shape[0] == 0 or X_outer_train.shape[0] != y_outer_train.shape[0]:
                print(f"Not enough valid data in outer training set for {target}. Skipping.")
                continue

            unique_labels_outer_train = y_outer_train.nunique()
            if unique_labels_outer_train < 2:
                print(f"Skipping training for {target} in outer fold {outer_fold + 1}: Not enough unique labels ({unique_labels_outer_train}) for classification.")
                continue

            # Inner loop for hyperparameter tuning and model training
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
 
            # Define a preprocessor for the static and dynamic features, and now including precomputed LSTM features
            feature_preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_static_features), # Numerical static features
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features), # Categorical static features
                    ('boolean', 'passthrough', boolean_cols), # Pass through boolean columns
                    # Include precomputed LSTM features directly as numerical features if they exist
                    ('lstm_features', numerical_transformer, list(precomputed_lstm_features_df.columns))
                ],
                remainder='drop' # Drop any other columns not explicitly handled
            )
 
            # Calculate scale_pos_weight for XGBoost
            neg_count = (y_outer_train == 0).sum()
            pos_count = (y_outer_train == 1).sum()
            scale_pos_weight_value = neg_count / pos_count if pos_count > 0 else 1

            # Create a base pipeline for BayesSearchCV
            base_pipeline = ImblearnPipeline(steps=[('feature_preprocessor', feature_preprocessor),\
                                                     ('smote', SMOTE(random_state=42)),\
                                                     ('classifier', XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=scale_pos_weight_value))]) # Changed for XGBoost
 
            f1_scorer = make_scorer(f1_score, average='weighted', zero_division=0)
 
            opt = BayesSearchCV(
                estimator=base_pipeline,
                search_spaces=param_space,
                n_iter=10, # Further reduced n_iter for faster execution during nested CV
                cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42), # Further reduced n_splits for faster inner CV
                scoring=f1_scorer,
                n_jobs=-1, # Use all available cores
                random_state=42,
                verbose=0 # Suppress verbose output for inner loop
            )
 
            print(f"Starting inner loop hyperparameter tuning for {target} in outer fold {outer_fold + 1}...")
            opt.fit(X_outer_train, y_outer_train)
 
            print(f"Inner loop tuning for {target} completed. Best parameters: {opt.best_params_}")
            
            # Retrain final model on the entire outer_training_set with best hyperparameters
            final_model_pipeline = opt.best_estimator_
            
            # Evaluate on the final_test_set (from the outer loop)
            # X_final_test now includes the precomputed LSTM features
            # Combine static, categorical, boolean, and LSTM features for testing
            X_final_test = df_final_test[features_for_training].copy() # Use the same feature list as training
            # Convert target labels to numerical (0 or 1)
            # Ensure robustness by converting to string, stripping whitespace, and lowercasing
            y_final_test = df_final_test.loc[X_final_test.index, target].astype(str).str.strip().str.lower().apply(
                lambda x: 1 if x == target.split('_VS_')[0].lower() else 0
            ).astype(int)

            # Handle missing target values in y_final_test
            if y_final_test.dtype == 'int8' and (y_final_test == -1).any():
                original_len = len(X_final_test)
                X_final_test = X_final_test[y_final_test != -1]
                y_final_test = y_final_test[y_final_test != -1]
                if len(X_final_test) < original_len:
                    print(f"Dropped {original_len - len(X_final_test)} rows due to NaN/missing target values for {target} in final test set.")

            if X_final_test.shape[0] == 0 or X_final_test.shape[0] != y_final_test.shape[0]:
                print(f"Not enough valid data in final test set for {target}. Skipping evaluation for this target in outer fold {outer_fold + 1}.")
                continue
            
            y_pred_final = final_model_pipeline.predict(X_final_test)
            
            accuracy = accuracy_score(y_final_test, y_pred_final)
            precision = precision_score(y_final_test, y_pred_final, average='weighted', zero_division=0)
            recall = recall_score(y_final_test, y_pred_final, average='weighted', zero_division=0)
            f1 = f1_score(y_final_test, y_pred_final, average='weighted', zero_division=0)
            f1_per_class = f1_score(y_final_test, y_pred_final, average=None, zero_division=0)
            
            print(f"Outer Fold {outer_fold + 1} - {target} - Test Metrics: Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}")
            
            # Store metrics for aggregation
            overall_xgb_metrics[target]["fold_metrics"].append({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "best_params": convert_numpy_types(opt.best_params_),
                "confusion_matrix": confusion_matrix(y_final_test, y_pred_final).tolist(),
                "class_distribution": convert_numpy_types(y_final_test.value_counts().to_dict()),
                "f1_per_class": f1_per_class.tolist()
            })
            
            # Save the trained model for this outer fold (optional, could save only the best overall)
            # For simplicity, we'll just keep the last trained model in the global `models` dict
            models["xgboost"][target] = final_model_pipeline
            
            # Get feature importances for XGBoost
            classifier_xgb = final_model_pipeline.named_steps['classifier']
            if hasattr(classifier_xgb, 'feature_importances_'):
                importances_xgb = classifier_xgb.feature_importances_
                fitted_preprocessor_xgb = final_model_pipeline.named_steps['feature_preprocessor']
                all_processed_feature_names_xgb = fitted_preprocessor_xgb.get_feature_names_out()
                feature_importances_dict["xgboost"][target] = dict(zip(all_processed_feature_names_xgb, importances_xgb))
                print(f"Feature importances for XGBoost - {target} calculated and stored.")
            else:
                print(f"XGBoost classifier does not have feature_importances_ attribute.")

    # Aggregate metrics from all outer folds
    for target in target_labels:
        if overall_xgb_metrics[target]["fold_metrics"]:
            avg_accuracy = np.mean([m["accuracy"] for m in overall_xgb_metrics[target]["fold_metrics"]])
            avg_precision = np.mean([m["precision"] for m in overall_xgb_metrics[target]["fold_metrics"]])
            avg_recall = np.mean([m["recall"] for m in overall_xgb_metrics[target]["fold_metrics"]])
            avg_f1 = np.mean([m["f1_score"] for m in overall_xgb_metrics[target]["fold_metrics"]])
            
            # For overall confusion matrix, sum up individual confusion matrices
            summed_cm = np.sum([m["confusion_matrix"] for m in overall_xgb_metrics[target]["fold_metrics"]], axis=0).tolist()
            
            # For class distribution, sum up individual class distributions
            summed_class_dist = {}
            for m in overall_xgb_metrics[target]["fold_metrics"]:
                for cls, count in m["class_distribution"].items():
                    summed_class_dist[cls] = summed_class_dist.get(cls, 0) + count

            model_performance_metrics["xgboost"][target] = {
                "training_accuracy": "N/A (Nested CV)", # Training accuracy is for inner loop, not overall
                "test_accuracy": float(avg_accuracy),
                "precision": float(avg_precision),
                "recall": float(avg_recall),
                "f1_score": float(avg_f1),
                "model_type": "XGBoost (Nested CV Tuned)",
                "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_size": len(df_full),
                "features_used": len(numerical_static_features) + len(categorical_features) + len(boolean_cols) + 64, # 64 for LSTM features
                "cross_validation_folds": outer_skf.get_n_splits(df_full, df_full[target_labels[0]]),
                "best_params": {f"Fold {i+1}": m["best_params"] for i, m in enumerate(overall_xgb_metrics[target]["fold_metrics"])},
                "fold_accuracies": [m["accuracy"] for m in overall_xgb_metrics[target]["fold_metrics"]],
                "fold_precisions": [m["precision"] for m in overall_xgb_metrics[target]["fold_metrics"]],
                "fold_recalls": [m["recall"] for m in overall_xgb_metrics[target]["fold_metrics"]],
                "fold_f1_scores": [m["f1_score"] for m in overall_xgb_metrics[target]["fold_metrics"]],
                "confusion_matrices_per_fold": [summed_cm], # Store the summed confusion matrix
                "class_distribution": convert_numpy_types(summed_class_dist)
            }
            print(f"Overall XGB Metrics for {target}: Acc: {avg_accuracy:.4f}, Prec: {avg_precision:.4f}, Rec: {avg_recall:.4f}, F1: {avg_f1:.4f}")
            
            # Save the final aggregated model and its associated metrics and feature importances
            model_filename = os.path.join(MODEL_PATH, f"model_XGB_{target}.joblib")
            model_data_to_save = {
                'model': models["xgboost"][target],
                'performance_metrics': model_performance_metrics["xgboost"][target],
                'feature_importances': feature_importances_dict["xgboost"][target]
            }
            joblib.dump(model_data_to_save, model_filename)
            print(f"Final XGB Model and aggregated data for {target} saved to {model_filename}")
        else:
            print(f"No sufficient data to calculate overall XGB metrics for {target}.")
 
@app.on_event("startup")
async def startup_event():
    # Check if models are already trained and saved
    # If not, train them. Otherwise, load them.
    
    # Attempt to load LSTM model first
    lstm_model_successfully_loaded = load_lstm_model()
 
    # Define model types and their corresponding training functions
    model_types = ["random_forest", "xgboost"]
    training_functions = {
        "random_forest": train_model,
        "xgboost": train_xgboost_model
    }
 
    for model_type in model_types:
        for target in target_labels:
            model_filename = os.path.join(MODEL_PATH, f"model_{model_type.upper()}_{target}.joblib")
            
            model_needs_training = False
            if os.path.exists(model_filename):
                print(f"Attempting to load {model_type.replace('_', ' ').upper()} model for {target} from {model_filename}...")
                try:
                    loaded_data = joblib.load(model_filename)
                    models[model_type][target] = loaded_data['model']
                    model_performance_metrics[model_type][target] = loaded_data['performance_metrics']
                    feature_importances_dict[model_type][target] = loaded_data['feature_importances']
                    print(f"Loaded {model_type.replace('_', ' ').upper()} model for {target}.")
                except Exception as e:
                    print(f"Error loading {model_type.replace('_', ' ').upper()} model for {target}: {e}. Retraining this model.")
                    model_needs_training = True
            else:
                print(f"{model_type.replace('_', ' ').upper()} model for {target} not found at {model_filename}. Training this model.")
                model_needs_training = True

            if model_needs_training:
                print(f"Initiating training for {model_type.replace('_', ' ').upper()} model for {target}...")
                # To train only the specific target, we need to modify train_model and train_xgboost_model
                # For now, these functions train all targets. We will call them if any target needs training.
                # A more refined approach would be to train only the missing target.
                # For simplicity in this fix, if any target needs training, the whole model type training is triggered.
                await training_functions[model_type](lstm_model_already_loaded=lstm_model_successfully_loaded)
                # After training, attempt to reload to ensure it's available
                try:
                    loaded_data = joblib.load(model_filename)
                    models[model_type][target] = loaded_data['model']
                    model_performance_metrics[model_type][target] = loaded_data['performance_metrics']
                    feature_importances_dict[model_type][target] = loaded_data['feature_importances']
                    print(f"Successfully reloaded trained {model_type.replace('_', ' ').upper()} model for {target}.")
                except Exception as e:
                    print(f"Error reloading {model_type.replace('_', ' ').upper()} model for {target} after training: {e}")
                    # If it fails to reload even after training, there's a serious issue.
                    # This might require manual intervention or a more robust error handling strategy.
 
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
        # Convert all keys to uppercase to match the training data format
        student_data_upper = {k.upper(): v for k, v in student_data.items()}
        input_df = pd.DataFrame([student_data_upper])
 
        # Define dynamic and static features for prediction - imported from dependencies
        all_combined_features = [f.upper() for f in numerical_static_features] + [f.upper() for f in categorical_features] + [f.upper() for f in boolean_cols]

        # Ensure all expected columns are present in the input_df, fill missing with NaN
        for col in all_combined_features:
            if col not in input_df.columns:
                input_df[col] = np.nan
 
        # Convert boolean fields to numerical (0 or 1)
        for col in boolean_cols:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(int)
 
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
 
        # Add STUDENT_ID to input_df for LSTMFeatureExtractor
        if 'STUDENT_ID' not in input_df.columns:
            input_df['STUDENT_ID'] = 'sim_predict_student' # Dummy ID for prediction if not provided

        # Make predictions for all models
        predictions = {}
        for model_type, models_by_type in models.items():
            for target, model_pipeline in models_by_type.items():
                prediction = model_pipeline.predict(input_df)
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