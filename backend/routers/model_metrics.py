from fastapi import APIRouter, HTTPException
from sklearn.metrics import confusion_matrix, accuracy_score
from backend.dependencies import models, target_labels, db, boolean_cols, numerical_features, categorical_features
import pandas as pd
import numpy as np

router = APIRouter()

@router.get("/confusion-matrix")
async def get_confusion_matrix():
    print("DEBUG: Entering get_confusion_matrix function.")
    if not models:
        print("DEBUG: Models not trained yet. Raising 404.")
        raise HTTPException(status_code=404, detail="Models not trained yet.")
    
    metrics = {}
    print("DEBUG: Fetching training data from MongoDB.")
    data = list(db["training_data"].find({}))
    if not data:
        print("DEBUG: No data found in MongoDB. Raising 404.")
        raise HTTPException(status_code=404, detail="No data found in MongoDB to train the models.")

    df = pd.DataFrame(data).copy() # Explicitly copy to avoid SettingWithCopyWarning
    df.columns = df.columns.str.upper()
    # Handle duplicate columns by keeping the first occurrence
    print(f"DEBUG: Found {len(data)} documents. Processing DataFrame.")
    df = pd.DataFrame(data).copy() # Explicitly copy to avoid SettingWithCopyWarning
    df.columns = df.columns.str.upper()
    # Handle duplicate columns by keeping the first occurrence
    df = df.loc[:,~df.columns.duplicated()].copy()

    # Convert boolean fields to numerical (0 or 1)
    print("DEBUG: Converting boolean columns to numerical.")
    for col in boolean_cols:
        if col in df.columns:
            df.loc[:, col] = df[col].astype(int)

    print("DEBUG: Processing numerical features.")
    # Numerical feature imputation is now handled by the preprocessor_obj
    # Ensure numerical columns are numeric before passing to preprocessor,
    # but let SimpleImputer handle NaNs.
    for col in numerical_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # If the column is missing, SimpleImputer in preprocessor_obj will handle it
        # by filling with the mean from training data, so no need for manual 0 fill here.
        # If a column is entirely missing from the dataframe, preprocessor_obj will handle
        # it by adding a column of zeros if 'passthrough' or mean if SimpleImputer.
        # No need for explicit 'df[col] = 0' here.

    print("DEBUG: Processing categorical features.")
    # Explicitly convert categorical features to string, filling NaNs with 'missing_category'
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('missing_category')
        else:
            # Add missing categorical columns with default 'missing_category'
            df[col] = 'missing_category'
            
    # Define features (X) by selecting numerical and categorical features
    # This should be done *after* the cleaning steps above
    print(f"DEBUG (model_metrics): df columns before X selection: {df.columns.tolist()}")
    X = df[numerical_features + categorical_features]

    # X.dropna() is no longer needed as SimpleImputer in preprocessor_obj handles NaNs
    # print("DEBUG: Dropping rows with NaN values from X.")
    # X = X.dropna()
    print(f"DEBUG: X shape before prediction: {X.shape}")

    print("DEBUG: Iterating through target labels for model predictions.")
    for target in target_labels:
        if target in models:
            print(f"DEBUG: Processing target: {target}")
            # Ensure y_true is consistent with the numerical target values (0 or 1) the models were trained on
            # If the original dataframe's target column is already numerical (int), use it directly.
            # Otherwise, convert it to numerical using the same logic as in main.py during training.
            if df[target].dtype == 'object':
                y_true = df.loc[X.index, target].astype('category').cat.codes
            else:
                y_true = df.loc[X.index, target].astype(int) # Ensure it's integer type
            
            # Apply the preprocessor to X before prediction
            print(f"DEBUG: Predicting for target {target}.")
            # Ensure X has the same index as y_true after any potential row dropping
            y_pred = models[target].predict(X.loc[y_true.index]) # Predict on the original dataset, pipeline handles preprocessing
            
            # Calculate Confusion Matrix
            print(f"DEBUG: Calculating confusion matrix for {target}.")
            cm = confusion_matrix(y_true, y_pred)
            
            # Calculate Accuracy
            print(f"DEBUG: Calculating accuracy for {target}.")
            accuracy = accuracy_score(y_true, y_pred)
            
            metrics[target] = {
                "confusion_matrix": cm.tolist(),
                "accuracy": accuracy
            }
        else:
            print(f"DEBUG: Model not found for target: {target}. Skipping.")
    
    if not metrics:
        print("DEBUG: No metrics calculated. Raising 404.")
        raise HTTPException(status_code=404, detail="Metrics not calculated yet.")

    print("DEBUG: Returning calculated metrics.")
    return metrics