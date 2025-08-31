from fastapi import APIRouter, HTTPException
from sklearn.metrics import confusion_matrix
from backend.dependencies import models, target_labels, db, boolean_cols, numerical_features, categorical_features
import pandas as pd

router = APIRouter()

@router.get("/confusion-matrix")
async def get_confusion_matrix():
    if not models:
        raise HTTPException(status_code=404, detail="Models not trained yet.")
    
    confusion_matrices = {}
    data = list(db["training_data"].find({}))
    if not data:
        raise HTTPException(status_code=404, detail="No data found in MongoDB to train the models.")

    df = pd.DataFrame(data)
    df.columns = df.columns.str.upper()

    # Convert boolean fields to numerical (0 or 1)
    for col in boolean_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 1 if x else 0)

    X = df[numerical_features + categorical_features]
    for col in numerical_features:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X.dropna(inplace=True)

    for target in target_labels:
        if target in models:
            y_true = df.loc[X.index, target].astype('category').cat.codes # Ensure y_true is numerical
            y_pred = models[target].predict(X) # Predict on the full dataset for confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            confusion_matrices[target] = cm.tolist()
    
    if not confusion_matrices:
        raise HTTPException(status_code=404, detail="Confusion matrices not calculated yet.")

    return confusion_matrices