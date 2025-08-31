from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import os

app = FastAPI()

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client["student_db"]
students_collection = db["students"]

# Global variable for the model and features
model = None
features = None

async def train_model():
    global model, features
    # Fetch data from MongoDB
    data = list(students_collection.find({}))
    if not data:
        print("No data found in MongoDB to train the model.")
        return

    df = pd.DataFrame(data)

    # Ensure all required columns exist
    required_columns = ['gpa', 'attendance', 'extracurriculars', 'study_hours', 'risk_level']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column for training: {col}")

    # Convert 'extracurriculars' to numerical (e.g., 0 for No, 1 for Yes)
    df['extracurriculars'] = df['extracurriculars'].apply(lambda x: 1 if x == 'Yes' else 0)

    # Define features (X) and target (y)
    features = ['gpa', 'attendance', 'extracurriculars', 'study_hours']
    target = 'risk_level'

    X = df[features]
    y = df[target]

    # Handle potential non-numeric values gracefully by converting to numeric, coercing errors
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    # Drop rows with NaN values that resulted from coercion
    X.dropna(inplace=True)
    y.dropna(inplace=True)

    # Ensure X and y have the same number of samples after dropping NaNs
    if X.shape[0] == 0 or X.shape[0] != y.shape[0]:
        print("Not enough valid data after cleaning to train the model.")
        return

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForestClassifier model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model trained successfully.")

@app.on_event("startup")
async def startup_event():
    await train_model()

@app.get("/")
async def read_root():
    return {"message": "FastAPI backend is running"}

@app.post("/predict")
async def predict_risk(student_data: dict):
    if model is None or features is None:
        raise HTTPException(status_code=500, detail="Model not trained yet.")

    try:
        # Prepare input data for prediction
        input_df = pd.DataFrame([student_data])

        # Ensure all required features are present and in the correct order
        for col in features:
            if col not in input_df.columns:
                raise HTTPException(status_code=400, detail=f"Missing feature: {col}")
            # Convert 'extracurriculars' to numerical if present
            if col == 'extracurriculars':
                input_df[col] = input_df[col].apply(lambda x: 1 if x == 'Yes' else 0)
            # Convert to numeric, coercing errors
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        # Drop rows with NaN values that resulted from coercion
        input_df.dropna(inplace=True)

        if input_df.empty:
            raise HTTPException(status_code=400, detail="Invalid input data after cleaning.")

        # Make prediction
        prediction = model.predict(input_df[features])
        return {"risk_level_prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))