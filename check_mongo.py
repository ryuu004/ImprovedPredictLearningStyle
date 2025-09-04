from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv(".env.local")

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client["ml_database"]
simulated_students_collection = db["simulated_students"]

print("Attempting to count documents in 'simulated_students' collection...")
try:
    count = simulated_students_collection.count_documents({})
    print(f"Number of documents in 'simulated_students' collection: {count}")
    
    if count > 0:
        print("First 5 documents:")
        for i, doc in enumerate(simulated_students_collection.find().limit(5)):
            print(f"Document {i+1}: {doc}")
    
except Exception as e:
    print(f"Error accessing MongoDB: {e}")

finally:
    client.close()