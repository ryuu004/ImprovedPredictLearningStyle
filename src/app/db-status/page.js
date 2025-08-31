import clientPromise, { getTrainingDataCollection } from '../../lib/db';

export default async function DbStatusPage() {
  let dbStatus = "Checking database connection...";
  let collectionStatus = "Checking collection status...";

  try {
    await clientPromise;
    dbStatus = "Database Connected";
  } catch (error) {
    console.error("Database connection failed:", error);
    dbStatus = `Database Connection Failed: ${error.message}`;
  }

  try {
    const collection = await getTrainingDataCollection();
    if (collection) {
      collectionStatus = "Collection 'training_data' in 'ml_database' is accessible.";
    } else {
      collectionStatus = "Failed to access 'training_data' collection.";
    }
  } catch (error) {
    console.error("Collection access failed:", error);
    collectionStatus = `Failed to access 'training_data' collection: ${error.message}`;
  }

  return (
    <div>
      <h1>Database Status</h1>
      <p>{dbStatus}</p>
      <h1>Collection Status</h1>
      <p>{collectionStatus}</p>
    </div>
  );
}