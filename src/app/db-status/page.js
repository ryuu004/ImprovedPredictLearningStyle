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
    <div className="min-h-screen bg-deep-space-navy text-white p-8">
      <h1 className="text-h1 font-bold mb-6 text-electric-purple">Database Status</h1>
      <p className="text-body text-gray-300 mb-4">{dbStatus}</p>
      <h1 className="text-h1 font-bold mb-6 text-electric-purple">Collection Status</h1>
      <p className="text-body text-gray-300 mb-4">{collectionStatus}</p>
    </div>
  );
}