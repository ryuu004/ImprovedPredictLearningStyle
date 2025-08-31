import clientPromise from '../lib/db';

export default async function DbStatusPage() {
  let isConnected = false;
  try {
    await clientPromise;
    isConnected = true;
  } catch (error) {
    console.error("Database connection failed:", error);
    isConnected = false;
  }

  return (
    <div>
      {isConnected ? (
        <h1>Database Connected</h1>
      ) : (
        <h1>Database Connection Failed</h1>
      )}
    </div>
  );
}