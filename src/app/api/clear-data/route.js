import { getTrainingDataCollection } from '../../../lib/db';

export async function POST(request) {
  try {
    const collection = await getTrainingDataCollection();
    const result = await collection.deleteMany({});
    return new Response(JSON.stringify({ message: "Data cleared successfully", deletedCount: result.deletedCount }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (error) {
    console.error("Error clearing data:", error);
    return new Response(JSON.stringify({ message: "Failed to clear data", error: error.message }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}