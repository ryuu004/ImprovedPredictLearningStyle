// lib/db.js
import { MongoClient } from "mongodb";

const uri = process.env.MONGODB_URI;
const options = {};

let client;
let clientPromise;

if (!process.env.MONGODB_URI) {
  throw new Error("Please add your Mongo URI to .env.local");
}

if (process.env.NODE_ENV === "development") {
  // Reuse client in dev mode (Hot Reload fix)
  if (!global._mongoClientPromise) {
    client = new MongoClient(uri, options);
    global._mongoClientPromise = client.connect();
  }
  clientPromise = global._mongoClientPromise;
} else {
  client = new MongoClient(uri, options);
  clientPromise = client.connect();
}

export default clientPromise;

/**
 * @typedef {Object} StudentLearningData
 * @property {string} student_id - Unique identifier for the student.
 * @property {number} age
 * @property {string} gender
 * @property {string} academic_program - e.g., IT, Engineering, Business
 * @property {string} year_level - 1st year, 2nd year, etc.
 * @property {number} GPA - or previous_academic_performance
 * @property {number} time_spent_on_videos - hours per week
 * @property {number} time_spent_on_text_materials
 * @property {number} time_spent_on_interactive_activities
 * @property {number} forum_participation_count
 * @property {number} group_activity_participation
 * @property {number} individual_activity_preference
 * @property {string} note_taking_style - typed, handwritten, none
 * @property {boolean} preference_for_visual_materials
 * @property {boolean} preference_for_textual_materials
 * @property {number} quiz_attempts
 * @property {number} time_to_complete_assignments
 * @property {string} learning_path_navigation - linear vs. jumping around modules
 * @property {string} problem_solving_preference - step-by-step vs. holistic solution approach
 * @property {string} response_speed_in_quizzes - fast vs. slow
 * @property {number} accuracy_in_detail_oriented_questions
 * @property {number} accuracy_in_conceptual_questions
 * @property {boolean} preference_for_examples - asks for examples vs. theories
 * @property {boolean} self_reflection_activity - journal entries, reflections submitted
 * @property {string[]} clickstream_sequence - order of resource access
 * @property {number} video_pause_and_replay_count
 * @property {number} quiz_review_frequency
 * @property {number} skipped_content_ratio
 * @property {number} login_frequency_per_week
 * @property {number} average_study_session_length
 * @property {string} active_vs_reflective - Target Variable
 * @property {string} sensing_vs_intuitive - Target Variable
 * @property {string} visual_vs_verbal - Target Variable
 * @property {string} sequential_vs_global - Target Variable
 */

export async function getTrainingDataCollection() {
  const client = await clientPromise;
  const db = client.db("ml_database");
  const collection = db.collection("training_data");
  return collection;
}

/**
 * Inserts an array of student learning data into the training_data collection.
 * @param {StudentLearningData[]} data - An array of student learning data objects.
 */
export async function insertTrainingData(data) {
  const collection = await getTrainingDataCollection();
  const result = await collection.insertMany(data);
  return result;
}
