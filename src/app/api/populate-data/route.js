import { insertTrainingData } from '../../../lib/db';

export async function POST(request) {
  try {
    const generateRandomInt = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;
    const generateRandomFloat = (min, max) => (Math.random() * (max - min) + min).toFixed(1);

    const learningStyles = {
      active_vs_reflective: ['active', 'reflective'],
      sensing_vs_intuitive: ['sensing', 'intuitive'],
      visual_vs_verbal: ['visual', 'verbal'],
      sequential_vs_global: ['sequential', 'global'],
    };

    const noteTakingStyles = ['typed', 'handwritten', 'none'];
    const problemSolvingPreferences = ['step-by-step', 'holistic solution approach'];
    const responseSpeed = ['fast', 'slow'];
    const academicPrograms = ['IT', 'Engineering', 'Business', 'Arts', 'Science'];
    const yearLevels = ['1st year', '2nd year', '3rd year', '4th year'];
    const genders = ['Male', 'Female', 'Other'];

    const generateStudentData = (id) => {
      const student_id = `S${String(id).padStart(5, '0')}`; // Ensure uniqueness with a larger pad
      const age = generateRandomInt(18, 25);
      const gender = genders[generateRandomInt(0, genders.length - 1)];
      const academic_program = academicPrograms[generateRandomInt(0, academicPrograms.length - 1)];
      const year_level = yearLevels[generateRandomInt(0, yearLevels.length - 1)];
      const GPA = parseFloat(generateRandomFloat(2.5, 4.0));
      const time_spent_on_videos = generateRandomInt(1, 20);
      const time_spent_on_text_materials = generateRandomInt(1, 20);
      const time_spent_on_interactive_activities = generateRandomInt(1, 20);
      const forum_participation_count = generateRandomInt(0, 30);
      const group_activity_participation = generateRandomInt(0, 15);
      const individual_activity_preference = generateRandomInt(1, 10);
      const note_taking_style = noteTakingStyles[generateRandomInt(0, noteTakingStyles.length - 1)];
      const preference_for_visual_materials = Math.random() > 0.5;
      const preference_for_textual_materials = Math.random() > 0.5;
      const quiz_attempts = generateRandomInt(1, 10);
      const time_to_complete_assignments = generateRandomInt(1, 7);
      const learning_path_navigation = Math.random() > 0.5 ? 'linear' : 'jumping around modules';
      const problem_solving_preference = problemSolvingPreferences[generateRandomInt(0, problemSolvingPreferences.length - 1)];
      const response_speed_in_quizzes = responseSpeed[generateRandomInt(0, responseSpeed.length - 1)];
      const accuracy_in_detail_oriented_questions = generateRandomInt(60, 100);
      const accuracy_in_conceptual_questions = generateRandomInt(60, 100);
      const preference_for_examples = Math.random() > 0.5;
      const self_reflection_activity = Math.random() > 0.5;
      const clickstream_sequence = [`resource${generateRandomInt(1, 5)}`, `quiz${generateRandomInt(1, 5)}`, `forum${generateRandomInt(1, 5)}`];
      const video_pause_and_replay_count = generateRandomInt(0, 30);
      const quiz_review_frequency = generateRandomInt(0, 5);
      const skipped_content_ratio = parseFloat(generateRandomFloat(0, 0.5));
      const login_frequency_per_week = generateRandomInt(1, 7);
      const average_study_session_length = generateRandomInt(30, 120);
      const active_vs_reflective = learningStyles.active_vs_reflective[generateRandomInt(0, 1)];
      const sensing_vs_intuitive = learningStyles.sensing_vs_intuitive[generateRandomInt(0, 1)];
      const visual_vs_verbal = learningStyles.visual_vs_verbal[generateRandomInt(0, 1)];
      const sequential_vs_global = learningStyles.sequential_vs_global[generateRandomInt(0, 1)];

      return {
        student_id, age, gender, academic_program, year_level, GPA,
        time_spent_on_videos, time_spent_on_text_materials, time_spent_on_interactive_activities,
        forum_participation_count, group_activity_participation, individual_activity_preference,
        note_taking_style, preference_for_visual_materials, preference_for_textual_materials,
        quiz_attempts, time_to_complete_assignments, learning_path_navigation,
        problem_solving_preference, response_speed_in_quizzes,
        accuracy_in_detail_oriented_questions, accuracy_in_conceptual_questions,
        preference_for_examples, self_reflection_activity, clickstream_sequence,
        video_pause_and_replay_count, quiz_review_frequency, skipped_content_ratio,
        login_frequency_per_week, average_study_session_length,
        active_vs_reflective, sensing_vs_intuitive, visual_vs_verbal, sequential_vs_global,
      };
    };

    const sampleData = [];
    for (let i = 1; i <= 40; i++) { // Generate 40 unique students
      sampleData.push(generateStudentData(i));
    }

    const result = await insertTrainingData(sampleData);
    return new Response(JSON.stringify({ message: "Data populated successfully", insertedCount: result.insertedCount }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (error) {
    console.error("Error populating data:", error);
    return new Response(JSON.stringify({ message: "Failed to populate data", error: error.message }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}