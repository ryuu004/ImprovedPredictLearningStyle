import { getTrainingDataCollection } from '../../../lib/db';

export async function GET() {
  try {
    const collection = await getTrainingDataCollection();
    const students = await collection.find({}, {
      projection: {
        student_id: 1,
        age: 1,
        gender: 1,
        academic_program: 1,
        year_level: 1,
        GPA: 1,
        active_vs_reflective: 1,
        sensing_vs_intuitive: 1,
        visual_vs_verbal: 1,
        sequential_vs_global: 1,
        _id: 0, // Exclude _id from the results
      }
    }).toArray();

    // Calculate total students
    const totalStudents = students.length;

    // Calculate total students per learning style
    const studentsPerLearningStyle = {
      active_vs_reflective: { active: 0, reflective: 0 },
      sensing_vs_intuitive: { sensing: 0, intuitive: 0 },
      visual_vs_verbal: { visual: 0, verbal: 0 },
      sequential_vs_global: { sequential: 0, global: 0 },
    };

    students.forEach(student => {
      if (student.active_vs_reflective) {
        studentsPerLearningStyle.active_vs_reflective[student.active_vs_reflective]++;
      }
      if (student.sensing_vs_intuitive) {
        studentsPerLearningStyle.sensing_vs_intuitive[student.sensing_vs_intuitive]++;
      }
      if (student.visual_vs_verbal) {
        studentsPerLearningStyle.visual_vs_verbal[student.visual_vs_verbal]++;
      }
      if (student.sequential_vs_global) {
        studentsPerLearningStyle.sequential_vs_global[student.sequential_vs_global]++;
      }
    });

    return new Response(JSON.stringify({ students, totalStudents, studentsPerLearningStyle }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (error) {
    console.error("Error fetching students:", error);
    return new Response(JSON.stringify({ message: "Failed to fetch students", error: error.message }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}