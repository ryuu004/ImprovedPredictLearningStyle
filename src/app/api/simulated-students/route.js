import { getAllSimulatedStudents } from '@/lib/db';
import { NextResponse } from 'next/server';

export async function GET() {
  try {
    const students = await getAllSimulatedStudents();
    return NextResponse.json(students);
  } catch (error) {
    console.error("Failed to fetch simulated students:", error);
    return NextResponse.json({ message: "Failed to fetch simulated students", error: error.message }, { status: 500 });
  }
}