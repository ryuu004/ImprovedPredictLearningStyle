import { NextResponse } from 'next/server';

export async function POST(request) {
  try {
    const { num_students, days_old } = await request.json();
    const backendResponse = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/simulate-students`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ num_students, days_old }),
    });

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json();
      throw new Error(errorData.detail || `Backend error! status: ${backendResponse.status}`);
    }

    const data = await backendResponse.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Failed to simulate students:", error);
    return NextResponse.json({ message: "Failed to simulate students", error: error.message }, { status: 500 });
  }
}