import { NextResponse } from 'next/server';

export async function POST(request) {
  try {
    const { student_ids } = await request.json();
    const backendResponse = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/update-days-old`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ student_ids, days_to_add: 1 }),
    });

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json();
      throw new Error(errorData.detail || `Backend error! status: ${backendResponse.status}`);
    }

    const data = await backendResponse.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Failed to update days old:", error);
    return NextResponse.json({ message: "Failed to update days old", error: error.message }, { status: 500 });
  }
}