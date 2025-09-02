import { NextResponse } from 'next/server';

export async function DELETE(request) {
  try {
    const { student_ids } = await request.json();
    const backendResponse = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/delete-simulated-students`, {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ student_ids }),
    });

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json();
      throw new Error(errorData.detail || `Backend error! status: ${backendResponse.status}`);
    }

    const data = await backendResponse.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Failed to delete selected students:", error);
    return NextResponse.json({ message: "Failed to delete selected students", error: error.message }, { status: 500 });
  }
}