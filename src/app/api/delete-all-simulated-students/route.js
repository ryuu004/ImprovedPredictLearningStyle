import { NextResponse } from 'next/server';

export async function DELETE(request) {
  try {
    const backendResponse = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/delete-all-simulated-students`, {
      method: 'DELETE',
    });

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json();
      throw new Error(errorData.detail || `Backend error! status: ${backendResponse.status}`);
    }

    const data = await backendResponse.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Failed to delete all students:", error);
    return NextResponse.json({ message: "Failed to delete all students", error: error.message }, { status: 500 });
  }
}