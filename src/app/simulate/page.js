"use client";
import React, { useState, useEffect } from 'react';

export default function SimulatePage() {
  const [numStudents, setNumStudents] = useState(1);
  const [simulatedStudents, setSimulatedStudents] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState("");
  const [selectedStudents, setSelectedStudents] = useState([]);

  useEffect(() => {
    fetchSimulatedStudents();
  }, []);

  const fetchSimulatedStudents = async () => {
    try {
      const response = await fetch('/api/simulated-students'); // Assuming a new API route for fetching
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      console.log("Fetched simulated students data:", JSON.stringify(data, null, 2));
      setSimulatedStudents(data);
    } catch (error) {
      console.error("Failed to fetch simulated students:", error);
    }
  };

  const handleSimulate = async () => {
    setLoading(true);
    setLoadingMessage("Simulating students...");
    try {
      const response = await fetch('/api/simulate-students', { // Assuming a new API route for simulation
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ num_students: numStudents }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      setLoadingMessage("Predicting learning styles...");
      const data = await response.json();
      console.log("Simulated students response:", data);
      await fetchSimulatedStudents(); // Refresh the table
    } catch (error) {
      console.error("Error simulating students:", error);
    } finally {
      setLoading(false);
      setLoadingMessage("");
    }
  };

  const handleCheckboxChange = (studentId) => {
    setSelectedStudents((prevSelected) =>
      prevSelected.includes(studentId)
        ? prevSelected.filter((id) => id !== studentId)
        : [...prevSelected, studentId]
    );
  };

  const handleDeleteSelected = async () => {
    if (selectedStudents.length === 0) return;

    setLoading(true);
    setLoadingMessage("Deleting selected students...");
    try {
      const response = await fetch('/api/delete-simulated-students', { // Assuming a new API route for deletion
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ student_ids: selectedStudents }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      setSelectedStudents([]); // Clear selection
      await fetchSimulatedStudents(); // Refresh the table
    } catch (error) {
      console.error("Error deleting selected students:", error);
    } finally {
      setLoading(false);
      setLoadingMessage("");
    }
  };

  const handleDeleteAll = async () => {
    setLoading(true);
    setLoadingMessage("Deleting all simulated students...");
    try {
      const response = await fetch('/api/delete-all-simulated-students', { // Assuming a new API route for delete all
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      setSelectedStudents([]); // Clear selection
      setSimulatedStudents([]); // Clear table
    } catch (error) {
      console.error("Error deleting all students:", error);
    } finally {
      setLoading(false);
      setLoadingMessage("");
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-gray-100 dark:bg-gray-900 text-gray-900 dark:text-gray-100">
      <main className="flex-1 p-8">
        <h1 className="text-3xl font-bold mb-6">Simulate Student Data</h1>

        {/* Simulation Panel */}
        <div className="bg-white dark:bg-gray-800 shadow-md rounded-lg p-6 mb-8">
          <h2 className="text-2xl font-semibold mb-4">Simulation Panel</h2>
          <div className="flex items-center space-x-4">
            <label htmlFor="numStudents" className="text-lg">Number of Students:</label>
            <input
              type="number"
              id="numStudents"
              value={numStudents}
              onChange={(e) => setNumStudents(Math.max(1, parseInt(e.target.value) || 1))}
              min="1"
              className="w-24 p-2 border border-gray-300 dark:border-gray-600 rounded-md bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            />
            <button
              onClick={handleSimulate}
              disabled={loading}
              className="px-6 py-2 bg-blue-600 text-white font-semibold rounded-md hover:bg-blue-700 disabled:opacity-50"
            >
              {loading ? loadingMessage : "Simulate"}
            </button>
          </div>
          {loading && <p className="mt-4 text-blue-500">{loadingMessage}</p>}
        </div>

        {/* Simulated Students Table */}
        <div className="bg-white dark:bg-gray-800 shadow-md rounded-lg p-6">
          <h2 className="text-2xl font-semibold mb-4">Simulated Students</h2>
          <div className="mb-4 flex space-x-2">
            <button
              onClick={handleDeleteSelected}
              disabled={selectedStudents.length === 0 || loading}
              className="px-4 py-2 bg-red-600 text-white font-semibold rounded-md hover:bg-red-700 disabled:opacity-50"
            >
              Delete Selected ({selectedStudents.length})
            </button>
            <button
              onClick={handleDeleteAll}
              disabled={simulatedStudents.length === 0 || loading}
              className="px-4 py-2 bg-red-600 text-white font-semibold rounded-md hover:bg-red-700 disabled:opacity-50"
            >
              Delete All
            </button>
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-700">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    <input
                      type="checkbox"
                      onChange={(e) =>
                        setSelectedStudents(
                          e.target.checked
                            ? simulatedStudents.map((s) => s._id) // Use s._id as key
                               : []
                           )
                         }
                         checked={selectedStudents.length === simulatedStudents.length && simulatedStudents.length > 0}
                      disabled={simulatedStudents.length === 0}
                    />
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Student ID
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Age
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Gender
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Academic Program
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Year Level
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    GPA
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Active vs Reflective
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Sensing vs Intuitive
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Visual vs Verbal
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Sequential vs Global
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                {simulatedStudents.length === 0 ? (
                  <tr>
                    <td colSpan="11" className="px-6 py-4 whitespace-nowrap text-center text-gray-500 dark:text-gray-400">
                      No simulated students yet. Click &apos;Simulate&apos; to generate some.
                    </td>
                  </tr>
                ) : (
                  simulatedStudents.map((student) => (
                    <tr key={student._id}>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <input
                          type="checkbox"
                          checked={selectedStudents.includes(student._id)}
                          onChange={() => handleCheckboxChange(student._id)}
                        />
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-gray-100">
                        {student.student_id}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                        {student.age}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                        {student.gender}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                        {student.academic_program}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                        {student.year_level}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                        {student.gpa}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                        {student.active_vs_reflective}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                        {student.sensing_vs_intuitive}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                        {student.visual_vs_verbal}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                        {student.sequential_vs_global}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  );
}