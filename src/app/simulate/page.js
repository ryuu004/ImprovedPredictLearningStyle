"use client";
import React, { useState, useEffect } from 'react';
import DashboardLayout from '../dashboard/layout';

export default function SimulatePage() {
  const [numStudents, setNumStudents] = useState(1);
  const [daysOld, setDaysOld] = useState(0); // New state for days old
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
        body: JSON.stringify({ num_students: numStudents, days_old: daysOld }), // Include days_old
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

  const handleNextDay = async () => {
    if (selectedStudents.length === 0) return;

    setLoading(true);
    setLoadingMessage("Advancing selected students to next day...");
    try {
      const response = await fetch('/api/update-days-old', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ student_ids: selectedStudents }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Backend error! status: ${response.status}`);
      }
      setSelectedStudents([]); // Clear selection
      await fetchSimulatedStudents(); // Refresh the table
    } catch (error) {
      console.error("Error advancing students to next day:", error);
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
    <DashboardLayout>
      <div className="bg-dark-navy text-white p-2 min-h-screen overflow-y-auto">
        <h1 className="text-3xl font-bold mb-6 text-accent-blue">Simulate Student Data</h1>

        {/* Simulation Panel */}
        <div className="bg-charcoal rounded-xl p-3 shadow-lg mb-3 border border-transparent hover:border-accent-blue transition-all duration-300 ease-in-out">
          <h2 className="text-xl font-bold mb-3 text-accent-blue">Simulation Panel</h2>
          <div className="flex items-center space-x-4 mb-4">
            <label htmlFor="numStudents" className="text-lg text-gray-300">Number of Students:</label>
            <input
              type="number"
              id="numStudents"
              value={numStudents}
              onChange={(e) => setNumStudents(Math.max(1, parseInt(e.target.value) || 1))}
              min="1"
              className="w-24 p-2 border border-subtle-gray-light rounded-md bg-subtle-gray-dark text-accent-cyan"
            />
          </div>
          <div className="flex items-center space-x-4">
            <label htmlFor="daysOld" className="text-lg text-gray-300">Days Old for LSTM:</label>
            <input
              type="number"
              id="daysOld"
              value={daysOld}
              onChange={(e) => setDaysOld(Math.max(0, parseInt(e.target.value) || 0))}
              min="0"
              className="w-24 p-2 border border-subtle-gray-light rounded-md bg-subtle-gray-dark text-accent-cyan"
            />
            <button
              onClick={handleSimulate}
              disabled={loading}
              className="px-6 py-2 bg-accent-blue text-white font-semibold rounded-md hover:bg-blue-700 disabled:opacity-50 transition-colors duration-300"
            >
              {loading ? loadingMessage : "Simulate"}
            </button>
          </div>
          {loading && <p className="mt-4 text-accent-cyan">{loadingMessage}</p>}
        </div>

        {/* Next Day Update Panel */}
        <div className="bg-charcoal rounded-xl p-3 shadow-lg mb-3 border border-transparent hover:border-accent-blue transition-all duration-300 ease-in-out">
          <h2 className="text-xl font-bold mb-3 text-accent-blue">Update Simulated Students (Next Day)</h2>
          <button
            onClick={handleNextDay}
            disabled={selectedStudents.length === 0 || loading}
            className="px-6 py-2 bg-green-600 text-white font-semibold rounded-md hover:bg-green-700 disabled:opacity-50 transition-colors duration-300"
          >
            Advance Selected Students to Next Day ({selectedStudents.length})
          </button>
        </div>

        {/* Simulated Students Table */}
        <div className="bg-charcoal rounded-xl p-3 shadow-lg mb-3 border border-transparent hover:border-accent-blue transition-all duration-300 ease-in-out">
          <h2 className="text-xl font-bold mb-3 text-accent-blue">Simulated Students</h2>
          <div className="mb-4 flex space-x-2">
            <button
              onClick={handleDeleteSelected}
              disabled={selectedStudents.length === 0 || loading}
              className="px-4 py-2 bg-red-600 text-white font-semibold rounded-md hover:bg-red-700 disabled:opacity-50 transition-colors duration-300"
            >
              Delete Selected ({selectedStudents.length})
            </button>
            <button
              onClick={handleDeleteAll}
              disabled={simulatedStudents.length === 0 || loading}
              className="px-4 py-2 bg-red-600 text-white font-semibold rounded-md hover:bg-red-700 disabled:opacity-50 transition-colors duration-300"
            >
              Delete All
            </button>
          </div>
          <div className="overflow-x-auto overflow-y-auto relative rounded-lg border border-subtle-gray-dark" style={{ maxHeight: '400px' }}>
            <table className="min-w-full text-left table-auto table-compact">
              <thead className="bg-charcoal sticky top-0 border-b border-subtle-gray-light">
                <tr>
                  <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">
                    <input
                      type="checkbox"
                      onChange={(e) =>
                        setSelectedStudents(
                          e.target.checked
                            ? simulatedStudents.map((s) => s.student_id) // Use student_id as key
                               : []
                            )
                         }
                         checked={selectedStudents.length === simulatedStudents.length && simulatedStudents.length > 0}
                      disabled={simulatedStudents.length === 0}
                    />
                  </th>
                  <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">
                    Student ID
                  </th>
                  <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">
                    Days Old
                  </th>
                  <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">
                    Age
                  </th>
                  <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">
                    Gender
                  </th>
                  <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">
                    Academic Program
                  </th>
                  <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">
                    Year Level
                  </th>
                  <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">
                    GPA
                  </th>
                  <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">
                    Active vs Reflective
                  </th>
                  <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">
                    Sensing vs Intuitive
                  </th>
                  <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">
                    Visual vs Verbal
                  </th>
                  <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">
                    Sequential vs Global
                  </th>
                </tr>
              </thead>
              <tbody className="text-gray-300 text-sm font-light">
                {simulatedStudents.length === 0 ? (
                  <tr>
                    <td colSpan="11" className="px-2 py-1 whitespace-nowrap text-center text-gray-500">
                      No simulated students yet. Click 'Simulate' to generate some.
                    </td>
                  </tr>
                ) : (
                  simulatedStudents.map((student) => (
                    <tr key={student.student_id}>
                      <td className="px-2 py-1 whitespace-nowrap">
                        <input
                          type="checkbox"
                          checked={selectedStudents.includes(student.student_id)}
                          onChange={() => handleCheckboxChange(student.student_id)}
                        />
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-gray-100">
                        {student.student_id}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                        {student.days_old}
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
        </div>
    </DashboardLayout>
  );
}