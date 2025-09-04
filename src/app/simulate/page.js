"use client";

import { useState, useEffect } from 'react';

export default function SimulatePage() {
  const [numStudents, setNumStudents] = useState(1);
  const [daysOld, setDaysOld] = useState(0); // New state for days old
  const [simulatedStudents, setSimulatedStudents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [loadingMessage, setLoadingMessage] = useState("");
  const [selectedStudents, setSelectedStudents] = useState([]);
  const [error, setError] = useState(null);

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
      setSimulatedStudents(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
      setLoadingMessage("");
    }
  };

  const handleSimulate = async () => {
    setLoading(true);
    setLoadingMessage("Simulating students...");
    setError(null);
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
      await fetchSimulatedStudents(); // Refresh the table
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
      setLoadingMessage("");
    }
  };

  const handleNextDay = async () => {
    if (selectedStudents.length === 0) return;

    setLoading(true);
    setLoadingMessage("Advancing selected students to next day...");
    setError(null);
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
    } catch (e) {
      setError(e.message);
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
    setError(null);
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
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
      setLoadingMessage("");
    }
  };

  const handleDeleteAll = async () => {
    setLoading(true);
    setLoadingMessage("Deleting all simulated students...");
    setError(null);
    try {
      const response = await fetch('/api/delete-all-simulated-students', { // Assuming a new API route for delete all
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      setSelectedStudents([]); // Clear selection
      setSimulatedStudents([]); // Clear table
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
      setLoadingMessage("");
    }
  };

  if (loading) return (
    <div className="min-h-screen bg-deep-space-navy text-white flex items-center justify-center">
      <p>Loading simulation data...</p>
    </div>
  );

  if (error) return (
    <div className="min-h-screen bg-deep-space-navy text-white flex items-center justify-center">
      <p className="text-rose-500">Error: {error}</p>
    </div>
  );

  return (
    <div className="bg-deep-space-navy text-white p-2 h-screen flex flex-col overflow-hidden">
      <h1 className="text-3xl font-bold mb-6 text-electric-purple">Simulate Student Data</h1>

      {/* Simulation Panel */}
      <div className="glass-morphism p-3 rounded-card shadow-elevation-1 mb-3 flex-shrink-0 border border-transparent hover:border-electric-purple hover:shadow-elevation-3 transition-all duration-300 ease-in-out">
        <h2 className="text-xl font-bold mb-3 text-electric-purple">Simulation Panel</h2>
        <div className="flex items-center space-x-4 mb-4">
          <label htmlFor="numStudents" className="text-lg text-gray-300">Number of Students:</label>
          <input
            type="number"
            id="numStudents"
            value={numStudents}
            onChange={(e) => setNumStudents(Math.max(1, parseInt(e.target.value) || 1))}
            min="1"
            className="w-24 p-2 rounded-form bg-charcoal-elevated border border-transparent text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-electric-purple focus:border-electric-purple transition-all duration-300 ease-in-out text-sm shadow-inner"
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
            className="w-24 p-2 rounded-form bg-charcoal-elevated border border-transparent text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-electric-purple focus:border-electric-purple transition-all duration-300 ease-in-out text-sm shadow-inner"
          />
          <button
            onClick={handleSimulate}
            disabled={loading}
            className="px-6 py-2 rounded-button bg-electric-purple text-white font-semibold hover:bg-electric-purple-dark disabled:opacity-50 transition-colors duration-300 shadow-elevation-1"
          >
            {loading ? loadingMessage : "Simulate"}
          </button>
        </div>
        {loading && <p className="mt-4 text-electric-purple">{loadingMessage}</p>}
      </div>

      {/* Next Day Update Panel */}
      <div className="glass-morphism p-3 rounded-card shadow-elevation-1 mb-3 flex-shrink-0 border border-transparent hover:border-electric-purple hover:shadow-elevation-3 transition-all duration-300 ease-in-out">
        <h2 className="text-xl font-bold mb-3 text-electric-purple">Update Simulated Students (Next Day)</h2>
        <button
          onClick={handleNextDay}
          disabled={selectedStudents.length === 0 || loading}
          className="px-6 py-2 rounded-button bg-emerald-success text-white font-semibold hover:bg-emerald-success-dark disabled:opacity-50 transition-colors duration-300 shadow-elevation-1"
        >
          Advance Selected Students to Next Day ({selectedStudents.length})
        </button>
      </div>

      {/* Simulated Students Table */}
      <div className="glass-morphism p-3 rounded-card shadow-elevation-1 mb-3 flex-shrink-0 border border-transparent hover:border-electric-purple hover:shadow-elevation-3 transition-all duration-300 ease-in-out">
        <h2 className="text-xl font-bold mb-3 text-electric-purple">Simulated Students</h2>
        <div className="mb-4 flex space-x-2">
          <button
            onClick={handleDeleteSelected}
            disabled={selectedStudents.length === 0 || loading}
            className="px-4 py-2 rounded-button bg-rose-danger text-white font-semibold hover:bg-rose-danger-dark disabled:opacity-50 transition-colors duration-300 shadow-elevation-1"
          >
            Delete Selected ({selectedStudents.length})
          </button>
          <button
            onClick={handleDeleteAll}
            disabled={simulatedStudents.length === 0 || loading}
            className="px-4 py-2 rounded-button bg-rose-danger text-white font-semibold hover:bg-rose-danger-dark disabled:opacity-50 transition-colors duration-300 shadow-elevation-1"
          >
            Delete All
          </button>
        </div>
        <div className="overflow-x-auto overflow-y-auto relative rounded-card border border-transparent" style={{ maxHeight: '400px' }}>
          <table className="w-full text-left table-auto table-compact">
            <thead className="bg-charcoal-elevated sticky top-0 border-b border-transparent">
              <tr>
                <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider rounded-tl-card">
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
                    className="form-checkbox h-4 w-4 text-electric-purple transition duration-150 ease-in-out"
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
                <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider rounded-tr-card">
                  Sequential vs Global
                </th>
              </tr>
            </thead>
            <tbody className="text-gray-300 text-sm font-light">
              {simulatedStudents.length === 0 ? (
                <tr>
                  <td colSpan="12" className="px-2 py-1 whitespace-nowrap text-center text-gray-500">
                    No simulated students yet. Click 'Simulate' to generate some.
                  </td>
                </tr>
              ) : (
                simulatedStudents.map((student, index) => (
                  <tr key={student.student_id} className={`border-b border-transparent ${index % 2 === 0 ? 'bg-charcoal-elevated' : 'bg-charcoal-elevated'} hover:bg-charcoal-elevated transition-all duration-200 ease-in-out`}>
                    <td className="px-2 py-1 whitespace-nowrap">
                      <input
                        type="checkbox"
                        checked={selectedStudents.includes(student.student_id)}
                        onChange={() => handleCheckboxChange(student.student_id)}
                        className="form-checkbox h-4 w-4 text-electric-purple transition duration-150 ease-in-out"
                      />
                    </td>
                    <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-300">
                      {student.student_id}
                    </td>
                    <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-300">
                      {student.days_old}
                    </td>
                    <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-300">
                      {student.age}
                    </td>
                    <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-300">
                      {student.gender}
                    </td>
                    <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-300">
                      {student.academic_program}
                    </td>
                    <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-300">
                      {student.year_level}
                    </td>
                    <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-300">
                      {student.gpa}
                    </td>
                    <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-300 capitalize">
                      {student.active_vs_reflective}
                    </td>
                    <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-300 capitalize">
                      {student.sensing_vs_intuitive}
                    </td>
                    <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-300 capitalize">
                      {student.visual_vs_verbal}
                    </td>
                    <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-300 capitalize">
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
  );
}