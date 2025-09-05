"use client";

import { useState, useEffect, useRef } from 'react';

export default function SimulatePage() {
  const [numStudents, setNumStudents] = useState(1);
  const [daysOld, setDaysOld] = useState(0); // New state for days old
  const [simulatedStudents, setSimulatedStudents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [loadingMessage, setLoadingMessage] = useState("");
  const [selectedStudents, setSelectedStudents] = useState([]);
  const [error, setError] = useState(null);
  const previousSimulatedStudents = useRef([]);

  useEffect(() => {
    fetchSimulatedStudents();
  }, []);

  useEffect(() => {
    // Store the current students as previous ones for the next render
    previousSimulatedStudents.current = simulatedStudents;
  }, [simulatedStudents]);

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
    setLoadingMessage("");
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



  if (error) return (
    <div className="min-h-screen bg-deep-space-navy text-white flex items-center justify-center">
      <p className="text-rose-500">Error: {error}</p>
    </div>
  );

  return (
    <>
      <div className="bg-deep-space-navy text-white p-2 flex flex-col overflow-y-auto min-h-screen">
        {/* Main content area */}
        <div className="flex-grow flex flex-col overflow-y-auto">

          {/* Simulation Panel */}
          <div className="glass-morphism p-3 rounded-card shadow-xl mb-3 flex-shrink-0 border border-transparent hover:border-electric-purple hover:shadow-2xl transition-all duration-300 ease-in-out backdrop-filter backdrop-blur-md bg-opacity-10 bg-charcoal-elevated relative overflow-hidden group transition-all duration-300 ease-out glow-on-hover">
            <h2 className="text-xl font-bold mb-3 text-electric-purple">Simulation Panel</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              <div className="flex flex-col">
                <label htmlFor="numStudents" className="text-lg text-gray-300 mb-2">Number of Students:</label>
                <input
                  type="number"
                  id="numStudents"
                  value={numStudents}
                  onChange={(e) => setNumStudents(Math.max(1, parseInt(e.target.value) || 1))}
                  min="1"
                  className="w-full px-3 py-1.5 rounded-form bg-charcoal-elevated border border-transparent text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-electric-purple focus:border-electric-purple transition-all duration-300 ease-in-out text-sm shadow-inner"
                />
              </div>
              <div className="flex flex-col">
                <label htmlFor="daysOld" className="text-lg text-gray-300 mb-2">Days Old for LSTM:</label>
                <input
                  type="number"
                  id="daysOld"
                  value={daysOld}
                  onChange={(e) => setDaysOld(Math.max(0, parseInt(e.target.value) || 0))}
                  min="0"
                  className="w-full px-3 py-1.5 rounded-form bg-charcoal-elevated border border-transparent text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-electric-purple focus:border-electric-purple transition-all duration-300 ease-in-out text-sm shadow-inner"
                />
              </div>
            </div>
            <button
              onClick={handleSimulate}
              disabled={loading}
              className="w-full py-3 rounded-button bg-gradient-to-r from-button-purple-start to-button-purple-end text-white font-semibold text-lg hover:from-button-purple-end hover:to-button-purple-start hover:shadow-lg transition-all duration-300 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed transform hover:-translate-y-0.5 active:scale-95"
            >
              {loading ? loadingMessage : "Simulate"}
            </button>
            {loading && <p className="mt-4 text-electric-purple text-center">{loadingMessage}</p>}
          </div>

          {/* Next Day Update Panel */}
          <div className="glass-morphism p-3 rounded-card shadow-xl mb-3 flex-shrink-0 border border-transparent hover:border-emerald-success hover:shadow-2xl transition-all duration-300 ease-in-out backdrop-filter backdrop-blur-md bg-opacity-10 bg-charcoal-elevated relative overflow-hidden group transition-all duration-300 ease-out glow-on-hover">
            <h2 className="text-xl font-bold mb-3 text-emerald-success">Update Simulated Students</h2>
            <button
              onClick={handleNextDay}
              disabled={selectedStudents.length === 0 || loading}
              className="w-full py-3 rounded-button bg-emerald-success text-white font-semibold text-lg hover:bg-emerald-success-dark disabled:opacity-50 transition-colors duration-300 shadow-elevation-1 transform hover:-translate-y-0.5 active:scale-95"
            >
              {loading ? (
                <svg className="animate-spin h-5 w-5 text-white mx-auto" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              ) : (
                `Advance Selected Students to Next Day (${selectedStudents.length})`
              )}
            </button>
          </div>

          {/* Simulated Students Table */}
          <div className="glass-morphism p-3 rounded-card shadow-xl flex-grow flex flex-col border border-transparent hover:border-electric-purple hover:shadow-2xl transition-all duration-300 ease-in-out backdrop-filter backdrop-blur-md bg-opacity-10 bg-charcoal-elevated relative overflow-hidden group transition-all duration-300 ease-out glow-on-hover">
            <h2 className="text-xl font-bold mb-3 text-electric-purple">Simulated Students</h2>
            <div className="mb-4 flex flex-wrap gap-3 justify-center">
              <button
                onClick={handleDeleteSelected}
                disabled={selectedStudents.length === 0 || loading}
                className="px-6 py-3 rounded-button bg-rose-danger text-white font-semibold text-base hover:bg-rose-danger-dark disabled:opacity-50 transition-colors duration-300 shadow-elevation-1 transform hover:-translate-y-0.5 active:scale-95"
              >
                Delete Selected ({selectedStudents.length})
              </button>
              <button
                onClick={handleDeleteAll}
                disabled={simulatedStudents.length === 0 || loading}
                className="px-6 py-3 rounded-button bg-rose-danger text-white font-semibold text-base hover:bg-rose-danger-dark disabled:opacity-50 transition-colors duration-300 shadow-elevation-1 transform hover:-translate-y-0.5 active:scale-95"
              >
                Delete All
              </button>
            </div>
            <div className="overflow-x-auto overflow-y-auto relative rounded-card border border-transparent flex-grow">
              <table className="w-full text-left table-auto table-compact">
                <thead className="bg-charcoal-elevated sticky top-0 border-b border-transparent">
                  <tr>
                    <th className="px-4 py-2 text-left text-xs font-bold text-gray-200 uppercase tracking-wider rounded-tl-card">
                      <input
                        type="checkbox"
                        onChange={(e) =>
                          setSelectedStudents(
                            e.target.checked
                              ? simulatedStudents.map((s) => s.student_id)
                              : []
                          )
                        }
                        checked={selectedStudents.length === simulatedStudents.length && simulatedStudents.length > 0}
                        disabled={simulatedStudents.length === 0}
                        className="form-checkbox"
                      />
                    </th>
                    <th className="px-4 py-2 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">
                      Student ID
                    </th>
                    <th className="px-4 py-2 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">
                      Days Old
                    </th>
                    <th className="px-4 py-2 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">
                      Age
                    </th>
                    <th className="px-4 py-2 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">
                      Gender
                    </th>
                    <th className="px-4 py-2 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">
                      Academic Program
                    </th>
                    <th className="px-4 py-2 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">
                      Year Level
                    </th>
                    <th className="px-4 py-2 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">
                      GPA
                    </th>
                    <th className="px-4 py-2 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">
                      Active vs Reflective
                    </th>
                    <th className="px-4 py-2 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">
                      Sensing vs Intuitive
                    </th>
                    <th className="px-4 py-2 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">
                      Visual vs Verbal
                    </th>
                    <th className="px-4 py-2 text-left text-xs font-bold text-gray-200 uppercase tracking-wider rounded-tr-card">
                      Sequential vs Global
                    </th>
                  </tr>
                </thead>
                <tbody className="text-gray-300 text-sm font-light">
                  {simulatedStudents.length === 0 ? (
                    <tr>
                      <td colSpan="12" className="px-4 py-3 whitespace-nowrap text-center text-gray-500">
                        No simulated students yet. Click {'Simulate'} to generate some.
                      </td>
                    </tr>
                  ) : (
                    simulatedStudents.map((student, index) => {
                      const prevStudent = previousSimulatedStudents.current.find(
                        (prev) => prev.student_id === student.student_id
                      );

                      const getHighlightClass = (currentValue, previousValue) => {
                        return currentValue !== previousValue ? 'highlight-flash' : '';
                      };

                      return (
                        <tr key={student.student_id} className={`border-b border-transparent ${index % 2 === 0 ? 'bg-charcoal-elevated' : 'bg-charcoal-elevated/50'} hover:bg-charcoal-elevated/75 transition-all duration-200 ease-in-out`}>
                          <td className="px-4 py-3 whitespace-nowrap">
                            <input
                              type="checkbox"
                              checked={selectedStudents.includes(student.student_id)}
                              onChange={() => handleCheckboxChange(student.student_id)}
                              className="form-checkbox"
                            />
                          </td>
                          <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">
                            {student.student_id}
                          </td>
                          <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">
                            {student.days_old}
                          </td>
                          <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">
                            {student.age}
                          </td>
                          <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">
                            {student.gender}
                          </td>
                          <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">
                            {student.academic_program}
                          </td>
                          <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">
                            {student.year_level}
                          </td>
                          <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">
                            {student.gpa}
                          </td>
                          <td className={`px-4 py-3 whitespace-nowrap text-sm text-gray-300 capitalize ${getHighlightClass(student.active_vs_reflective, prevStudent?.active_vs_reflective)}`}>
                            {student.active_vs_reflective}
                          </td>
                          <td className={`px-4 py-3 whitespace-nowrap text-sm text-gray-300 capitalize ${getHighlightClass(student.sensing_vs_intuitive, prevStudent?.sensing_vs_intuitive)}`}>
                            {student.sensing_vs_intuitive}
                          </td>
                          <td className={`px-4 py-3 whitespace-nowrap text-sm text-gray-300 capitalize ${getHighlightClass(student.visual_vs_verbal, prevStudent?.visual_vs_verbal)}`}>
                            {student.visual_vs_verbal}
                          </td>
                          <td className={`px-4 py-3 whitespace-nowrap text-sm text-gray-300 capitalize ${getHighlightClass(student.sequential_vs_global, prevStudent?.sequential_vs_global)}`}>
                            {student.sequential_vs_global}
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
      {loading && (
        <div className="fixed inset-0 bg-dark-navy-start bg-opacity-75 flex flex-col items-center justify-center z-50">
          <svg className="animate-spin h-10 w-10 text-electric-purple mb-3" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          <p className="text-white text-lg">{loadingMessage || "Loading..."}</p>
        </div>
      )}
    </>
  );
}