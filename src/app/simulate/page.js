"use client";

import { useState, useEffect, useRef } from 'react';

export default function SimulatePage() {
  const [numStudents, setNumStudents] = useState(1);
  const [daysOld, setDaysOld] = useState(0); // New state for days old
  const [selectedModel, setSelectedModel] = useState('random_forest'); // Default model
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
        body: JSON.stringify({ num_students: numStudents, days_old: daysOld, model_type: selectedModel }), // Include days_old and model_type
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
      <div className="bg-deep-space-navy text-white p-2 h-screen flex flex-col overflow-hidden">
        <div className="flex flex-col space-y-2 flex-grow overflow-hidden">
          
          {/* Simulation Panel */}
          <div className="glass-morphism p-2 rounded-card shadow-elevation-1 mb-2 flex-shrink-0 border border-transparent hover:border-electric-purple hover:shadow-elevation-3 transition-all duration-300 ease-in-out">
            <h2 className="text-lg font-bold text-electric-purple mb-3">Simulation Panel</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="flex flex-col">
                <label htmlFor="numStudents" className="text-xs text-gray-300 mb-1">Number of Students:</label>
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
                <label htmlFor="daysOld" className="text-xs text-gray-300 mb-1">Days Old for LSTM:</label>
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
            <div className="mt-4">
              <label htmlFor="modelSelect" className="text-xs text-gray-300 mb-1 block">Prediction Model:</label>
              <select
                id="modelSelect"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full px-3 py-1.5 rounded-form bg-charcoal-elevated border border-transparent text-white focus:outline-none focus:ring-2 focus:ring-electric-purple focus:border-electric-purple transition-all duration-300 ease-in-out text-sm shadow-inner"
              >
                <option value="random_forest">Random Forest</option>
                <option value="xgboost">XGBoost</option>
              </select>
            </div>
            <div className="flex justify-end mt-4 pt-2">
              <button
                onClick={handleSimulate}
                disabled={loading}
                className="px-3 py-1.5 rounded-button bg-charcoal-elevated border border-transparent text-white focus:outline-none focus:ring-2 focus:ring-electric-purple focus:border-electric-purple transition-all duration-300 ease-in-out text-sm shadow-inner flex items-center justify-center space-x-1 hover:bg-charcoal-hover"
              >
                {loading ? loadingMessage : <><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 3c2.75 0 5 2.25 5 5v.75a.75.75 0 01-.75.75H7.75a.75.75 0 01-.75-.75V8c0-2.75 2.25-5 5-5z" />
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 10.5a2.25 2.25 0 100 4.5 2.25 2.25 0 000-4.5z" />
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 16.5c-2.75 0-5-2.25-5-5v-.75a.75.75 0 01.75-.75h8.5a.75.75 0 01.75.75v.75c0 2.75-2.25 5-5 5z" />
                </svg><span>Simulate</span></>}
              </button>
            </div>
          </div>
          

          {/* Simulated Students Table */}
          <div className="glass-morphism p-2 rounded-card shadow-elevation-1 flex flex-col flex-grow floating-element">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-bold text-electric-purple">Simulated Students</h2>
              <div className="flex gap-2">
                <button
                  onClick={handleNextDay}
                  disabled={selectedStudents.length === 0 || loading}
                  className="px-3 py-1.5 rounded-button bg-charcoal-elevated border border-transparent text-white focus:outline-none focus:ring-2 focus:ring-electric-purple focus:border-electric-purple transition-all duration-300 ease-in-out text-sm shadow-inner flex items-center justify-center space-x-1 hover:bg-charcoal-hover"
                >
                  {loading ? (
                    <svg className="animate-spin h-4 w-4 text-white mx-auto" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                  ) : (
                    <><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M12 3c2.75 0 5 2.25 5 5v.75a.75.75 0 01-.75.75H7.75a.75.75 0 01-.75-.75V8c0-2.75 2.25-5 5-5z" />
                      <path strokeLinecap="round" strokeLinejoin="round" d="M12 10.5a2.25 2.25 0 100 4.5 2.25 2.25 0 000-4.5z" />
                      <path strokeLinecap="round" strokeLinejoin="round" d="M12 16.5c-2.75 0-5-2.25-5-5v-.75a.75.75 0 01.75-.75h8.5a.75.75 0 01.75.75v.75c0 2.75-2.25 5-5 5z" />
                    </svg><span>Advance Selected ({selectedStudents.length})</span></>
                  )}
                </button>
                <button
                  onClick={handleDeleteSelected}
                  disabled={selectedStudents.length === 0 || loading}
                  className="px-3 py-1.5 rounded-button bg-charcoal-elevated border border-transparent text-rose-400 focus:outline-none focus:ring-2 focus:ring-electric-purple focus:border-electric-purple transition-all duration-300 ease-in-out text-sm shadow-inner flex items-center justify-center space-x-1 hover:bg-charcoal-hover hover:text-rose-500"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.927a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m-1.022.165L5.34 19.673a2.25 2.25 0 002.244 2.077h8.114a2.25 2.25 0 002.244-2.077L19.58 5.79m-1.876 0A.75.75 0 0118 6.5v.75m-6 0v.75m6.75 0H12a.75.75 0 01-.75-.75v-.75m0 0H5.25" />
                  </svg>
                  <span>Delete Selected ({selectedStudents.length})</span>
                </button>
                <button
                  onClick={handleDeleteAll}
                  disabled={simulatedStudents.length === 0 || loading}
                  className="px-3 py-1.5 rounded-button bg-charcoal-elevated border border-transparent text-rose-400 focus:outline-none focus:ring-2 focus:ring-electric-purple focus:border-electric-purple transition-all duration-300 ease-in-out text-sm shadow-inner flex items-center justify-center space-x-1 hover:bg-charcoal-hover hover:text-rose-500"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.927a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m-1.022.165L5.34 19.673a2.25 2.25 0 002.244 2.077h8.114a2.25 2.25 0 002.244-2.077L19.58 5.79m-1.876 0A.75.75 0 0118 6.5v.75m-6 0v.75m6.75 0H12a.75.75 0 01-.75-.75v-.75m0 0H5.25" />
                  </svg>
                  <span>Delete All</span>
                </button>
              </div>
            </div>
            <div className="overflow-x-auto overflow-y-auto relative rounded-card border border-transparent max-h-[300px]">
              <table className="w-full text-left table-auto">
                <thead className="bg-charcoal-elevated sticky top-0 border-b border-transparent">
                  <tr>
                    <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider cursor-pointer hover:bg-charcoal-elevated transition-colors duration-200 rounded-tl-card">
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
                    <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider cursor-pointer hover:bg-charcoal-elevated transition-colors duration-200">Student ID</th>
                    <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider cursor-pointer hover:bg-charcoal-elevated transition-colors duration-200">Days Old</th>
                    <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider cursor-pointer hover:bg-charcoal-elevated transition-colors duration-200">Age</th>
                    <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider cursor-pointer hover:bg-charcoal-elevated transition-colors duration-200">Gender</th>
                    <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider cursor-pointer hover:bg-charcoal-elevated transition-colors duration-200">Academic Program</th>
                    <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider cursor-pointer hover:bg-charcoal-elevated transition-colors duration-200">Year Level</th>
                    <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider cursor-pointer hover:bg-charcoal-elevated transition-colors duration-200">GPA</th>
                    <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider cursor-pointer hover:bg-charcoal-elevated transition-colors duration-200">Active vs Reflective</th>
                    <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider cursor-pointer hover:bg-charcoal-elevated transition-colors duration-200">Sensing vs Intuitive</th>
                    <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider cursor-pointer hover:bg-charcoal-elevated transition-colors duration-200">Visual vs Verbal</th>
                    <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider cursor-pointer hover:bg-charcoal-elevated transition-colors duration-200 rounded-tr-card">Sequential vs Global</th>
                  </tr>
                </thead>
                <tbody>
                  {simulatedStudents.length === 0 ? (
                    <tr>
                      <td colSpan="12" className="px-2 py-1 text-center text-gray-500 text-xs">
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
                        <tr key={student.student_id} className={`border-b border-transparent ${index % 2 === 0 ? 'bg-charcoal-elevated' : 'bg-charcoal-elevated'} hover:bg-charcoal-elevated transition-all duration-200 ease-in-out`}>
                          <td className="px-2 py-1 whitespace-nowrap">
                            <input
                              type="checkbox"
                              checked={selectedStudents.includes(student.student_id)}
                              onChange={() => handleCheckboxChange(student.student_id)}
                              className="form-checkbox"
                            />
                          </td>
                          <td className="px-2 py-1 whitespace-nowrap text-xs text-gray-300">{student.student_id}</td>
                          <td className="px-2 py-1 whitespace-nowrap text-xs text-gray-300">{student.days_old}</td>
                          <td className="px-2 py-1 whitespace-nowrap text-xs text-gray-300">{student.age}</td>
                          <td className="px-2 py-1 whitespace-nowrap text-xs text-gray-300">{student.gender}</td>
                          <td className="px-2 py-1 whitespace-nowrap text-xs text-gray-300">{student.academic_program}</td>
                          <td className="px-2 py-1 whitespace-nowrap text-xs text-gray-300">{student.year_level}</td>
                          <td className="px-2 py-1 whitespace-nowrap text-xs text-gray-300">{student.gpa}</td>
                          <td className={`px-2 py-1 whitespace-nowrap text-xs text-gray-300 capitalize ${getHighlightClass(student.active_vs_reflective, prevStudent?.active_vs_reflective)}`}>{student.active_vs_reflective}</td>
                          <td className={`px-2 py-1 whitespace-nowrap text-xs text-gray-300 capitalize ${getHighlightClass(student.sensing_vs_intuitive, prevStudent?.sensing_vs_intuitive)}`}>{student.sensing_vs_intuitive}</td>
                          <td className={`px-2 py-1 whitespace-nowrap text-xs text-gray-300 capitalize ${getHighlightClass(student.visual_vs_verbal, prevStudent?.visual_vs_verbal)}`}>{student.visual_vs_verbal}</td>
                          <td className={`px-2 py-1 whitespace-nowrap text-xs text-gray-300 capitalize ${getHighlightClass(student.sequential_vs_global, prevStudent?.sequential_vs_global)}`}>{student.sequential_vs_global}</td>
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
        <div className="fixed inset-0 bg-deep-space-navy bg-opacity-75 flex flex-col items-center justify-center z-50">
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