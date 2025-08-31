"use client";
import { useState, useEffect } from 'react';

export default function DashboardPage() {
  const [studentsData, setStudentsData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterLearningStyle, setFilterLearningStyle] = useState('all');
  const [showFilterDropdown, setShowFilterDropdown] = useState(false);

  useEffect(() => {
    async function fetchStudents() {
      try {
        const response = await fetch('/api/students');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setStudentsData(data);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    }
    fetchStudents();
  }, []);

  const handleSearchChange = (event) => {
    setSearchTerm(event.target.value);
  };

  const handleFilterChange = (style) => {
    setFilterLearningStyle(style);
    setShowFilterDropdown(false);
  };

  const toggleFilterDropdown = () => {
    setShowFilterDropdown(!showFilterDropdown);
  };

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (showFilterDropdown && !event.target.closest('.filter-dropdown-container')) {
        setShowFilterDropdown(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showFilterDropdown]);

  const filteredStudents = studentsData?.students.filter(student => {
    const matchesSearch = searchTerm === '' ||
      Object.values(student).some(value =>
        String(value).toLowerCase().includes(searchTerm.toLowerCase())
      );

    const matchesFilter = filterLearningStyle === 'all' ||
      student.active_vs_reflective === filterLearningStyle ||
      student.sensing_vs_intuitive === filterLearningStyle ||
      student.visual_vs_verbal === filterLearningStyle ||
      student.sequential_vs_global === filterLearningStyle;

    return matchesSearch && matchesFilter;
  }) || [];

  if (loading) return (
    <div className="min-h-screen bg-dark-navy text-white flex items-center justify-center">
      <p>Loading dashboard data...</p>
    </div>
  );
  if (error) return (
    <div className="min-h-screen bg-dark-navy text-white flex items-center justify-center">
      <p className="text-red-500">Error: {error}</p>
    </div>
  );

  const { totalStudents, studentsPerLearningStyle } = studentsData;

  return (
    <div className="bg-dark-navy text-white p-2 h-screen flex flex-col overflow-hidden">

      {/* Interpretation and Metrics */}
      <div className="bg-charcoal p-3 rounded-xl shadow-lg mb-3 flex-shrink-0 border border-transparent hover:border-accent-blue transition-all duration-300 ease-in-out">
        <h2 className="text-xl font-bold mb-3 text-accent-blue">Data Interpretation and Metrics</h2>
        <p className="mb-3 text-sm text-gray-300"><strong>Total Students:</strong> <span className="text-accent-cyan font-bold">{totalStudents}</span></p>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
          {Object.entries(studentsPerLearningStyle).map(([styleType, styles]) => (
            <div key={styleType} className="relative bg-gradient-to-br from-charcoal to-subtle-gray-dark p-3 rounded-lg shadow-md border border-subtle-gray-light hover:shadow-xl transition-all duration-300 ease-in-out">
              <h3 className="font-semibold capitalize text-sm mb-2 text-accent-blue">{styleType.replace(/_/g, ' ')}</h3>
              {Object.entries(styles).map(([style, count]) => {
                const percentage = totalStudents > 0 ? ((count / totalStudents) * 100).toFixed(1) : 0;
                return (
                  <div key={style} className="mb-2">
                    <div className="flex justify-between items-center text-gray-300 text-xs mb-1">
                      <span className="capitalize">{style}:</span>
                      <span className="font-semibold">{count} students ({percentage}%)</span>
                    </div>
                    <div className="w-full bg-subtle-gray-dark rounded-full h-1.5">
                      <div
                        className="bg-accent-cyan h-1.5 rounded-full"
                        style={{ width: `${percentage}%` }}
                      ></div>
                    </div>
                  </div>
                );
              })}
            </div>
          ))}
        </div>
        <p className="mt-3 text-gray-400 text-xs leading-relaxed">
          This dashboard provides comprehensive insights into student demographics and learning styles, offering a detailed summary of student distribution across various learning dimensions.
        </p>
      </div>

     {/* Student Table */}
    <div className="bg-charcoal p-2 rounded-xl shadow-lg flex flex-col flex-grow min-h-0">
      <h2 className="text-xl font-bold mb-3 text-accent-blue">Student Information</h2>
      {/* Filter and Search */}
      <div className="mb-3 flex flex-row gap-2 items-center flex-wrap">
        <div className="flex-grow">
          <label htmlFor="search" className="sr-only">Search Students:</label>
          <input
            type="text"
            id="search"
            className="w-full px-3 py-1.5 rounded-full bg-subtle-gray-dark border border-subtle-gray-light text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-accent-blue focus:border-accent-blue transition-all duration-300 ease-in-out text-sm shadow-inner"
            placeholder="Search by ID, program, or learning style..."
            value={searchTerm}
            onChange={handleSearchChange}
          />
        </div>
        <div className="relative filter-dropdown-container">
          <button
            onClick={toggleFilterDropdown}
            className="w-full px-3 py-1.5 rounded-full bg-subtle-gray-dark border border-subtle-gray-light text-white focus:outline-none focus:ring-2 focus:ring-accent-blue focus:border-accent-blue transition-all duration-300 ease-in-out text-sm shadow-inner flex items-center justify-center space-x-1 hover:bg-subtle-gray-light"
          >
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 3c2.75 0 5 2.25 5 5v.75a.75.75 0 01-.75.75H7.75a.75.75 0 01-.75-.75V8c0-2.75 2.25-5 5-5z" />
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 10.5a2.25 2.25 0 100 4.5 2.25 2.25 0 000-4.5z" />
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 16.5c-2.75 0-5-2.25-5-5v-.75a.75.75 0 01.75-.75h8.5a.75.75 0 01.75.75v.75c0 2.75-2.25 5-5 5z" />
            </svg>
            <span>Filter</span>
          </button>
          {showFilterDropdown && (
            <div className="absolute right-0 mt-2 w-56 rounded-md shadow-xl bg-charcoal ring-1 ring-black ring-opacity-5 z-10">
              <div className="py-1" role="menu" aria-orientation="vertical" aria-labelledby="options-menu">
                <button
                  onClick={() => handleFilterChange('all')}
                  className="block w-full text-left px-3 py-1.5 text-sm text-gray-300 hover:bg-subtle-gray-light"
                  role="menuitem"
                >
                  All Learning Styles
                </button>
                <div className="border-t border-subtle-gray-dark my-0.5"></div>
                <p className="px-3 py-1 text-xs text-gray-400 uppercase">Active vs Reflective</p>
                <button
                  onClick={() => handleFilterChange('active')}
                  className="block w-full text-left px-3 py-1.5 text-sm text-gray-300 hover:bg-subtle-gray-light"
                  role="menuitem"
                >
                  Active
                </button>
                <button
                  onClick={() => handleFilterChange('reflective')}
                  className="block w-full text-left px-3 py-1.5 text-sm text-gray-300 hover:bg-subtle-gray-light"
                  role="menuitem"
                >
                  Reflective
                </button>
                <div className="border-t border-subtle-gray-dark my-0.5"></div>
                <p className="px-3 py-1 text-xs text-gray-400 uppercase">Sensing vs Intuitive</p>
                <button
                  onClick={() => handleFilterChange('sensing')}
                  className="block w-full text-left px-3 py-1.5 text-sm text-gray-300 hover:bg-subtle-gray-light"
                  role="menuitem"
                >
                  Sensing
                </button>
                <button
                  onClick={() => handleFilterChange('intuitive')}
                  className="block w-full text-left px-3 py-1.5 text-sm text-gray-300 hover:bg-subtle-gray-light"
                  role="menuitem"
                >
                  Intuitive
                </button>
                <div className="border-t border-subtle-gray-dark my-0.5"></div>
                <p className="px-3 py-1 text-xs text-gray-400 uppercase">Visual vs Verbal</p>
                <button
                  onClick={() => handleFilterChange('visual')}
                  className="block w-full text-left px-3 py-1.5 text-sm text-gray-300 hover:bg-subtle-gray-light"
                  role="menuitem"
                >
                  Visual
                </button>
                <button
                  onClick={() => handleFilterChange('verbal')}
                  className="block w-full text-left px-3 py-1.5 text-sm text-gray-300 hover:bg-subtle-gray-light"
                  role="menuitem"
                >
                  Verbal
                </button>
                <div className="border-t border-subtle-gray-dark my-0.5"></div>
                <p className="px-3 py-1 text-xs text-gray-400 uppercase">Sequential vs Global</p>
                <button
                  onClick={() => handleFilterChange('sequential')}
                  className="block w-full text-left px-3 py-1.5 text-sm text-gray-300 hover:bg-subtle-gray-light"
                  role="menuitem"
                >
                  Sequential
                </button>
                <button
                  onClick={() => handleFilterChange('global')}
                  className="block w-full text-left px-3 py-1.5 text-sm text-gray-300 hover:bg-subtle-gray-light"
                  role="menuitem"
                >
                  Global
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
      <div className="overflow-x-auto overflow-y-auto relative rounded-lg border border-subtle-gray-dark" style={{ maxHeight: 'calc(100vh - 150px)' }}>
         <table className="w-full text-left table-auto table-compact">
           <thead className="bg-charcoal sticky top-0 border-b border-subtle-gray-light">
             <tr>
               <th className="px-1 py-0.5 text-left text-xs font-bold text-gray-200 uppercase tracking-wider cursor-pointer hover:bg-subtle-gray-light transition-colors duration-200 rounded-tl-lg">ID</th>
               <th className="px-1 py-0.5 text-left text-xs font-bold text-gray-200 uppercase tracking-wider cursor-pointer hover:bg-subtle-gray-light transition-colors duration-200">Age</th>
               <th className="px-1 py-0.5 text-left text-xs font-bold text-gray-200 uppercase tracking-wider cursor-pointer hover:bg-subtle-gray-light transition-colors duration-200">Gender</th>
               <th className="px-1 py-0.5 text-left text-xs font-bold text-gray-200 uppercase tracking-wider cursor-pointer hover:bg-subtle-gray-light transition-colors duration-200">Program</th>
               <th className="px-1 py-0.5 text-left text-xs font-bold text-gray-200 uppercase tracking-wider cursor-pointer hover:bg-subtle-gray-light transition-colors duration-200">Year</th>
               <th className="px-1 py-0.5 text-left text-xs font-bold text-gray-200 uppercase tracking-wider cursor-pointer hover:bg-subtle-gray-light transition-colors duration-200">GPA</th>
               <th className="px-1 py-0.5 text-left text-xs font-bold text-gray-200 uppercase tracking-wider cursor-pointer hover:bg-subtle-gray-light transition-colors duration-200">Active/Reflective</th>
               <th className="px-1 py-0.5 text-left text-xs font-bold text-gray-200 uppercase tracking-wider cursor-pointer hover:bg-subtle-gray-light transition-colors duration-200">Sensing/Intuitive</th>
               <th className="px-1 py-0.5 text-left text-xs font-bold text-gray-200 uppercase tracking-wider cursor-pointer hover:bg-subtle-gray-light transition-colors duration-200">Visual/Verbal</th>
               <th className="px-1 py-0.5 text-left text-xs font-bold text-gray-200 uppercase tracking-wider cursor-pointer hover:bg-subtle-gray-light transition-colors duration-200 rounded-tr-lg">Sequential/Global</th>
             </tr>
           </thead>
           <tbody>
             {filteredStudents.map((student, index) => (
               <tr key={`${student.student_id}-${index}`} className={`border-b border-subtle-gray-dark ${index % 2 === 0 ? 'bg-subtle-gray-dark' : 'bg-charcoal'} hover:bg-subtle-gray-light transition-all duration-200 ease-in-out`}>
                 <td className="px-2 py-1 whitespace-nowrap text-sm text-gray-300">{student.student_id}</td>
                 <td className="px-2 py-1 whitespace-nowrap text-sm text-gray-300">{student.age}</td>
                 <td className="px-2 py-1 whitespace-nowrap text-sm text-gray-300">{student.gender}</td>
                 <td className="px-2 py-1 whitespace-nowrap text-sm text-gray-300">{student.academic_program}</td>
                 <td className="px-2 py-1 whitespace-nowrap text-sm text-gray-300">{student.year_level}</td>
                 <td className="px-2 py-1 whitespace-nowrap text-sm text-gray-300">{student.GPA}</td>
                 <td className="px-2 py-1 whitespace-nowrap text-sm text-gray-300 capitalize">{student.active_vs_reflective}</td>
                 <td className="px-2 py-1 whitespace-nowrap text-sm text-gray-300 capitalize">{student.sensing_vs_intuitive}</td>
                 <td className="px-2 py-1 whitespace-nowrap text-sm text-gray-300 capitalize">{student.visual_vs_verbal}</td>
                 <td className="px-2 py-1 whitespace-nowrap text-sm text-gray-300 capitalize">{student.sequential_vs_global}</td>
               </tr>
             ))}
             {filteredStudents.length === 0 && (
               <tr>
                 <td colSpan="10" className="px-2 py-1 text-center text-gray-500 text-xs">No students found matching your criteria.</td>
               </tr>
             )}
           </tbody>
         </table>
       </div>
     </div>
    </div>
  );
}