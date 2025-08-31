"use client";

import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar
} from 'recharts';
import DashboardLayout from '../dashboard/layout'; // Import DashboardLayout

export default function ModelStatsPage() {
  const [modelStats, setModelStats] = useState(null);
  const [featureImportances, setFeatureImportances] = useState(null);
  const [confusionMatrices, setConfusionMatrices] = useState(null); // New state for confusion matrices
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statsResponse, importancesResponse, confusionMatrixResponse] = await Promise.all([
          fetch('http://127.0.0.1:8000/model-performance'),
          fetch('http://127.0.0.1:8000/feature-importances'),
          fetch('http://127.0.0.1:8000/confusion-matrix') // Fetch confusion matrix data
        ]);

        if (!statsResponse.ok) {
          throw new Error(`HTTP error! status: ${statsResponse.status}`);
        }
        if (!importancesResponse.ok) {
          throw new Error(`HTTP error! status: ${importancesResponse.status}`);
        }

        const statsData = await statsResponse.json();
        const importancesData = await importancesResponse.json();
        const confusionMatrixData = await confusionMatrixResponse.json(); // Parse confusion matrix data

        console.log('Fetched modelStats:', JSON.stringify(statsData, null, 2));
        console.log('Fetched featureImportances:', JSON.stringify(importancesData, null, 2));
        console.log('Fetched confusionMatrices:', JSON.stringify(confusionMatrixData, null, 2));

        setModelStats(statsData);
        setFeatureImportances(importancesData);
        setConfusionMatrices(confusionMatrixData); // Set confusion matrix data
      } catch (e) {
        setError(e);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return <div className="min-h-screen bg-gray-900 text-white p-8 flex justify-center items-center text-2xl">Loading model statistics...</div>;
  }

  if (error) {
    return <div className="min-h-screen bg-gray-900 text-white p-8 flex justify-center items-center text-2xl text-red-500">Error: {error.message}</div>;
  }

  // Assuming a single model for now, you might want to iterate if multiple models are returned
  // For overall display, we can use the first model's data or an aggregate if needed
  const firstModelKey = Object.keys(modelStats)[0];
  const model = modelStats[firstModelKey]; // Still use first model for overall stats

  if (!model) {
    return <div className="min-h-screen bg-gray-900 text-white p-8 flex justify-center items-center text-2xl">No model data available.</div>;
  }

  const calculateStrokeDashoffset = (value) => {
    const numericValue = parseFloat(value);
    if (isNaN(numericValue)) {
      return 0; // Return 0 or a sensible default if the value is NaN
    }
    const circumference = 2 * Math.PI * 40; // r=40 from SVG
    const progress = numericValue / 100;
    return circumference * (1 - progress);
  };

  const formatPercentage = (value, decimals = 1) => {
    const numericValue = parseFloat(value);
    if (isNaN(numericValue)) {
      return "N/A"; // Return "N/A" for display if NaN
    }
    return (numericValue * 100).toFixed(decimals); // Return string without '%'
  };

  return (
    <DashboardLayout>
      <div className="bg-dark-navy text-white p-2 min-h-screen overflow-y-auto">

      {/* Model Overview Card */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 mb-3">
        <div className="bg-charcoal rounded-xl p-3 shadow-lg border border-transparent hover:border-accent-blue transition-all duration-300 ease-in-out">
          <h2 className="text-xl font-bold mb-3 text-accent-blue">Overall Accuracy</h2>
          <div className="text-center">
            <p className="text-5xl font-extrabold text-accent-cyan">{formatPercentage(model.test_accuracy)}%</p>
            <span className="text-xs text-gray-400 mt-2">Excellent Performance</span>
          </div>
        </div>

        {/* Key Performance Indicators (KPIs) */}
        <div className="bg-charcoal rounded-xl p-3 shadow-lg border border-transparent hover:border-accent-blue transition-all duration-300 ease-in-out">
          <h2 className="text-xl font-bold mb-3 text-accent-blue">Model Details</h2>
          <div className="space-y-1 text-sm">
            <p className="flex justify-between items-center"><span className="font-medium text-gray-300">Model Type:</span> <span className="text-accent-cyan">{model.model_type}</span></p>
            <p className="flex justify-between items-center"><span className="font-medium text-gray-300">Last Trained:</span> <span className="text-accent-cyan">{model.last_trained}</span></p>
            <p className="flex justify-between items-center"><span className="font-medium text-gray-300">Dataset Size:</span> <span className="text-accent-cyan">{model.dataset_size} samples</span></p>
            <p className="flex justify-between items-center"><span className="font-medium text-gray-300">Features Used:</span> <span className="text-accent-cyan">{model.features_used}</span></p>
          </div>
        </div>

        {/* Model Health Status */}
        <div className="bg-charcoal rounded-xl p-3 shadow-lg border border-transparent hover:border-accent-blue transition-all duration-300 ease-in-out">
          <h2 className="text-xl font-bold mb-3 text-accent-blue">Model Health</h2>
          <div className="flex flex-col items-center justify-center flex-1">
            <div className="relative w-24 h-24 flex items-center justify-center">
              <svg className="w-full h-full absolute">
                <circle className="text-subtle-gray-dark" strokeWidth="8" stroke="currentColor" fill="transparent" r="40" cx="50%" cy="50%" />
                <circle className="text-accent-cyan" strokeWidth="8" strokeDasharray="251.32" strokeDashoffset={calculateStrokeDashoffset(parseFloat(formatPercentage(model.test_accuracy, 0)))} strokeLinecap="round" stroke="currentColor" fill="transparent" r="40" cx="50%" cy="50%" transform="rotate(-90 48 48)" />
              </svg>
              <div className="absolute w-full h-full flex items-center justify-center text-xl font-bold text-accent-cyan">
                {formatPercentage(model.test_accuracy, 0)}%
              </div>
            </div>
            <p className="text-xs text-gray-400 mt-2">Operational & Stable</p>
          </div>
        </div>
      </div>

      {/* Metrics Visualization Area */}
      <div className="bg-charcoal rounded-xl p-3 shadow-lg mb-3 border border-transparent hover:border-accent-blue transition-all duration-300 ease-in-out">
        <h2 className="text-xl font-bold mb-3 text-accent-blue">Performance Metrics</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
          {/* Gauge Chart for Accuracy */}
          <div className="flex flex-col items-center bg-subtle-gray-dark rounded-lg p-2 shadow-md">
            <div className="relative w-24 h-24 flex items-center justify-center">
              <svg className="w-full h-full absolute">
                <circle className="text-subtle-gray-light" strokeWidth="8" stroke="currentColor" fill="transparent" r="40" cx="50%" cy="50%" />
                <circle className="text-accent-cyan" strokeWidth="8" strokeDasharray="251.32" strokeDashoffset={calculateStrokeDashoffset(parseFloat(formatPercentage(model.test_accuracy, 0)))} strokeLinecap="round" stroke="currentColor" fill="transparent" r="40" cx="50%" cy="50%" transform="rotate(-90 48 48)" />
              </svg>
              <div className="absolute w-full h-full flex items-center justify-center text-xl font-bold text-accent-cyan">
                {formatPercentage(model.test_accuracy)}%
              </div>
            </div>
            <p className="text-sm mt-1 text-gray-300">Accuracy</p>
          </div>

          {/* Gauge Chart for Precision */}
          <div className="flex flex-col items-center bg-subtle-gray-dark rounded-lg p-2 shadow-md">
            <div className="relative w-24 h-24 flex items-center justify-center">
              <svg className="w-full h-full absolute">
                <circle className="text-subtle-gray-light" strokeWidth="8" stroke="currentColor" fill="transparent" r="40" cx="50%" cy="50%" />
                <circle className="text-accent-blue" strokeWidth="8" strokeDasharray="251.32" strokeDashoffset={calculateStrokeDashoffset(parseFloat(model.precision || 0) * 100)} strokeLinecap="round" stroke="currentColor" fill="transparent" r="40" cx="50%" cy="50%" transform="rotate(-90 48 48)" />
              </svg>
              <div className="absolute w-full h-full flex items-center justify-center text-xl font-bold text-accent-blue">
                {formatPercentage(model.precision)}%
              </div>
            </div>
            <p className="text-sm mt-1 text-gray-300">Precision</p>
          </div>

          {/* Gauge Chart for Recall */}
          <div className="flex flex-col items-center bg-subtle-gray-dark rounded-lg p-2 shadow-md">
            <div className="relative w-24 h-24 flex items-center justify-center">
              <svg className="w-full h-full absolute">
                <circle className="text-subtle-gray-light" strokeWidth="8" stroke="currentColor" fill="transparent" r="40" cx="50%" cy="50%" />
                <circle className="text-accent-blue" strokeWidth="8" strokeDasharray="251.32" strokeDashoffset={calculateStrokeDashoffset(parseFloat(model.recall || 0) * 100)} strokeLinecap="round" stroke="currentColor" fill="transparent" r="40" cx="50%" cy="50%" transform="rotate(-90 48 48)" />
              </svg>
              <div className="absolute w-full h-full flex items-center justify-center text-xl font-bold text-accent-blue">
                {formatPercentage(model.recall)}%
              </div>
            </div>
            <p className="text-sm mt-1 text-gray-300">Recall</p>
          </div>

          {/* Gauge Chart for F1-Score */}
          <div className="flex flex-col items-center bg-subtle-gray-dark rounded-lg p-2 shadow-md">
            <div className="relative w-24 h-24 flex items-center justify-center">
              <svg className="w-full h-full absolute">
                <circle className="text-subtle-gray-light" strokeWidth="8" stroke="currentColor" fill="transparent" r="40" cx="50%" cy="50%" />
                <circle className="text-accent-blue" strokeWidth="8" strokeDasharray="251.32" strokeDashoffset={calculateStrokeDashoffset(parseFloat(model.f1_score || 0) * 100)} strokeLinecap="round" stroke="currentColor" fill="transparent" r="40" cx="50%" cy="50%" transform="rotate(-90 48 48)" />
              </svg>
              <div className="absolute w-full h-full flex items-center justify-center text-xl font-bold text-accent-blue">
                {formatPercentage(model.f1_score)}%
              </div>
            </div>
            <p className="text-sm mt-1 text-gray-300">F1-Score</p>
          </div>
        </div>
      </div>

      {/* Cross-Validation Results Section (Using Line Chart for Accuracy over different targets) */}
      <div className="bg-charcoal rounded-xl p-3 shadow-lg mb-3 border border-transparent hover:border-accent-blue transition-all duration-300 ease-in-out">
        <h2 className="text-xl font-bold mb-3 text-accent-blue">Accuracy by Learning Style Dimension</h2>
        <div className="h-[350px]"> {/* Explicit height for the chart container */}
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={Object.keys(modelStats).map(key => ({
                name: key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
                accuracy: parseFloat(formatPercentage(modelStats[key].test_accuracy, 2)) || 0
              }))}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#2D333B" />
              <XAxis dataKey="name" stroke="#9CA3AF" interval="preserveStartEnd" angle={-15} textAnchor="end" />
              <YAxis stroke="#9CA3AF" domain={[0, 100]} />
              <Tooltip
                contentStyle={{ backgroundColor: '#161B22', borderColor: '#2D333B', color: '#E5E7EB' }}
                labelStyle={{ color: '#9CA3AF' }}
              />
              <Legend wrapperStyle={{ color: '#E5E7EB' }} />
              <Line type="monotone" dataKey="accuracy" stroke="#00FFFF" activeDot={{ r: 6 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Feature Importance Visualization (Using Bar Chart) */}
      <div className="bg-charcoal rounded-xl p-3 shadow-lg mb-3 border border-transparent hover:border-accent-blue transition-all duration-300 ease-in-out">
        <h2 className="text-xl font-bold mb-3 text-accent-blue">Feature Importance (Overall)</h2>
        <div className="h-[350px]"> {/* Explicit height for the chart container */}
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              layout="vertical"
              data={Object.entries(featureImportances['ACTIVE_VS_REFLECTIVE']) // Default to first model for overall feature importance
                .sort(([, a], [, b]) => b - a)
                .slice(0, 10) // Display top 10 features
                .map(([name, value]) => ({ name: name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()), value: parseFloat(formatPercentage(value, 2)) || 0 }))}
              margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#2D333B" />
              <XAxis type="number" stroke="#9CA3AF" />
              <YAxis type="category" dataKey="name" stroke="#9CA3AF" width={180} />
              <Tooltip
                contentStyle={{ backgroundColor: '#161B22', borderColor: '#2D333B', color: '#E5E7EB' }}
                labelStyle={{ color: '#9CA3AF' }}
              />
              <Legend wrapperStyle={{ color: '#E5E7EB' }} />
              <Bar dataKey="value" fill="#007BFF" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Confusion Matrix Heatmap */}
      <div className="bg-charcoal rounded-xl p-3 shadow-lg mb-3 border border-transparent hover:border-accent-blue transition-all duration-300 ease-in-out">
        <h2 className="text-xl font-bold mb-3 text-accent-blue">Confusion Matrix</h2>
        {confusionMatrices && Object.keys(confusionMatrices).length > 0 ? (
          <div className="overflow-x-auto overflow-y-auto relative rounded-lg border border-subtle-gray-dark" style={{ maxHeight: '400px' }}>
            {Object.keys(confusionMatrices).map(target => (
              <div key={target} className="mb-4">
                <h3 className="text-md font-semibold mb-2 text-gray-300">{target.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</h3>
                <table className="w-full text-left table-auto table-compact">
                  <thead className="bg-charcoal sticky top-0 border-b border-subtle-gray-light">
                    <tr>
                      <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider"></th>
                      <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">Predicted 0</th>
                      <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">Predicted 1</th>
                    </tr>
                  </thead>
                  <tbody className="text-gray-300 text-sm font-light">
                    <tr className="border-b border-subtle-gray-dark bg-subtle-gray-dark hover:bg-subtle-gray-light transition-all duration-200 ease-in-out">
                      <td className="px-2 py-1 font-medium">Actual 0</td>
                      <td className="px-2 py-1 bg-subtle-gray-dark text-accent-cyan font-bold">{confusionMatrices[target].confusion_matrix[0][0]}</td>
                      <td className="px-2 py-1 bg-charcoal text-red-400 font-bold">{confusionMatrices[target].confusion_matrix[0][1]}</td>
                    </tr>
                    <tr className="border-b border-subtle-gray-dark bg-charcoal hover:bg-subtle-gray-light transition-all duration-200 ease-in-out">
                      <td className="px-2 py-1 font-medium">Actual 1</td>
                      <td className="px-2 py-1 bg-charcoal text-green-400 font-bold">{confusionMatrices[target].confusion_matrix[1][0]}</td>
                      <td className="px-2 py-1 bg-subtle-gray-dark text-accent-cyan font-bold">{confusionMatrices[target].confusion_matrix[1][1]}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            ))}
          </div>
        ) : (
          <div className="h-48 flex items-center justify-center text-gray-500 text-sm">
            No Confusion Matrix data available.
          </div>
        )}
      </div>

      {/* Model Comparison Section */}
      <div className="bg-charcoal rounded-xl p-3 shadow-lg mb-3 border border-transparent hover:border-accent-blue transition-all duration-300 ease-in-out">
        <h2 className="text-xl font-bold mb-3 text-accent-blue">Model Comparison</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
          {Object.keys(modelStats).map(target => (
            <div key={target} className="bg-subtle-gray-dark rounded-lg p-2 shadow-md">
              <h3 className="text-md font-semibold mb-2 text-center text-accent-blue">{target.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</h3>
              <div className="space-y-1 text-xs">
                <p className="flex justify-between"><span>Accuracy:</span> <span className="font-medium text-accent-cyan">{formatPercentage(modelStats[target].test_accuracy)}%</span></p>
                <p className="flex justify-between"><span>Precision:</span> <span className="font-medium text-accent-cyan">{formatPercentage(modelStats[target].precision)}%</span></p>
                <p className="flex justify-between"><span>Recall:</span> <span className="font-medium text-accent-cyan">{formatPercentage(modelStats[target].recall)}%</span></p>
                <p className="flex justify-between"><span>F1-Score:</span> <span className="font-medium text-accent-cyan">{formatPercentage(modelStats[target].f1_score)}%</span></p>
                <p className="flex justify-between"><span>Dataset Size:</span> <span className="font-medium text-gray-300">{modelStats[target].dataset_size}</span></p>
              </div>
            </div>
          ))}
        </div>
      </div>


      {/* Classification Report Table (Placeholder with enhanced styling) */}
      <div className="bg-charcoal rounded-xl p-3 shadow-lg border border-transparent hover:border-accent-blue transition-all duration-300 ease-in-out">
        <h2 className="text-xl font-bold mb-3 text-accent-blue">Classification Report</h2>
        <div className="overflow-x-auto overflow-y-auto relative rounded-lg border border-subtle-gray-dark" style={{ maxHeight: '400px' }}>
          <table className="w-full text-left table-auto table-compact">
            <thead className="bg-charcoal sticky top-0 border-b border-subtle-gray-light">
              <tr>
                <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">Class</th>
                <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">Precision</th>
                <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">Recall</th>
                <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">F1-Score</th>
                <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">Support</th>
              </tr>
            </thead>
            <tbody className="text-gray-300 text-sm font-light">
              {Object.keys(modelStats).map((target, index) => (
                <tr key={target} className={`border-b border-subtle-gray-dark ${index % 2 === 0 ? 'bg-subtle-gray-dark' : 'bg-charcoal'} hover:bg-subtle-gray-light transition-all duration-200 ease-in-out`}>
                  <td className="px-2 py-1 whitespace-nowrap text-sm text-gray-300">{target.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</td>
                  <td className="px-2 py-1 whitespace-nowrap text-sm text-gray-300">{formatPercentage(modelStats[target].test_accuracy, 2)}</td>
                  <td className="px-2 py-1 whitespace-nowrap text-sm text-gray-300">{formatPercentage(modelStats[target].recall, 2)}</td>
                  <td className="px-2 py-1 whitespace-nowrap text-sm text-gray-300">{formatPercentage(modelStats[target].f1_score, 2)}</td>
                  <td className="px-2 py-1 whitespace-nowrap text-sm text-gray-300">{modelStats[target].dataset_size}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      </div>
    </DashboardLayout>
  );
}