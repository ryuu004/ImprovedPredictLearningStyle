"use client";

import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar
} from 'recharts';

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
    const circumference = 2 * Math.PI * 50; // r=50 from SVG
    const progress = value / 100;
    return circumference * (1 - progress);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <h1 className="text-4xl font-bold mb-8 text-center text-purple-400">Model Performance Dashboard</h1>

      {/* Model Overview Card */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-12">
        <div className="bg-gray-800 rounded-lg p-6 shadow-lg transform transition duration-300 hover:scale-105 hover:shadow-2xl">
          <h2 className="text-2xl font-semibold mb-4 text-center text-teal-300">Overall Accuracy</h2>
          <div className="text-center">
            <p className="text-6xl font-extrabold text-green-500">{(model.accuracy * 100).toFixed(1)}%</p>
            <span className="text-sm text-gray-400 mt-2">Excellent Performance</span>
          </div>
        </div>

        {/* Key Performance Indicators (KPIs) */}
        <div className="bg-gray-800 rounded-lg p-6 shadow-lg transform transition duration-300 hover:scale-105 hover:shadow-2xl">
          <h2 className="text-2xl font-semibold mb-4 text-center text-teal-300">Model Details</h2>
          <div className="space-y-2">
            <p className="flex justify-between items-center text-lg"><span className="font-medium text-gray-300">Model Type:</span> <span className="text-blue-400">{model.model_type}</span></p>
            <p className="flex justify-between items-center text-lg"><span className="font-medium text-gray-300">Last Trained:</span> <span className="text-blue-400">{model.last_trained}</span></p>
            <p className="flex justify-between items-center text-lg"><span className="font-medium text-gray-300">Dataset Size:</span> <span className="text-blue-400">{model.dataset_size} samples</span></p>
            <p className="flex justify-between items-center text-lg"><span className="font-medium text-gray-300">Features Used:</span> <span className="text-blue-400">{model.features_used}</span></p>
          </div>
        </div>

        {/* Model Health Status */}
        <div className="bg-gray-800 rounded-lg p-6 shadow-lg transform transition duration-300 hover:scale-105 hover:shadow-2xl">
          <h2 className="text-2xl font-semibold mb-4 text-center text-teal-300">Model Health</h2>
          <div className="flex flex-col items-center justify-center h-full">
            <div className="relative w-32 h-32">
              <svg className="w-full h-full transform -rotate-90">
                <circle className="text-gray-700" strokeWidth="10" stroke="currentColor" fill="transparent" r="50" cx="60" cy="60" />
                <circle className="text-green-500" strokeWidth="10" strokeDasharray="314" strokeDashoffset={calculateStrokeDashoffset(model.accuracy * 100)} strokeLinecap="round" stroke="currentColor" fill="transparent" r="50" cx="60" cy="60" />
              </svg>
              <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-3xl font-bold text-green-500">{(model.accuracy * 100).toFixed(0)}%</div>
            </div>
            <p className="text-sm text-gray-400 mt-4">Operational & Stable</p>
          </div>
        </div>
      </div>

      {/* Metrics Visualization Area */}
      <div className="bg-gray-800 rounded-lg p-8 shadow-lg mb-12">
        <h2 className="text-3xl font-semibold mb-8 text-center text-yellow-300">Performance Metrics</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {/* Gauge Chart for Accuracy */}
          <div className="flex flex-col items-center">
            <div className="relative w-32 h-32">
              <svg className="w-full h-full transform -rotate-90">
                <circle className="text-gray-700" strokeWidth="10" stroke="currentColor" fill="transparent" r="50" cx="60" cy="60" />
                <circle className="text-green-500" strokeWidth="10" strokeDasharray="314" strokeDashoffset={calculateStrokeDashoffset(model.accuracy * 100)} strokeLinecap="round" stroke="currentColor" fill="transparent" r="50" cx="60" cy="60" />
              </svg>
              <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-2xl font-bold text-green-500">{(model.accuracy * 100).toFixed(1)}%</div>
            </div>
            <p className="text-lg mt-2 text-gray-300">Accuracy</p>
          </div>

          {/* Gauge Chart for Precision */}
          <div className="flex flex-col items-center">
            <div className="relative w-32 h-32">
              <svg className="w-full h-full transform -rotate-90">
                <circle className="text-gray-700" strokeWidth="10" stroke="currentColor" fill="transparent" r="50" cx="60" cy="60" />
                <circle className="text-blue-400" strokeWidth="10" strokeDasharray="314" strokeDashoffset={calculateStrokeDashoffset(model.precision * 100)} strokeLinecap="round" stroke="currentColor" fill="transparent" r="50" cx="60" cy="60" />
              </svg>
              <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-2xl font-bold text-blue-400">{(model.precision * 100).toFixed(1)}%</div>
            </div>
            <p className="text-lg mt-2 text-gray-300">Precision</p>
          </div>

          {/* Gauge Chart for Recall */}
          <div className="flex flex-col items-center">
            <div className="relative w-32 h-32">
              <svg className="w-full h-full transform -rotate-90">
                <circle className="text-gray-700" strokeWidth="10" stroke="currentColor" fill="transparent" r="50" cx="60" cy="60" />
                <circle className="text-purple-400" strokeWidth="10" strokeDasharray="314" strokeDashoffset={calculateStrokeDashoffset(model.recall * 100)} strokeLinecap="round" stroke="currentColor" fill="transparent" r="50" cx="60" cy="60" />
              </svg>
              <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-2xl font-bold text-purple-400">{(model.recall * 100).toFixed(1)}%</div>
            </div>
            <p className="text-lg mt-2 text-gray-300">Recall</p>
          </div>

          {/* Gauge Chart for F1-Score */}
          <div className="flex flex-col items-center">
            <div className="relative w-32 h-32">
              <svg className="w-full h-full transform -rotate-90">
                <circle className="text-gray-700" strokeWidth="10" stroke="currentColor" fill="transparent" r="50" cx="60" cy="60" />
                <circle className="text-pink-400" strokeWidth="10" strokeDasharray="314" strokeDashoffset={calculateStrokeDashoffset(model.f1_score * 100)} strokeLinecap="round" stroke="currentColor" fill="transparent" r="50" cx="60" cy="60" />
              </svg>
              <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-2xl font-bold text-pink-400">{(model.f1_score * 100).toFixed(1)}%</div>
            </div>
            <p className="text-lg mt-2 text-gray-300">F1-Score</p>
          </div>
        </div>
      </div>

      {/* Cross-Validation Results Section (Using Line Chart for Accuracy over different targets) */}
      <div className="bg-gray-800 rounded-lg p-8 shadow-lg mb-12">
        <h2 className="text-3xl font-semibold mb-8 text-center text-orange-300">Accuracy by Learning Style Dimension</h2>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart
            data={Object.keys(modelStats).map(key => ({
              name: key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
              accuracy: (modelStats[key].accuracy * 100).toFixed(2)
            }))}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#4a5568" />
            <XAxis dataKey="name" stroke="#cbd5e0" />
            <YAxis stroke="#cbd5e0" domain={[0, 100]} />
            <Tooltip
              contentStyle={{ backgroundColor: '#2d3748', borderColor: '#4a5568', color: '#cbd5e0' }}
              labelStyle={{ color: '#a0aec0' }}
            />
            <Legend wrapperStyle={{ color: '#cbd5e0' }} />
            <Line type="monotone" dataKey="accuracy" stroke="#82ca9d" activeDot={{ r: 8 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Feature Importance Visualization (Using Bar Chart) */}
      <div className="bg-gray-800 rounded-lg p-8 shadow-lg mb-12">
        <h2 className="text-3xl font-semibold mb-8 text-center text-blue-300">Feature Importance (Overall)</h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart
            layout="vertical"
            data={Object.entries(featureImportances['ACTIVE_VS_REFLECTIVE']) // Default to first model for overall feature importance
              .sort(([, a], [, b]) => b - a)
              .slice(0, 10) // Display top 10 features
              .map(([name, value]) => ({ name: name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()), value: (value * 100).toFixed(2) }))}
            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#4a5568" />
            <XAxis type="number" stroke="#cbd5e0" />
            <YAxis type="category" dataKey="name" stroke="#cbd5e0" width={150} />
            <Tooltip
              contentStyle={{ backgroundColor: '#2d3748', borderColor: '#4a5568', color: '#cbd5e0' }}
              labelStyle={{ color: '#a0aec0' }}
            />
            <Legend wrapperStyle={{ color: '#cbd5e0' }} />
            <Bar dataKey="value" fill="#8884d8" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Confusion Matrix Heatmap */}
      <div className="bg-gray-800 rounded-lg p-8 shadow-lg mb-12">
        <h2 className="text-3xl font-semibold mb-8 text-center text-red-300">Confusion Matrix</h2>
        {confusionMatrices && Object.keys(confusionMatrices).length > 0 ? (
          <div>
            {Object.keys(confusionMatrices).map(target => (
              <div key={target} className="mb-8">
                <h3 className="text-xl font-semibold mb-4 text-gray-300">{target.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</h3>
                <div className="overflow-x-auto">
                  <table className="min-w-full bg-gray-700 rounded-lg overflow-hidden text-center">
                    <thead>
                      <tr className="bg-gray-600 text-gray-200 uppercase text-sm leading-normal">
                        <th className="py-3 px-4"></th>
                        {/* Assuming binary classification for simplicity, adjust for multi-class */}
                        <th className="py-3 px-4">Predicted 0</th>
                        <th className="py-3 px-4">Predicted 1</th>
                      </tr>
                    </thead>
                    <tbody className="text-gray-300 text-sm font-light">
                      <tr className="border-b border-gray-600">
                        <td className="py-3 px-4 font-medium">Actual 0</td>
                        <td className="py-3 px-4 bg-blue-800 text-white font-bold">{confusionMatrices[target][0][0]}</td>
                        <td className="py-3 px-4 bg-red-800 text-white font-bold">{confusionMatrices[target][0][1]}</td>
                      </tr>
                      <tr className="border-b border-gray-600">
                        <td className="py-3 px-4 font-medium">Actual 1</td>
                        <td className="py-3 px-4 bg-green-800 text-white font-bold">{confusionMatrices[target][1][0]}</td>
                        <td className="py-3 px-4 bg-blue-800 text-white font-bold">{confusionMatrices[target][1][1]}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="h-64 flex items-center justify-center text-gray-500 text-xl">
            No Confusion Matrix data available.
          </div>
        )}
      </div>

      {/* Model Comparison Section */}
      <div className="bg-gray-800 rounded-lg p-8 shadow-lg mb-12">
        <h2 className="text-3xl font-semibold mb-8 text-center text-green-300">Model Comparison</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {Object.keys(modelStats).map(target => (
            <div key={target} className="bg-gray-700 rounded-lg p-6 shadow-md transform transition duration-300 hover:scale-105 hover:shadow-xl">
              <h3 className="text-xl font-semibold mb-4 text-center text-green-200">{target.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</h3>
              <div className="space-y-2 text-sm">
                <p className="flex justify-between"><span>Accuracy:</span> <span className="font-medium text-green-400">{(modelStats[target].accuracy * 100).toFixed(1)}%</span></p>
                <p className="flex justify-between"><span>Precision:</span> <span className="font-medium text-blue-300">{(modelStats[target].precision * 100).toFixed(1)}%</span></p>
                <p className="flex justify-between"><span>Recall:</span> <span className="font-medium text-purple-300">{(modelStats[target].recall * 100).toFixed(1)}%</span></p>
                <p className="flex justify-between"><span>F1-Score:</span> <span className="font-medium text-pink-300">{(modelStats[target].f1_score * 100).toFixed(1)}%</span></p>
                <p className="flex justify-between"><span>Dataset Size:</span> <span className="font-medium text-gray-300">{modelStats[target].dataset_size}</span></p>
              </div>
            </div>
          ))}
        </div>
      </div>


      {/* Classification Report Table (Placeholder with enhanced styling) */}
      <div className="bg-gray-800 rounded-lg p-8 shadow-lg">
        <h2 className="text-3xl font-semibold mb-8 text-center text-yellow-300">Classification Report</h2>
        <div className="overflow-x-auto">
          {/* This table will need actual classification report data from the backend */}
          <table className="min-w-full bg-gray-700 rounded-lg overflow-hidden">
            <thead>
              <tr className="bg-gray-600 text-gray-200 uppercase text-sm leading-normal">
                <th className="py-3 px-6 text-left">Class</th>
                <th className="py-3 px-6 text-left">Precision</th>
                <th className="py-3 px-6 text-left">Recall</th>
                <th className="py-3 px-6 text-left">F1-Score</th>
                <th className="py-3 px-6 text-left">Support</th>
              </tr>
            </thead>
            <tbody className="text-gray-300 text-sm font-light">
              {/* Example row, replace with dynamic data */}
              {Object.keys(modelStats).map(target => (
                <tr key={target} className="border-b border-gray-600 hover:bg-gray-700">
                  <td className="py-3 px-6 text-left whitespace-nowrap">{target.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</td>
                  <td className="py-3 px-6 text-left">{(modelStats[target].precision).toFixed(2)}</td>
                  <td className="py-3 px-6 text-left">{(modelStats[target].recall).toFixed(2)}</td>
                  <td className="py-3 px-6 text-left">{(modelStats[target].f1_score).toFixed(2)}</td>
                  <td className="py-3 px-6 text-left">{modelStats[target].dataset_size}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}