"use client";

import React, { useState, useEffect, useRef } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar
} from 'recharts';
import { useSpring, animated } from '@react-spring/web';

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

  if (loading) return (
    <div className="min-h-screen bg-deep-space-navy text-white flex items-center justify-center p-4">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 w-full max-w-4xl">
        {[...Array(6)].map((_, index) => (
          <div key={index} className="glass-morphism p-3 rounded-card shadow-xl h-48 animate-pulse bg-charcoal-elevated/50"></div>
        ))}
      </div>
    </div>
  );

  if (error) return (
    <div className="min-h-screen bg-dark-navy text-white flex items-center justify-center">
      <p className="text-red-500">Error: {error.message}</p>
    </div>
  );

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

  const AnimatedNumber = ({ value }) => {
    const { number } = useSpring({
      from: { number: 0 },
      number: value,
      delay: 200,
      config: { mass: 1, tension: 20, friction: 10 }
    });
    return <animated.span>{number.to(n => n.toFixed(0))}</animated.span>;
  };

  const AnimatedPercentage = ({ value }) => {
    const { number } = useSpring({
      from: { number: 0 },
      number: value,
      delay: 200,
      config: { mass: 1, tension: 20, friction: 10 }
    });
    return <animated.span>{number.to(n => `${n.toFixed(1)}%`)}</animated.span>;
  };

  const rippleEffect = (event) => {
    const card = event.currentTarget;
    const diameter = Math.max(card.clientWidth, card.clientHeight);
    const radius = diameter / 2;
    const ripple = document.createElement('span');
    ripple.style.width = ripple.style.height = `${diameter}px`;
    ripple.style.left = `${event.clientX - (card.getBoundingClientRect().left + radius)}px`;
    ripple.style.top = `${event.clientY - (card.getBoundingClientRect().top + radius)}px`;
    ripple.classList.add('ripple');
    card.appendChild(ripple);
    ripple.addEventListener('animationend', () => {
      card.removeChild(ripple);
    });
  };

  const handleCardClick = (event) => {
    rippleEffect(event);
  };

  return (
    <div className="bg-deep-space-navy text-white p-2 flex flex-col overflow-y-auto min-h-screen">

      {/* Model Overview Section */}
      <div className="glass-morphism p-3 rounded-card shadow-xl mb-3 flex-shrink-0 border border-transparent hover:border-electric-purple hover:shadow-2xl transition-all duration-300 ease-in-out backdrop-filter backdrop-blur-md bg-opacity-10 bg-charcoal-elevated relative overflow-hidden group transition-all duration-300 ease-out glow-on-hover">
        <h2 className="text-xl font-bold mb-3 text-electric-purple">Model Overview</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {/* Overall Accuracy */}
          <div className="charcoal-elevated p-3 rounded-card shadow-lg transform transition-transform duration-300 ease-in-out hover:scale-102 cursor-pointer relative overflow-hidden group transition-all duration-300 ease-out glow-on-hover" onMouseDown={rippleEffect}>
            <h3 className="font-semibold mb-2 text-electric-purple">Overall Accuracy</h3>
            <div className="text-center">
              <p className="text-5xl font-extrabold text-emerald-success"><AnimatedPercentage value={parseFloat(formatPercentage(model.test_accuracy))} /></p>
              <span className="text-xs text-gray-400 mt-2">Excellent Performance</span>
            </div>
          </div>

          {/* Model Details */}
          <div className="charcoal-elevated p-3 rounded-card shadow-lg transform transition-transform duration-300 ease-in-out hover:scale-102 cursor-pointer relative overflow-hidden group transition-all duration-300 ease-out glow-on-hover" onMouseDown={rippleEffect}>
            <h3 className="font-semibold mb-2 text-electric-purple">Model Details</h3>
            <div className="space-y-1 text-sm">
              <p className="flex justify-between items-center"><span className="font-medium text-gray-300">Model Type:</span> <span className="text-emerald-success">{model.model_type}</span></p>
              <p className="flex justify-between items-center"><span className="font-medium text-gray-300">Last Trained:</span> <span className="text-emerald-success">{model.last_trained}</span></p>
              <p className="flex justify-between items-center"><span className="font-medium text-gray-300">Dataset Size:</span> <span className="text-emerald-success">{model.dataset_size} samples</span></p>
              <p className="flex justify-between items-center"><span className="font-medium text-gray-300">Features Used:</span> <span className="text-emerald-success">{model.features_used}</span></p>
            </div>
          </div>

          {/* Model Health Status */}
          <div className="charcoal-elevated p-3 rounded-card shadow-lg transform transition-transform duration-300 ease-in-out hover:scale-102 cursor-pointer relative overflow-hidden group transition-all duration-300 ease-out glow-on-hover" onMouseDown={rippleEffect}>
            <h3 className="font-semibold mb-2 text-electric-purple">Model Health</h3>
            <div className="flex flex-col items-center justify-center flex-1">
              <div className="relative w-24 h-24 flex items-center justify-center">
                <svg className="w-full h-full absolute">
                  <circle className="text-charcoal-elevated" strokeWidth="8" stroke="currentColor" fill="transparent" r="40" cx="50%" cy="50%" />
                  <circle className="text-emerald-success" strokeWidth="8" strokeDasharray="251.32" strokeDashoffset={calculateStrokeDashoffset(parseFloat(formatPercentage(model.test_accuracy, 0)))} strokeLinecap="round" stroke="url(#gradientAccuracy)" fill="transparent" r="40" cx="50%" cy="50%" transform="rotate(-90 48 48)" />
                  <defs>
                    <linearGradient id="gradientAccuracy" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#8B5CF6" />
                      <stop offset="100%" stopColor="#10B981" />
                    </linearGradient>
                  </defs>
                </svg>
                <div className="absolute w-full h-full flex items-center justify-center text-xl font-bold text-emerald-success">
                  <AnimatedPercentage value={parseFloat(formatPercentage(model.test_accuracy, 0))} />
                </div>
              </div>
              <p className="text-xs text-gray-400 mt-2">Operational & Stable</p>
            </div>
          </div>
        </div>
      </div>

      {/* Metrics Visualization Area */}
      <div className="glass-morphism p-3 rounded-card shadow-xl mb-3 flex-shrink-0 border border-transparent hover:border-electric-purple hover:shadow-2xl transition-all duration-300 ease-in-out backdrop-filter backdrop-blur-md bg-opacity-10 bg-charcoal-elevated relative overflow-hidden group" onMouseDown={handleCardClick}>
        <h2 className="text-xl font-bold mb-3 text-electric-purple">Performance Metrics</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
          {/* Gauge Chart for Accuracy */}
          <div className="flex flex-col items-center charcoal-elevated rounded-card p-2 shadow-lg transform transition-transform duration-300 ease-in-out hover:scale-102 cursor-pointer relative overflow-hidden group transition-all duration-300 ease-out glow-on-hover" onMouseDown={rippleEffect}>
            <div className="relative w-24 h-24 flex items-center justify-center">
              <svg className="w-full h-full absolute">
                <circle className="text-deep-space-navy" strokeWidth="8" stroke="currentColor" fill="transparent" r="40" cx="50%" cy="50%" />
                <circle className="text-emerald-success" strokeWidth="8" strokeDasharray="251.32" strokeDashoffset={calculateStrokeDashoffset(parseFloat(formatPercentage(model.test_accuracy, 0)))} strokeLinecap="round" stroke="url(#gradientAccuracy)" fill="transparent" r="40" cx="50%" cy="50%" transform="rotate(-90 48 48)" />
                <defs>
                  <linearGradient id="gradientAccuracy" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#8B5CF6" />
                    <stop offset="100%" stopColor="#10B981" />
                  </linearGradient>
                </defs>
              </svg>
              <div className="absolute w-full h-full flex items-center justify-center text-xl font-bold text-emerald-success">
                <AnimatedPercentage value={parseFloat(formatPercentage(model.test_accuracy))} />
              </div>
            </div>
            <p className="text-sm mt-1 text-gray-300">Accuracy</p>
          </div>

          {/* Gauge Chart for Precision */}
          <div className="flex flex-col items-center charcoal-elevated rounded-card p-2 shadow-lg transform transition-transform duration-300 ease-in-out hover:scale-102 cursor-pointer relative overflow-hidden group transition-all duration-300 ease-in-out glow-on-hover" onMouseDown={rippleEffect}>
            <div className="relative w-24 h-24 flex items-center justify-center">
              <svg className="w-full h-full absolute">
                <circle className="text-deep-space-navy" strokeWidth="8" stroke="currentColor" fill="transparent" r="40" cx="50%" cy="50%" />
                <circle className="text-electric-purple" strokeWidth="8" strokeDasharray="251.32" strokeDashoffset={calculateStrokeDashoffset(parseFloat(model.precision || 0) * 100)} strokeLinecap="round" stroke="url(#gradientPurple)" fill="transparent" r="40" cx="50%" cy="50%" transform="rotate(-90 48 48)" />
                <defs>
                  <linearGradient id="gradientPurple" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#8B5CF6" />
                    <stop offset="100%" stopColor="#A78BFA" />
                  </linearGradient>
                </defs>
              </svg>
              <div className="absolute w-full h-full flex items-center justify-center text-xl font-bold text-electric-purple">
                <AnimatedPercentage value={parseFloat(formatPercentage(model.precision))} />
              </div>
            </div>
            <p className="text-sm mt-1 text-gray-300">Precision</p>
          </div>

          {/* Gauge Chart for Recall */}
          <div className="flex flex-col items-center charcoal-elevated rounded-card p-2 shadow-lg transform transition-transform duration-300 ease-in-out hover:scale-102 cursor-pointer relative overflow-hidden group transition-all duration-300 ease-in-out glow-on-hover" onMouseDown={rippleEffect}>
            <div className="relative w-24 h-24 flex items-center justify-center">
              <svg className="w-full h-full absolute">
                <circle className="text-deep-space-navy" strokeWidth="8" stroke="currentColor" fill="transparent" r="40" cx="50%" cy="50%" />
                <circle className="text-electric-purple" strokeWidth="8" strokeDasharray="251.32" strokeDashoffset={calculateStrokeDashoffset(parseFloat(model.recall || 0) * 100)} strokeLinecap="round" stroke="url(#gradientPurple)" fill="transparent" r="40" cx="50%" cy="50%" transform="rotate(-90 48 48)" />
                <defs>
                  <linearGradient id="gradientPurple" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#8B5CF6" />
                    <stop offset="100%" stopColor="#A78BFA" />
                  </linearGradient>
                </defs>
              </svg>
              <div className="absolute w-full h-full flex items-center justify-center text-xl font-bold text-electric-purple">
                <AnimatedPercentage value={parseFloat(formatPercentage(model.recall))} />
              </div>
            </div>
            <p className="text-sm mt-1 text-gray-300">Recall</p>
          </div>

          {/* Gauge Chart for F1-Score */}
          <div className="flex flex-col items-center charcoal-elevated rounded-card p-2 shadow-lg transform transition-transform duration-300 ease-in-out hover:scale-102 cursor-pointer relative overflow-hidden group transition-all duration-300 ease-in-out glow-on-hover" onMouseDown={rippleEffect}>
            <div className="relative w-24 h-24 flex items-center justify-center">
              <svg className="w-full h-full absolute">
                <circle className="text-deep-space-navy" strokeWidth="8" stroke="currentColor" fill="transparent" r="40" cx="50%" cy="50%" />
                <circle className="text-electric-purple" strokeWidth="8" strokeDasharray="251.32" strokeDashoffset={calculateStrokeDashoffset(parseFloat(model.f1_score || 0) * 100)} strokeLinecap="round" stroke="url(#gradientPurple)" fill="transparent" r="40" cx="50%" cy="50%" transform="rotate(-90 48 48)" />
                <defs>
                  <linearGradient id="gradientPurple" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#8B5CF6" />
                    <stop offset="100%" stopColor="#A78BFA" />
                  </linearGradient>
                </defs>
              </svg>
              <div className="absolute w-full h-full flex items-center justify-center text-xl font-bold text-electric-purple">
                <AnimatedPercentage value={parseFloat(formatPercentage(model.f1_score))} />
              </div>
            </div>
            <p className="text-sm mt-1 text-gray-300">F1-Score</p>
          </div>
        </div>
      </div>

      {/* Cross-Validation Results Section (Using Line Chart for Accuracy over different targets) */}
      <div className="glass-morphism p-3 rounded-card shadow-xl mb-3 flex-shrink-0 border border-transparent hover:border-electric-purple hover:shadow-2xl transition-all duration-300 ease-in-out backdrop-filter backdrop-blur-md bg-opacity-10 bg-charcoal-elevated relative overflow-hidden group" onMouseDown={handleCardClick}>
        <h2 className="text-xl font-bold mb-3 text-electric-purple">Accuracy by Learning Style Dimension</h2>
        <div className="h-[500px]"> {/* Explicit height for the chart container */}
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={Object.keys(modelStats).map(key => ({
                name: key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
                accuracy: parseFloat(formatPercentage(modelStats[key].test_accuracy, 2)) || 0
              }))}
              margin={{ top: 5, right: 30, left: 20, bottom: 60 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#2D333B" />
              <XAxis dataKey="name" stroke="#9CA3AF" interval="equidistantPreserveStart" angle={-30} textAnchor="end" height={80} />
              <YAxis stroke="#9CA3AF" domain={[0, 100]} />
              <Tooltip
                contentStyle={{ backgroundColor: '#161B22', borderColor: '#2D333B', color: '#E5E7EB' }}
                labelStyle={{ color: '#9CA3AF' }}
              />
              <Legend wrapperStyle={{ color: '#E5E7EB' }} />
              <defs>
                <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#8B5CF6" />
                  <stop offset="100%" stopColor="#6C2BD9" />
                </linearGradient>
              </defs>
              <Line type="monotone" dataKey="accuracy" stroke="url(#lineGradient)" activeDot={{ r: 6 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Feature Importance Visualization (Using Bar Chart) */}
      <div className="glass-morphism p-3 rounded-card shadow-xl mb-3 flex-shrink-0 border border-transparent hover:border-electric-purple hover:shadow-2xl transition-all duration-300 ease-in-out backdrop-filter backdrop-blur-md bg-opacity-10 bg-charcoal-elevated relative overflow-hidden group" onMouseDown={handleCardClick}>
        <h2 className="text-xl font-bold mb-3 text-electric-purple">Feature Importance (Overall)</h2>
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
              <defs>
                <linearGradient id="barGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor="#8B5CF6" />
                  <stop offset="100%" stopColor="#6C2BD9" />
                </linearGradient>
              </defs>
              <Bar dataKey="value" fill="url(#barGradient)" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Confusion Matrix Heatmap */}
      <div className="glass-morphism p-3 rounded-card shadow-xl mb-3 flex-shrink-0 border border-transparent hover:border-electric-purple hover:shadow-2xl transition-all duration-300 ease-in-out backdrop-filter backdrop-blur-md bg-opacity-10 bg-charcoal-elevated relative overflow-hidden group" onMouseDown={handleCardClick}>
        <h2 className="text-xl font-bold mb-3 text-electric-purple">Confusion Matrix</h2>
        {confusionMatrices && Object.keys(confusionMatrices).length > 0 ? (
          <div className="overflow-x-auto overflow-y-auto relative rounded-card border border-transparent" style={{ maxHeight: '400px' }}>
            {Object.keys(confusionMatrices).map(target => (
              <div key={target} className="mb-4 charcoal-elevated p-3 rounded-card shadow-lg border border-transparent hover:shadow-xl transition-all duration-300 ease-in-out group relative overflow-hidden transition-all duration-300 ease-out glow-on-hover" onMouseDown={rippleEffect}>
                <h3 className="text-md font-semibold mb-2 text-gray-300">{target.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</h3>
                <table className="w-full text-left table-auto table-compact">
                  <thead className="bg-charcoal-elevated sticky top-0 border-b border-transparent">
                    <tr>
                      <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider rounded-tl-card"></th>
                      <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">Predicted 0</th>
                      <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider rounded-tr-card">Predicted 1</th>
                    </tr>
                  </thead>
                  <tbody className="text-gray-300 text-sm font-light">
                    <tr className="border-b border-transparent bg-charcoal-elevated transition-all duration-200 ease-in-out">
                      <td className="px-2 py-1 font-medium">Actual 0</td>
                      <td className="px-2 py-1 bg-charcoal-elevated text-emerald-success font-bold">{confusionMatrices[target].confusion_matrix[0][0]}</td>
                      <td className="px-2 py-1 bg-charcoal-elevated text-rose-danger font-bold">{confusionMatrices[target].confusion_matrix[0][1]}</td>
                    </tr>
                    <tr className="border-b border-transparent bg-charcoal-elevated transition-all duration-200 ease-in-out">
                      <td className="px-2 py-1 font-medium">Actual 1</td>
                      <td className="px-2 py-1 bg-charcoal-elevated text-emerald-success font-bold">{confusionMatrices[target].confusion_matrix[1][0]}</td>
                      <td className="px-2 py-1 bg-charcoal-elevated text-emerald-success font-bold">{confusionMatrices[target].confusion_matrix[1][1]}</td>
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
      <div className="glass-morphism p-3 rounded-card shadow-xl mb-3 flex-shrink-0 border border-transparent hover:border-electric-purple hover:shadow-2xl transition-all duration-300 ease-in-out backdrop-filter backdrop-blur-md bg-opacity-10 bg-charcoal-elevated relative overflow-hidden group" onMouseDown={handleCardClick}>
        <h2 className="text-xl font-bold mb-3 text-electric-purple">Model Comparison</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
          {Object.keys(modelStats).map(target => (
            <div key={target} className="charcoal-elevated rounded-card p-2 shadow-lg transform transition-transform duration-300 ease-in-out hover:scale-102 cursor-pointer relative overflow-hidden group transition-all duration-300 ease-in-out glow-on-hover" onMouseDown={rippleEffect}>
              <h3 className="text-md font-semibold mb-2 text-center text-electric-purple">{target.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</h3>
              <div className="space-y-1 text-xs">
                <p className="flex justify-between"><span>Accuracy:</span> <span className="font-medium text-emerald-success">{formatPercentage(modelStats[target].test_accuracy)}%</span></p>
                <p className="flex justify-between"><span>Precision:</span> <span className="font-medium text-emerald-success">{formatPercentage(modelStats[target].precision)}%</span></p>
                <p className="flex justify-between"><span>Recall:</span> <span className="font-medium text-emerald-success">{formatPercentage(modelStats[target].recall)}%</span></p>
                <p className="flex justify-between"><span>F1-Score:</span> <span className="font-medium text-emerald-success">{formatPercentage(modelStats[target].f1_score)}%</span></p>
                <p className="flex justify-between"><span>Dataset Size:</span> <span className="font-medium text-gray-300">{modelStats[target].dataset_size}</span></p>
              </div>
            </div>
          ))}
        </div>
      </div>


      {/* Classification Report Table (Placeholder with enhanced styling) */}
      <div className="glass-morphism p-3 rounded-card shadow-xl mb-3 flex-shrink-0 border border-transparent hover:border-electric-purple hover:shadow-2xl transition-all duration-300 ease-in-out backdrop-filter backdrop-blur-md bg-opacity-10 bg-charcoal-elevated relative overflow-hidden group" onMouseDown={handleCardClick}>
        <h2 className="text-xl font-bold mb-3 text-electric-purple">Classification Report</h2>
        <div className="overflow-x-auto overflow-y-auto relative rounded-card border border-transparent" style={{ maxHeight: '400px' }}>
          <table className="w-full text-left table-auto table-compact">
            <thead className="bg-charcoal-elevated sticky top-0 border-b border-transparent">
              <tr>
                <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider rounded-tl-card">Class</th>
                <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">Precision</th>
                <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">Recall</th>
                <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">F1-Score</th>
                <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">Support</th>
              </tr>
            </thead>
            <tbody className="text-gray-300 text-sm font-light">
              {Object.keys(modelStats).map((target, index) => (
                <tr key={target} className={`border-b border-transparent ${index % 2 === 0 ? 'bg-charcoal-elevated' : 'bg-charcoal-elevated/50'} hover:bg-charcoal-elevated/75 transition-all duration-200 ease-in-out`}>
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
  );

  return (
    <div className="bg-deep-space-navy text-white p-2 flex flex-col overflow-y-auto min-h-screen">

      {/* Model Overview Section */}
      <div className="glass-morphism p-3 rounded-card shadow-xl mb-3 flex-shrink-0 border border-transparent hover:border-electric-purple hover:shadow-2xl transition-all duration-300 ease-in-out backdrop-filter backdrop-blur-md bg-opacity-10 bg-charcoal-elevated relative overflow-hidden group transition-all duration-300 ease-out glow-on-hover">
        <h2 className="text-xl font-bold mb-3 text-electric-purple">Model Overview</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {/* Overall Accuracy */}
          <div className="charcoal-elevated p-3 rounded-card shadow-lg transform transition-transform duration-300 ease-in-out hover:scale-102 cursor-pointer relative overflow-hidden group transition-all duration-300 ease-out glow-on-hover" onMouseDown={rippleEffect}>
            <h3 className="font-semibold mb-2 text-electric-purple">Overall Accuracy</h3>
            <div className="text-center">
              <p className="text-5xl font-extrabold text-emerald-success"><AnimatedPercentage value={parseFloat(formatPercentage(model.test_accuracy))} /></p>
              <span className="text-xs text-gray-400 mt-2">Excellent Performance</span>
            </div>
          </div>

          {/* Model Details */}
          <div className="charcoal-elevated p-3 rounded-card shadow-lg transform transition-transform duration-300 ease-in-out hover:scale-102 cursor-pointer relative overflow-hidden group transition-all duration-300 ease-out glow-on-hover" onMouseDown={rippleEffect}>
            <h3 className="font-semibold mb-2 text-electric-purple">Model Details</h3>
            <div className="space-y-1 text-sm">
              <p className="flex justify-between items-center"><span className="font-medium text-gray-300">Model Type:</span> <span className="text-emerald-success">{model.model_type}</span></p>
              <p className="flex justify-between items-center"><span className="font-medium text-gray-300">Last Trained:</span> <span className="text-emerald-success">{model.last_trained}</span></p>
              <p className="flex justify-between items-center"><span className="font-medium text-gray-300">Dataset Size:</span> <span className="text-emerald-success">{model.dataset_size} samples</span></p>
              <p className="flex justify-between items-center"><span className="font-medium text-gray-300">Features Used:</span> <span className="text-emerald-success">{model.features_used}</span></p>
            </div>
          </div>

          {/* Model Health Status */}
          <div className="charcoal-elevated p-3 rounded-card shadow-lg transform transition-transform duration-300 ease-in-out hover:scale-102 cursor-pointer relative overflow-hidden group transition-all duration-300 ease-out glow-on-hover" onMouseDown={rippleEffect}>
            <h3 className="font-semibold mb-2 text-electric-purple">Model Health</h3>
            <div className="flex flex-col items-center justify-center flex-1">
              <div className="relative w-24 h-24 flex items-center justify-center">
                <svg className="w-full h-full absolute">
                  <circle className="text-charcoal-elevated" strokeWidth="8" stroke="currentColor" fill="transparent" r="40" cx="50%" cy="50%" />
                  <circle className="text-emerald-success" strokeWidth="8" strokeDasharray="251.32" strokeDashoffset={calculateStrokeDashoffset(parseFloat(formatPercentage(model.test_accuracy, 0)))} strokeLinecap="round" stroke="url(#gradientAccuracy)" fill="transparent" r="40" cx="50%" cy="50%" transform="rotate(-90 48 48)" />
                  <defs>
                    <linearGradient id="gradientAccuracy" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#8B5CF6" />
                      <stop offset="100%" stopColor="#10B981" />
                    </linearGradient>
                  </defs>
                </svg>
                <div className="absolute w-full h-full flex items-center justify-center text-xl font-bold text-emerald-success">
                  <AnimatedPercentage value={parseFloat(formatPercentage(model.test_accuracy, 0))} />
                </div>
              </div>
              <p className="text-xs text-gray-400 mt-2">Operational & Stable</p>
            </div>
          </div>
        </div>
      </div>

      {/* Metrics Visualization Area */}
      <div className="glass-morphism p-3 rounded-card shadow-xl mb-3 flex-shrink-0 border border-transparent hover:border-electric-purple hover:shadow-2xl transition-all duration-300 ease-in-out backdrop-filter backdrop-blur-md bg-opacity-10 bg-charcoal-elevated relative overflow-hidden group" onMouseDown={handleCardClick}>
        <h2 className="text-xl font-bold mb-3 text-electric-purple">Performance Metrics</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
          {/* Gauge Chart for Accuracy */}
          <div className="flex flex-col items-center charcoal-elevated rounded-card p-2 shadow-lg transform transition-transform duration-300 ease-in-out hover:scale-102 cursor-pointer relative overflow-hidden group transition-all duration-300 ease-out glow-on-hover" onMouseDown={rippleEffect}>
            <div className="relative w-24 h-24 flex items-center justify-center">
              <svg className="w-full h-full absolute">
                <circle className="text-deep-space-navy" strokeWidth="8" stroke="currentColor" fill="transparent" r="40" cx="50%" cy="50%" />
                <circle className="text-emerald-success" strokeWidth="8" strokeDasharray="251.32" strokeDashoffset={calculateStrokeDashoffset(parseFloat(formatPercentage(model.test_accuracy, 0)))} strokeLinecap="round" stroke="url(#gradientAccuracy)" fill="transparent" r="40" cx="50%" cy="50%" transform="rotate(-90 48 48)" />
                <defs>
                  <linearGradient id="gradientAccuracy" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#8B5CF6" />
                    <stop offset="100%" stopColor="#10B981" />
                  </linearGradient>
                </defs>
              </svg>
              <div className="absolute w-full h-full flex items-center justify-center text-xl font-bold text-emerald-success">
                <AnimatedPercentage value={parseFloat(formatPercentage(model.test_accuracy))} />
              </div>
            </div>
            <p className="text-sm mt-1 text-gray-300">Accuracy</p>
          </div>

          {/* Gauge Chart for Precision */}
          <div className="flex flex-col items-center charcoal-elevated rounded-card p-2 shadow-lg transform transition-transform duration-300 ease-in-out hover:scale-102 cursor-pointer relative overflow-hidden group transition-all duration-300 ease-in-out glow-on-hover" onMouseDown={rippleEffect}>
            <div className="relative w-24 h-24 flex items-center justify-center">
              <svg className="w-full h-full absolute">
                <circle className="text-deep-space-navy" strokeWidth="8" stroke="currentColor" fill="transparent" r="40" cx="50%" cy="50%" />
                <circle className="text-electric-purple" strokeWidth="8" strokeDasharray="251.32" strokeDashoffset={calculateStrokeDashoffset(parseFloat(model.precision || 0) * 100)} strokeLinecap="round" stroke="url(#gradientPurple)" fill="transparent" r="40" cx="50%" cy="50%" transform="rotate(-90 48 48)" />
                <defs>
                  <linearGradient id="gradientPurple" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#8B5CF6" />
                    <stop offset="100%" stopColor="#A78BFA" />
                  </linearGradient>
                </defs>
              </svg>
              <div className="absolute w-full h-full flex items-center justify-center text-xl font-bold text-electric-purple">
                <AnimatedPercentage value={parseFloat(formatPercentage(model.precision))} />
              </div>
            </div>
            <p className="text-sm mt-1 text-gray-300">Precision</p>
          </div>

          {/* Gauge Chart for Recall */}
          <div className="flex flex-col items-center charcoal-elevated rounded-card p-2 shadow-lg transform transition-transform duration-300 ease-in-out hover:scale-102 cursor-pointer relative overflow-hidden group transition-all duration-300 ease-in-out glow-on-hover" onMouseDown={rippleEffect}>
            <div className="relative w-24 h-24 flex items-center justify-center">
              <svg className="w-full h-full absolute">
                <circle className="text-deep-space-navy" strokeWidth="8" stroke="currentColor" fill="transparent" r="40" cx="50%" cy="50%" />
                <circle className="text-electric-purple" strokeWidth="8" strokeDasharray="251.32" strokeDashoffset={calculateStrokeDashoffset(parseFloat(model.recall || 0) * 100)} strokeLinecap="round" stroke="url(#gradientPurple)" fill="transparent" r="40" cx="50%" cy="50%" transform="rotate(-90 48 48)" />
                <defs>
                  <linearGradient id="gradientPurple" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#8B5CF6" />
                    <stop offset="100%" stopColor="#A78BFA" />
                  </linearGradient>
                </defs>
              </svg>
              <div className="absolute w-full h-full flex items-center justify-center text-xl font-bold text-electric-purple">
                <AnimatedPercentage value={parseFloat(formatPercentage(model.recall))} />
              </div>
            </div>
            <p className="text-sm mt-1 text-gray-300">Recall</p>
          </div>

          {/* Gauge Chart for F1-Score */}
          <div className="flex flex-col items-center charcoal-elevated rounded-card p-2 shadow-lg transform transition-transform duration-300 ease-in-out hover:scale-102 cursor-pointer relative overflow-hidden group transition-all duration-300 ease-in-out glow-on-hover" onMouseDown={rippleEffect}>
            <div className="relative w-24 h-24 flex items-center justify-center">
              <svg className="w-full h-full absolute">
                <circle className="text-deep-space-navy" strokeWidth="8" stroke="currentColor" fill="transparent" r="40" cx="50%" cy="50%" />
                <circle className="text-electric-purple" strokeWidth="8" strokeDasharray="251.32" strokeDashoffset={calculateStrokeDashoffset(parseFloat(model.f1_score || 0) * 100)} strokeLinecap="round" stroke="url(#gradientPurple)" fill="transparent" r="40" cx="50%" cy="50%" transform="rotate(-90 48 48)" />
                <defs>
                  <linearGradient id="gradientPurple" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#8B5CF6" />
                    <stop offset="100%" stopColor="#A78BFA" />
                  </linearGradient>
                </defs>
              </svg>
              <div className="absolute w-full h-full flex items-center justify-center text-xl font-bold text-electric-purple">
                <AnimatedPercentage value={parseFloat(formatPercentage(model.f1_score))} />
              </div>
            </div>
            <p className="text-sm mt-1 text-gray-300">F1-Score</p>
          </div>
        </div>
      </div>

      {/* Cross-Validation Results Section (Using Line Chart for Accuracy over different targets) */}
      <div className="glass-morphism p-3 rounded-card shadow-xl mb-3 flex-shrink-0 border border-transparent hover:border-electric-purple hover:shadow-2xl transition-all duration-300 ease-in-out backdrop-filter backdrop-blur-md bg-opacity-10 bg-charcoal-elevated relative overflow-hidden group" onMouseDown={handleCardClick}>
        <h2 className="text-xl font-bold mb-3 text-electric-purple">Accuracy by Learning Style Dimension</h2>
        <div className="h-[350px]"> {/* Explicit height for the chart container */}
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={Object.keys(modelStats).map(key => ({
                name: key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
                accuracy: parseFloat(formatPercentage(modelStats[key].test_accuracy, 2)) || 0
              }))}
              margin={{ top: 5, right: 30, left: 20, bottom: 60 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#2D333B" />
              <XAxis dataKey="name" stroke="#9CA3AF" interval="equidistantPreserveStart" angle={-30} textAnchor="end" height={80} />
              <YAxis stroke="#9CA3AF" domain={[0, 100]} />
              <Tooltip
                contentStyle={{ backgroundColor: '#161B22', borderColor: '#2D333B', color: '#E5E7EB' }}
                labelStyle={{ color: '#9CA3AF' }}
              />
              <Legend wrapperStyle={{ color: '#E5E7EB' }} />
              <defs>
                <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#8B5CF6" />
                  <stop offset="100%" stopColor="#6C2BD9" />
                </linearGradient>
              </defs>
              <Line type="monotone" dataKey="accuracy" stroke="url(#lineGradient)" activeDot={{ r: 6 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Feature Importance Visualization (Using Bar Chart) */}
      <div className="glass-morphism p-3 rounded-card shadow-xl mb-3 flex-shrink-0 border border-transparent hover:border-electric-purple hover:shadow-2xl transition-all duration-300 ease-in-out backdrop-filter backdrop-blur-md bg-opacity-10 bg-charcoal-elevated relative overflow-hidden group" onMouseDown={handleCardClick}>
        <h2 className="text-xl font-bold mb-3 text-electric-purple">Feature Importance (Overall)</h2>
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
              <defs>
                <linearGradient id="barGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor="#8B5CF6" />
                  <stop offset="100%" stopColor="#6C2BD9" />
                </linearGradient>
              </defs>
              <Bar dataKey="value" fill="url(#barGradient)" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Confusion Matrix Heatmap */}
      <div className="glass-morphism p-3 rounded-card shadow-xl mb-3 flex-shrink-0 border border-transparent hover:border-electric-purple hover:shadow-2xl transition-all duration-300 ease-in-out backdrop-filter backdrop-blur-md bg-opacity-10 bg-charcoal-elevated relative overflow-hidden group" onMouseDown={handleCardClick}>
        <h2 className="text-xl font-bold mb-3 text-electric-purple">Confusion Matrix</h2>
        {confusionMatrices && Object.keys(confusionMatrices).length > 0 ? (
          <div className="overflow-x-auto overflow-y-auto relative rounded-card border border-transparent" style={{ maxHeight: '400px' }}>
            {Object.keys(confusionMatrices).map(target => (
              <div key={target} className="mb-4 charcoal-elevated p-3 rounded-card shadow-lg border border-transparent hover:shadow-xl transition-all duration-300 ease-in-out group relative overflow-hidden transition-all duration-300 ease-out glow-on-hover" onMouseDown={rippleEffect}>
                <h3 className="text-md font-semibold mb-2 text-gray-300">{target.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</h3>
                <table className="w-full text-left table-auto table-compact">
                  <thead className="bg-charcoal-elevated sticky top-0 border-b border-transparent">
                    <tr>
                      <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider rounded-tl-card"></th>
                      <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">Predicted 0</th>
                      <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider rounded-tr-card">Predicted 1</th>
                    </tr>
                  </thead>
                  <tbody className="text-gray-300 text-sm font-light">
                    <tr className="border-b border-transparent ${index % 2 === 0 ? 'bg-charcoal-elevated' : 'bg-charcoal-elevated/50'} hover:bg-charcoal-elevated/75 transition-all duration-200 ease-in-out">
                      <td className="px-2 py-1 font-medium bg-charcoal-elevated">Actual 0</td>
                      <td className="px-2 py-1 bg-charcoal-elevated text-emerald-success font-bold">{confusionMatrices[target].confusion_matrix[0][0]}</td>
                      <td className="px-2 py-1 bg-charcoal-elevated text-rose-danger font-bold">{confusionMatrices[target].confusion_matrix[0][1]}</td>
                    </tr>
                    <tr className="border-b border-transparent bg-charcoal-elevated transition-all duration-200 ease-in-out">
                      <td className="px-2 py-1 font-medium bg-charcoal-elevated">Actual 1</td>
                      <td className="px-2 py-1 bg-charcoal-elevated text-emerald-success font-bold">{confusionMatrices[target].confusion_matrix[1][0]}</td>
                      <td className="px-2 py-1 bg-charcoal-elevated text-emerald-success font-bold">{confusionMatrices[target].confusion_matrix[1][1]}</td>
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
      <div className="glass-morphism p-3 rounded-card shadow-xl mb-3 flex-shrink-0 border border-transparent hover:border-electric-purple hover:shadow-2xl transition-all duration-300 ease-in-out backdrop-filter backdrop-blur-md bg-opacity-10 bg-charcoal-elevated relative overflow-hidden group" onMouseDown={handleCardClick}>
        <h2 className="text-xl font-bold mb-3 text-electric-purple">Model Comparison</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
          {Object.keys(modelStats).map(target => (
            <div key={target} className="charcoal-elevated rounded-card p-2 shadow-lg transform transition-transform duration-300 ease-in-out hover:scale-102 cursor-pointer relative overflow-hidden group transition-all duration-300 ease-in-out glow-on-hover" onMouseDown={rippleEffect}>
              <h3 className="text-md font-semibold mb-2 text-center text-electric-purple">{target.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</h3>
              <div className="space-y-1 text-xs">
                <p className="flex justify-between"><span>Accuracy:</span> <span className="font-medium text-emerald-success">{formatPercentage(modelStats[target].test_accuracy)}%</span></p>
                <p className="flex justify-between"><span>Precision:</span> <span className="font-medium text-emerald-success">{formatPercentage(modelStats[target].precision)}%</span></p>
                <p className="flex justify-between"><span>Recall:</span> <span className="font-medium text-emerald-success">{formatPercentage(modelStats[target].recall)}%</span></p>
                <p className="flex justify-between"><span>F1-Score:</span> <span className="font-medium text-emerald-success">{formatPercentage(modelStats[target].f1_score)}%</span></p>
                <p className="flex justify-between"><span>Dataset Size:</span> <span className="font-medium text-gray-300">{modelStats[target].dataset_size}</span></p>
              </div>
            </div>
          ))}
        </div>
      </div>


      {/* Classification Report Table (Placeholder with enhanced styling) */}
      <div className="glass-morphism p-3 rounded-card shadow-xl mb-3 flex-shrink-0 border border-transparent hover:border-electric-purple hover:shadow-2xl transition-all duration-300 ease-in-out backdrop-filter backdrop-blur-md bg-opacity-10 bg-charcoal-elevated relative overflow-hidden group" onMouseDown={handleCardClick}>
        <h2 className="text-xl font-bold mb-3 text-electric-purple">Classification Report</h2>
        <div className="overflow-x-auto overflow-y-auto relative rounded-card border border-transparent" style={{ maxHeight: '400px' }}>
          <table className="w-full text-left table-auto table-compact">
            <thead className="bg-charcoal-elevated sticky top-0 border-b border-transparent">
              <tr>
                <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider rounded-tl-card">Class</th>
                <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">Precision</th>
                <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">Recall</th>
                <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">F1-Score</th>
                <th className="px-2 py-1 text-left text-xs font-bold text-gray-200 uppercase tracking-wider">Support</th>
              </tr>
            </thead>
            <tbody className="text-gray-300 text-sm font-light">
              {Object.keys(modelStats).map((target, index) => (
                <tr key={target} className={`border-b border-transparent ${index % 2 === 0 ? 'bg-charcoal-elevated' : 'bg-charcoal-elevated/50'} hover:bg-charcoal-elevated/75 transition-all duration-200 ease-in-out`}>
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
  );
}