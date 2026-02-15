"use client";

import { useState, useEffect } from "react";
import MetricCard from "./MetricCard";
import Chart from "./Chart";

interface TrainingMetrics {
  step: number;
  episode: number;
  tps: number;
  vram_mb: number;
  ram_mb: number;
  loss: number;
  reward: number;
  elapsed_time: number;
  predicted_total_time: number;
}

interface DashboardProps {
  metrics: TrainingMetrics | null;
  isConnected: boolean;
  isTraining: boolean;
  setIsTraining: (value: boolean) => void;
  error: string | null;
}

export default function Dashboard({
  metrics,
  isConnected,
  isTraining,
  setIsTraining,
  error,
}: DashboardProps) {
  const [history, setHistory] = useState<TrainingMetrics[]>([]);

  const handleStart = async () => {
    try {
      await fetch("http://localhost:8000/api/init", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ total_days: 200, steps_per_day: 100 }),
      });

      await fetch("http://localhost:8000/api/start", {
        method: "POST",
      });

      setIsTraining(true);
      setHistory([]);
    } catch (e) {
      console.error("Failed to start training:", e);
    }
  };

  const handleStop = async () => {
    try {
      await fetch("http://localhost:8000/api/stop", { method: "POST" });
      setIsTraining(false);
    } catch (e) {
      console.error("Failed to stop training:", e);
    }
  };

  useEffect(() => {
    if (metrics) {
      setHistory((prev) => {
        const newHistory = [...prev, metrics];
        return newHistory.slice(-100);
      });
    }
  }, [metrics]);

  return (
    <div className="container mx-auto px-4 py-8">
      <header className="mb-8">
        <h1 className="text-4xl font-bold text-white mb-2">
          RL Training Benchmark
        </h1>
        <p className="text-slate-400">
          Mac M4 환경에서 0.1B 파라미터 모델 학습 성능 측정
        </p>
      </header>

      {error && (
        <div className="mb-6 p-4 bg-red-500/10 border border-red-500 rounded-lg text-red-400">
          {error}
        </div>
      )}

      <div className="flex gap-4 mb-8">
        <button
          onClick={handleStart}
          disabled={isTraining}
          className="px-6 py-3 bg-green-600 hover:bg-green-700 disabled:bg-green-800 disabled:cursor-not-allowed text-white font-semibold rounded-lg transition-colors"
        >
          Start Training
        </button>
        <button
          onClick={handleStop}
          disabled={!isTraining}
          className="px-6 py-3 bg-red-600 hover:bg-red-700 disabled:bg-red-800 disabled:cursor-not-allowed text-white font-semibold rounded-lg transition-colors"
        >
          Stop Training
        </button>
      </div>

      <div className="mb-4">
        <div className="flex items-center gap-2">
          <div
            className={`w-3 h-3 rounded-full ${
              isConnected ? "bg-green-500" : "bg-red-500"
            }`}
          />
          <span className="text-slate-400">
            {isConnected ? "Connected" : "Disconnected"}
          </span>
          {isTraining && (
            <span className="ml-4 text-blue-400 font-semibold animate-pulse">
              Training in progress...
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <MetricCard
          title="TPS"
          value={metrics?.tps.toFixed(2) || "0.00"}
          unit="steps/sec"
          icon="⚡"
          color="yellow"
        />
        <MetricCard
          title="VRAM Usage"
          value={metrics?.vram_mb.toFixed(1) || "0.0"}
          unit="MB"
          icon="💾"
          color="blue"
        />
        <MetricCard
          title="RAM Usage"
          value={metrics?.ram_mb.toFixed(1) || "0.0"}
          unit="MB"
          icon="🧠"
          color="purple"
        />
        <MetricCard
          title="Loss"
          value={metrics?.loss.toFixed(4) || "0.0000"}
          unit=""
          icon="📉"
          color="red"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-6">
          <h3 className="text-xl font-semibold text-white mb-4">TPS History</h3>
          <Chart data={history} dataKey="tps" color="#eab308" />
        </div>
        <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-6">
          <h3 className="text-xl font-semibold text-white mb-4">Loss History</h3>
          <Chart data={history} dataKey="loss" color="#ef4444" />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-6">
          <h3 className="text-xl font-semibold text-white mb-4">VRAM History</h3>
          <Chart data={history} dataKey="vram_mb" color="#3b82f6" />
        </div>
        <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-6">
          <h3 className="text-xl font-semibold text-white mb-4">RAM History</h3>
          <Chart data={history} dataKey="ram_mb" color="#a855f7" />
        </div>
      </div>

      {metrics && (
        <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-6">
          <h3 className="text-xl font-semibold text-white mb-4">Training Progress</h3>
          <div className="space-y-4 text-slate-300">
            <div className="flex justify-between">
              <span>Current Step:</span>
              <span className="font-mono">{metrics.step}</span>
            </div>
            <div className="flex justify-between">
              <span>Episode:</span>
              <span className="font-mono">{metrics.episode}</span>
            </div>
            <div className="flex justify-between">
              <span>Elapsed Time:</span>
              <span className="font-mono">
                {(metrics.elapsed_time / 60).toFixed(2)} min
              </span>
            </div>
            <div className="flex justify-between">
              <span>Predicted Total Time:</span>
              <span className="font-mono">
                {(metrics.predicted_total_time / 3600).toFixed(2)} hours
              </span>
            </div>
            <div className="flex justify-between">
              <span>Last Reward:</span>
              <span className="font-mono">{metrics.reward.toFixed(2)}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
