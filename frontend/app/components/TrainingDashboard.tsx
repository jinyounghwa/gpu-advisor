"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";

// ===== Types =====
interface TrainingMetrics {
  step: number;
  episode: number;
  timestamp: number;
  elapsed_time: number;
  loss: number;
  policy_loss: number;
  value_loss: number;
  reward: number;
  entropy: number;
  tps: number;
  vram_mb: number;
  ram_mb: number;
  cpu_percent: number;
  learning_rate: number;
  grad_norm: number;
  win_rate: number;
  episode_length: number;
  action_probs: number[];
  progress: number;
}

// ===== SVG Icons =====
const PlayIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
);
const StopIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>
);
const ActivityIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
);
const CpuIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>
);
const TrendUpIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>
);
const TrendDownIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/></svg>
);
const ZapIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
);

// ===== Mini Line Chart Component =====
function MiniChart({
  data,
  color,
  height = 48,
  width = 120,
}: {
  data: number[];
  color: string;
  height?: number;
  width?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length < 2) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    const padding = 4;

    // Gradient fill
    const gradient = ctx.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, color + "30");
    gradient.addColorStop(1, color + "00");

    ctx.beginPath();
    ctx.moveTo(padding, height - padding);

    const points: [number, number][] = [];
    for (let i = 0; i < data.length; i++) {
      const x = padding + (i / (data.length - 1)) * (width - padding * 2);
      const y =
        padding +
        (1 - (data[i] - min) / range) * (height - padding * 2);
      points.push([x, y]);
      if (i === 0) {
        ctx.lineTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }

    // Fill
    ctx.lineTo(width - padding, height - padding);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Line
    ctx.beginPath();
    for (let i = 0; i < points.length; i++) {
      if (i === 0) ctx.moveTo(points[i][0], points[i][1]);
      else ctx.lineTo(points[i][0], points[i][1]);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Dot at last point
    const last = points[points.length - 1];
    ctx.beginPath();
    ctx.arc(last[0], last[1], 2.5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }, [data, color, height, width]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width, height }}
      className="block"
    />
  );
}

// ===== Large Chart Component =====
function LineChart({
  datasets,
  labels,
  height = 200,
  title,
  yLabel,
}: {
  datasets: { data: number[]; color: string; label: string }[];
  labels: string[];
  height?: number;
  title: string;
  yLabel?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 400, height });

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setDimensions({ width: entry.contentRect.width, height });
      }
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, [height]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = dimensions.width;
    const h = dimensions.height;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const pad = { top: 16, right: 16, bottom: 32, left: 50 };
    const chartW = w - pad.left - pad.right;
    const chartH = h - pad.top - pad.bottom;

    // Compute global min/max
    let allMin = Infinity;
    let allMax = -Infinity;
    datasets.forEach((ds) => {
      ds.data.forEach((v) => {
        if (v < allMin) allMin = v;
        if (v > allMax) allMax = v;
      });
    });
    const range = allMax - allMin || 1;
    allMin -= range * 0.05;
    allMax += range * 0.05;
    const finalRange = allMax - allMin;

    // Grid lines
    const numGridLines = 5;
    ctx.strokeStyle = "#1e2030";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#4a5568";
    ctx.font = "10px Inter, sans-serif";
    ctx.textAlign = "right";
    for (let i = 0; i <= numGridLines; i++) {
      const y = pad.top + (i / numGridLines) * chartH;
      const val = allMax - (i / numGridLines) * finalRange;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(w - pad.right, y);
      ctx.stroke();
      ctx.fillText(val.toFixed(val > 10 ? 0 : 2), pad.left - 6, y + 3);
    }

    // X labels (sparse)
    ctx.textAlign = "center";
    const maxLabels = Math.min(labels.length, 8);
    const labelStep = Math.max(1, Math.floor(labels.length / maxLabels));
    for (let i = 0; i < labels.length; i += labelStep) {
      const x = pad.left + (i / Math.max(labels.length - 1, 1)) * chartW;
      ctx.fillText(labels[i], x, h - pad.bottom + 14);
    }

    // Draw datasets
    datasets.forEach((ds) => {
      if (ds.data.length < 2) return;

      // Fill gradient
      const gradient = ctx.createLinearGradient(0, pad.top, 0, pad.top + chartH);
      gradient.addColorStop(0, ds.color + "20");
      gradient.addColorStop(1, ds.color + "00");

      const points: [number, number][] = [];
      for (let i = 0; i < ds.data.length; i++) {
        const x = pad.left + (i / Math.max(ds.data.length - 1, 1)) * chartW;
        const y =
          pad.top + (1 - (ds.data[i] - allMin) / finalRange) * chartH;
        points.push([x, y]);
      }

      // Area fill
      ctx.beginPath();
      ctx.moveTo(points[0][0], pad.top + chartH);
      points.forEach(([x, y]) => ctx.lineTo(x, y));
      ctx.lineTo(points[points.length - 1][0], pad.top + chartH);
      ctx.closePath();
      ctx.fillStyle = gradient;
      ctx.fill();

      // Line
      ctx.beginPath();
      points.forEach(([x, y], i) => {
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = ds.color;
      ctx.lineWidth = 2;
      ctx.stroke();
    });

    // Y label
    if (yLabel) {
      ctx.save();
      ctx.translate(12, pad.top + chartH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillStyle = "#64748b";
      ctx.font = "10px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(yLabel, 0, 0);
      ctx.restore();
    }
  }, [datasets, labels, dimensions, yLabel]);

  return (
    <div className="glass-card p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-white">{title}</h3>
        <div className="flex items-center gap-3">
          {datasets.map((ds) => (
            <div key={ds.label} className="flex items-center gap-1.5">
              <div
                className="h-2 w-2 rounded-full"
                style={{ background: ds.color }}
              />
              <span className="text-[10px] text-[#64748b]">{ds.label}</span>
            </div>
          ))}
        </div>
      </div>
      <div ref={containerRef} className="w-full">
        <canvas
          ref={canvasRef}
          style={{ width: "100%", height }}
          className="block"
        />
      </div>
    </div>
  );
}

// ===== Bar Chart Component =====
function BarChart({
  data,
  labels,
  colors,
  title,
  height = 160,
}: {
  data: number[];
  labels: string[];
  colors: string[];
  title: string;
  height?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(300);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) setW(entry.contentRect.width);
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, height);

    const pad = { top: 12, right: 12, bottom: 28, left: 12 };
    const chartW = w - pad.left - pad.right;
    const chartH = height - pad.top - pad.bottom;
    const maxVal = Math.max(...data, 0.001);
    const barWidth = Math.min(40, (chartW / data.length) * 0.6);
    const gap = (chartW - barWidth * data.length) / (data.length + 1);

    data.forEach((value, i) => {
      const x = pad.left + gap * (i + 1) + barWidth * i;
      const barH = (value / maxVal) * chartH;
      const y = pad.top + chartH - barH;
      const color = colors[i % colors.length];

      // Bar with rounded top
      const radius = Math.min(4, barWidth / 2);
      ctx.beginPath();
      ctx.moveTo(x, y + radius);
      ctx.arcTo(x, y, x + barWidth, y, radius);
      ctx.arcTo(x + barWidth, y, x + barWidth, y + barH, radius);
      ctx.lineTo(x + barWidth, pad.top + chartH);
      ctx.lineTo(x, pad.top + chartH);
      ctx.closePath();

      // Gradient fill
      const gradient = ctx.createLinearGradient(0, y, 0, pad.top + chartH);
      gradient.addColorStop(0, color);
      gradient.addColorStop(1, color + "40");
      ctx.fillStyle = gradient;
      ctx.fill();

      // Value label
      ctx.fillStyle = "#94a3b8";
      ctx.font = "10px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(
        value >= 1 ? value.toFixed(0) : value.toFixed(2),
        x + barWidth / 2,
        y - 4
      );

      // Bottom label
      ctx.fillStyle = "#4a5568";
      ctx.font = "9px Inter, sans-serif";
      ctx.fillText(labels[i] || "", x + barWidth / 2, height - pad.bottom + 14);
    });
  }, [data, labels, colors, w, height]);

  return (
    <div className="glass-card p-4">
      <h3 className="mb-3 text-sm font-semibold text-white">{title}</h3>
      <div ref={containerRef} className="w-full">
        <canvas ref={canvasRef} style={{ width: "100%", height }} className="block" />
      </div>
    </div>
  );
}

// ===== Donut/Gauge Chart =====
function GaugeChart({
  value,
  max,
  label,
  color,
  unit,
  size = 100,
}: {
  value: number;
  max: number;
  label: string;
  color: string;
  unit: string;
  size?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, size, size);

    const cx = size / 2;
    const cy = size / 2;
    const radius = (size - 16) / 2;
    const lineWidth = 8;
    const startAngle = Math.PI * 0.75;
    const endAngle = Math.PI * 2.25;
    const sweepAngle = endAngle - startAngle;
    const progress = Math.min(value / max, 1);

    // Background arc
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, endAngle);
    ctx.strokeStyle = "#1e2030";
    ctx.lineWidth = lineWidth;
    ctx.lineCap = "round";
    ctx.stroke();

    // Progress arc
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, startAngle + sweepAngle * progress);
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = "round";
    ctx.stroke();

    // Value text
    ctx.fillStyle = "#ffffff";
    ctx.font = `bold ${size / 5}px Inter, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(
      value >= 100 ? value.toFixed(0) : value.toFixed(1),
      cx,
      cy - 4
    );

    // Unit text
    ctx.fillStyle = "#64748b";
    ctx.font = `${size / 9}px Inter, sans-serif`;
    ctx.fillText(unit, cx, cy + size / 5.5);
  }, [value, max, color, unit, size]);

  return (
    <div className="flex flex-col items-center gap-1">
      <canvas
        ref={canvasRef}
        style={{ width: size, height: size }}
        className="block"
      />
      <span className="text-[11px] text-[#64748b]">{label}</span>
    </div>
  );
}

// ===== Progress Bar =====
function ProgressBar({
  value,
  max,
  color,
  label,
}: {
  value: number;
  max: number;
  color: string;
  label: string;
}) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-[#94a3b8]">{label}</span>
        <span className="font-mono text-white">{pct.toFixed(1)}%</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-[#1e2030]">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{
            width: `${pct}%`,
            background: `linear-gradient(90deg, ${color}, ${color}cc)`,
            boxShadow: `0 0 8px ${color}40`,
          }}
        />
      </div>
    </div>
  );
}

// ===== Metric Card =====
function MetricCard({
  label,
  value,
  unit,
  trend,
  trendDirection,
  icon,
  sparkData,
  sparkColor,
}: {
  label: string;
  value: string;
  unit?: string;
  trend?: string;
  trendDirection?: "up" | "down";
  icon: React.ReactNode;
  sparkData?: number[];
  sparkColor?: string;
}) {
  return (
    <div className="glass-card flex items-center gap-3 p-4">
      <div
        className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg text-[#94a3b8]"
        style={{ background: sparkColor ? sparkColor + "15" : "#1e2030" }}
      >
        {icon}
      </div>
      <div className="min-w-0 flex-1">
        <div className="text-[11px] text-[#64748b]">{label}</div>
        <div className="flex items-baseline gap-1">
          <span className="text-lg font-bold text-white">{value}</span>
          {unit && <span className="text-xs text-[#4a5568]">{unit}</span>}
        </div>
        {trend && (
          <div
            className={`flex items-center gap-0.5 text-[10px] ${
              trendDirection === "up" ? "text-emerald-400" : "text-rose-400"
            }`}
          >
            {trendDirection === "up" ? <TrendUpIcon /> : <TrendDownIcon />}
            {trend}
          </div>
        )}
      </div>
      {sparkData && sparkData.length > 2 && sparkColor && (
        <MiniChart data={sparkData} color={sparkColor} width={80} height={36} />
      )}
    </div>
  );
}

// ===== Main Dashboard Component =====
export default function TrainingDashboard() {
  const [metricsHistory, setMetricsHistory] = useState<TrainingMetrics[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [numSteps, setNumSteps] = useState(500);
  const eventSourceRef = useRef<EventSource | null>(null);

  const connectStream = useCallback(() => {
    if (eventSourceRef.current) eventSourceRef.current.close();

    const es = new EventSource("http://localhost:8000/api/training/metrics/stream");
    eventSourceRef.current = es;

    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "done") {
          setIsTraining(false);
          es.close();
          return;
        }
        setMetricsHistory((prev) => [...prev, data]);
      } catch {
        // ignore parse errors
      }
    };

    es.onerror = () => {
      setIsTraining(false);
      es.close();
    };
  }, []);

  // Check server on mount
  useEffect(() => {
    fetch("http://localhost:8000/")
      .then(() => setIsConnected(true))
      .catch(() => setIsConnected(false));

    // Check for already running training
    fetch("http://localhost:8000/api/training/status")
      .then((r) => r.json())
      .then((data) => {
        if (data.is_training) {
          setIsTraining(true);
          connectStream();
        } else if (data.history_length > 0) {
          // Load existing metrics
          fetch("http://localhost:8000/api/training/metrics?last_n=0")
            .then((r) => r.json())
            .then((d) => setMetricsHistory(d.metrics || []));
        }
      })
      .catch(() => {});

    return () => {
      if (eventSourceRef.current) eventSourceRef.current.close();
    };
  }, [connectStream]);

  const startTraining = async () => {
    try {
      setMetricsHistory([]);
      const res = await fetch("http://localhost:8000/api/training/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ num_steps: numSteps, learning_rate: 1e-4, batch_size: 32 }),
      });
      if (!res.ok) throw new Error("Failed to start");
      setIsTraining(true);
      connectStream();
    } catch (err) {
      console.error("Start training error:", err);
    }
  };

  const stopTraining = async () => {
    try {
      await fetch("http://localhost:8000/api/training/stop", { method: "POST" });
      setIsTraining(false);
      if (eventSourceRef.current) eventSourceRef.current.close();
    } catch (err) {
      console.error("Stop training error:", err);
    }
  };

  // Derived data
  const latest = metricsHistory.length > 0 ? metricsHistory[metricsHistory.length - 1] : null;
  const last50 = metricsHistory.slice(-50);
  const last20 = metricsHistory.slice(-20);

  const lossData = last50.map((m) => m.loss);
  const rewardData = last50.map((m) => m.reward);
  const tpsData = last50.map((m) => m.tps);
  const vramData = last50.map((m) => m.vram_mb);
  const winRateData = last50.map((m) => m.win_rate);
  const entropyData = last50.map((m) => m.entropy);
  const gradNormData = last50.map((m) => m.grad_norm);
  const stepLabels = last50.map((m) => String(m.step));

  const lossChange = last20.length >= 2
    ? ((last20[last20.length - 1].loss - last20[0].loss) / Math.max(last20[0].loss, 0.001)) * 100
    : 0;
  const rewardChange = last20.length >= 2
    ? last20[last20.length - 1].reward - last20[0].reward
    : 0;

  return (
    <div className="w-full space-y-6">
      {/* Controls */}
      <div className="glass-card flex flex-wrap items-center gap-4 p-4">
        <div className="flex items-center gap-2">
          <div
            className={`h-2.5 w-2.5 rounded-full ${
              isTraining
                ? "bg-emerald-400 shadow-lg shadow-emerald-400/30 animate-pulse"
                : isConnected
                ? "bg-amber-400"
                : "bg-red-400"
            }`}
          />
          <span className="text-sm font-medium text-white">
            {isTraining ? "학습 진행 중" : isConnected ? "대기 중" : "서버 오프라인"}
          </span>
        </div>

        <div className="flex items-center gap-2">
          <label className="text-xs text-[#64748b]">스텝 수:</label>
          <input
            type="number"
            value={numSteps}
            onChange={(e) => setNumSteps(Number(e.target.value))}
            className="w-20 rounded-lg border border-[#23262f] bg-[#0d0f14] px-2 py-1 text-sm text-white outline-none focus:border-indigo-500"
            disabled={isTraining}
            min={10}
            max={10000}
            step={10}
          />
        </div>

        <div className="flex items-center gap-2">
          {!isTraining ? (
            <button
              onClick={startTraining}
              disabled={!isConnected}
              className="btn-primary flex items-center gap-2 text-sm"
            >
              <PlayIcon /> 학습 시작
            </button>
          ) : (
            <button
              onClick={stopTraining}
              className="flex items-center gap-2 rounded-lg border border-rose-500/30 bg-rose-500/10 px-4 py-2 text-sm font-medium text-rose-400 transition-colors hover:bg-rose-500/20"
            >
              <StopIcon /> 중지
            </button>
          )}
        </div>

        {latest && (
          <div className="ml-auto flex items-center gap-2 text-xs text-[#64748b]">
            <span>Step {latest.step} / {numSteps}</span>
            <span>•</span>
            <span>{latest.progress.toFixed(1)}%</span>
          </div>
        )}
      </div>

      {/* Progress Bar */}
      {(isTraining || (latest && latest.progress > 0)) && (
        <ProgressBar
          value={latest?.progress || 0}
          max={100}
          color="#6366f1"
          label="학습 진행률"
        />
      )}

      {/* Top Metric Cards */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricCard
          label="Loss"
          value={latest?.loss.toFixed(4) || "—"}
          icon={<ActivityIcon />}
          trend={lossChange !== 0 ? `${Math.abs(lossChange).toFixed(1)}%` : undefined}
          trendDirection={lossChange < 0 ? "up" : "down"}
          sparkData={lossData.slice(-20)}
          sparkColor="#f43f5e"
        />
        <MetricCard
          label="Reward"
          value={latest?.reward.toFixed(2) || "—"}
          icon={<ZapIcon />}
          trend={rewardChange !== 0 ? `${rewardChange > 0 ? "+" : ""}${rewardChange.toFixed(2)}` : undefined}
          trendDirection={rewardChange > 0 ? "up" : "down"}
          sparkData={rewardData.slice(-20)}
          sparkColor="#10b981"
        />
        <MetricCard
          label="TPS"
          value={latest?.tps.toFixed(0) || "—"}
          unit="steps/s"
          icon={<CpuIcon />}
          sparkData={tpsData.slice(-20)}
          sparkColor="#6366f1"
        />
        <MetricCard
          label="Win Rate"
          value={latest ? (latest.win_rate * 100).toFixed(1) : "—"}
          unit="%"
          icon={<TrendUpIcon />}
          sparkData={winRateData.slice(-20)}
          sparkColor="#f59e0b"
        />
      </div>

      {/* Gauge Charts */}
      <div className="glass-card p-4">
        <h3 className="mb-4 text-sm font-semibold text-white">시스템 리소스</h3>
        <div className="flex flex-wrap items-center justify-around gap-4">
          <GaugeChart
            value={latest?.cpu_percent || 0}
            max={100}
            label="CPU 사용률"
            color="#6366f1"
            unit="%"
          />
          <GaugeChart
            value={latest?.vram_mb || 0}
            max={4000}
            label="VRAM"
            color="#8b5cf6"
            unit="MB"
          />
          <GaugeChart
            value={latest?.ram_mb || 0}
            max={16000}
            label="RAM"
            color="#06b6d4"
            unit="MB"
          />
          <GaugeChart
            value={latest?.tps || 0}
            max={300}
            label="처리 속도"
            color="#10b981"
            unit="TPS"
          />
        </div>
      </div>

      {/* Main Charts */}
      {metricsHistory.length > 2 && (
        <>
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <LineChart
              title="Loss (Total / Policy / Value)"
              datasets={[
                { data: last50.map((m) => m.loss), color: "#f43f5e", label: "Total" },
                { data: last50.map((m) => m.policy_loss), color: "#f97316", label: "Policy" },
                { data: last50.map((m) => m.value_loss), color: "#a855f7", label: "Value" },
              ]}
              labels={stepLabels}
              yLabel="Loss"
            />
            <LineChart
              title="Reward & Win Rate"
              datasets={[
                { data: rewardData, color: "#10b981", label: "Reward" },
                { data: winRateData.map((v) => v * 3), color: "#f59e0b", label: "Win Rate (×3)" },
              ]}
              labels={stepLabels}
              yLabel="Value"
            />
          </div>

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <LineChart
              title="처리 속도 (TPS)"
              datasets={[{ data: tpsData, color: "#6366f1", label: "TPS" }]}
              labels={stepLabels}
              yLabel="Steps/sec"
            />
            <LineChart
              title="Entropy & Gradient Norm"
              datasets={[
                { data: entropyData, color: "#06b6d4", label: "Entropy" },
                { data: gradNormData, color: "#ec4899", label: "Grad Norm" },
              ]}
              labels={stepLabels}
              yLabel="Value"
            />
          </div>

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <LineChart
              title="메모리 사용량"
              datasets={[
                { data: vramData, color: "#8b5cf6", label: "VRAM (MB)" },
                { data: last50.map((m) => m.ram_mb), color: "#06b6d4", label: "RAM (MB)" },
              ]}
              labels={stepLabels}
              yLabel="MB"
              height={180}
            />
            <BarChart
              title="Action Distribution"
              data={latest?.action_probs || []}
              labels={["Action 0", "Action 1", "Action 2", "Action 3", "Action 4"]}
              colors={["#6366f1", "#8b5cf6", "#a855f7", "#c084fc", "#e879f9"]}
              height={180}
            />
          </div>
        </>
      )}

      {/* Empty state */}
      {metricsHistory.length === 0 && !isTraining && (
        <div className="glass-card flex flex-col items-center py-16 text-center">
          <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-[#16181f] text-[#334155] border border-[#23262f]">
            <ActivityIcon />
          </div>
          <p className="text-sm text-[#4a5568]">
            학습을 시작하면 실시간 지표가 여기에 표시됩니다
          </p>
          <p className="mt-1 text-xs text-[#334155]">
            상단의 &lsquo;학습 시작&rsquo; 버튼을 눌러 시뮬레이션을 시작하세요
          </p>
        </div>
      )}

      {/* Training Summary */}
      {metricsHistory.length > 0 && !isTraining && (
        <div className="glass-card p-5">
          <h3 className="mb-4 text-sm font-semibold text-white">📊 학습 결과 요약</h3>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <div>
              <div className="text-[11px] text-[#64748b]">총 진행 스텝</div>
              <div className="text-lg font-bold text-white">{metricsHistory.length}</div>
            </div>
            <div>
              <div className="text-[11px] text-[#64748b]">최종 Loss</div>
              <div className="text-lg font-bold text-rose-400">
                {metricsHistory[metricsHistory.length - 1]?.loss.toFixed(4)}
              </div>
            </div>
            <div>
              <div className="text-[11px] text-[#64748b]">최고 Reward</div>
              <div className="text-lg font-bold text-emerald-400">
                {Math.max(...metricsHistory.map((m) => m.reward)).toFixed(2)}
              </div>
            </div>
            <div>
              <div className="text-[11px] text-[#64748b]">최종 Win Rate</div>
              <div className="text-lg font-bold text-amber-400">
                {(metricsHistory[metricsHistory.length - 1]?.win_rate * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
