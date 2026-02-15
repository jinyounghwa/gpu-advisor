"use client";

import { useEffect, useRef } from "react";

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

interface ChartProps {
  data: TrainingMetrics[];
  dataKey: keyof TrainingMetrics;
  color: string;
  title?: string;
  unit?: string;
}

export default function Chart({ data, dataKey, color, title, unit = "" }: ChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const padding = { top: 20, right: 20, bottom: 40, left: 60 };
    const width = rect.width - padding.left - padding.right;
    const height = rect.height - padding.top - padding.bottom;

    const values = data.map((d) => d[dataKey] as number);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    const displayMin = min - range * 0.1;
    const displayMax = max + range * 0.1;
    const displayRange = displayMax - displayMin;

    const drawChart = (progress: number = 1) => {
      ctx.clearRect(0, 0, rect.width, rect.height);

      // Draw gradient background
      const gradient = ctx.createLinearGradient(0, padding.top, 0, padding.top + height);
      gradient.addColorStop(0, `${color}30`);
      gradient.addColorStop(1, `${color}05`);

      // Draw grid lines
      ctx.strokeStyle = "#334155";
      ctx.lineWidth = 0.5;
      ctx.font = "11px 'SF Mono', Monaco, 'Cascadia Code', monospace";
      ctx.fillStyle = "#94a3b8";

      const ySteps = 6;
      for (let i = 0; i <= ySteps; i++) {
        const value = displayMin + (displayRange * i) / ySteps;
        const y = padding.top + height - (i / ySteps) * height;

        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(padding.left + width, y);
        ctx.stroke();

        const label = `${value.toFixed(2)}${unit}`;
        ctx.fillText(label, padding.left - 50, y + 4);
      }

      const xSteps = 10;
      for (let i = 0; i <= xSteps; i++) {
        const x = padding.left + (i / xSteps) * width;
        const dataIndex = Math.floor((i / xSteps) * (data.length - 1));
        const dataPoint = data[dataIndex];

        ctx.beginPath();
        ctx.moveTo(x, padding.top);
        ctx.lineTo(x, padding.top + height);
        ctx.stroke();

        if (dataPoint) {
          ctx.fillText(`#${dataPoint.step}`, x - 20, padding.top + height + 15);
        }
      }

      if (data.length < 2) return;

      const dataToDraw = data.slice(0, Math.floor(data.length * progress));

      // Draw smooth line
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";

      dataToDraw.forEach((point, i) => {
        const x = padding.left + (i / (data.length - 1 || 1)) * width;
        const normalizedValue = ((point[dataKey] as number - displayMin) / displayRange);
        const y = padding.top + height - normalizedValue * height;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          const prevX = padding.left + ((i - 1) / (data.length - 1)) * width;
          const prevPoint = dataToDraw[i - 1];
          const prevNormalized = ((prevPoint[dataKey] as number - displayMin) / displayRange);
          const prevY = padding.top + height - prevNormalized * height;
          const cpX = (prevX + x) / 2;
          ctx.bezierCurveTo(cpX, prevY, cpX, y, x, y);
        }
      });

      ctx.stroke();

      // Draw gradient fill under line
      ctx.lineTo(padding.left + ((dataToDraw.length - 1) / (data.length - 1 || 1)) * width, padding.top + height);
      ctx.lineTo(padding.left, padding.top + height);
      ctx.closePath();

      const fillGradient = ctx.createLinearGradient(0, padding.top, 0, padding.top + height);
      fillGradient.addColorStop(0, `${color}60`);
      fillGradient.addColorStop(0.5, `${color}30`);
      fillGradient.addColorStop(1, `${color}05`);

      ctx.fillStyle = fillGradient;
      ctx.fill();

      // Draw data points with glow
      dataToDraw.forEach((point, i) => {
        if (i % Math.max(1, Math.floor(dataToDraw.length / 20)) !== 0) return;

        const x = padding.left + (i / (data.length - 1 || 1)) * width;
        const normalizedValue = ((point[dataKey] as number - displayMin) / displayRange);
        const y = padding.top + height - normalizedValue * height;

        // Glow effect
        const glowGradient = ctx.createRadialGradient(x, y, 0, x, y, 10);
        glowGradient.addColorStop(0, `${color}60`);
        glowGradient.addColorStop(1, `${color}00`);
        ctx.fillStyle = glowGradient;
        ctx.beginPath();
        ctx.arc(x, y, 10, 0, Math.PI * 2);
        ctx.fill();

        // Point
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();

        // Point outline
        ctx.strokeStyle = "#0f172a";
        ctx.lineWidth = 2;
        ctx.stroke();
      });

      // Draw latest value indicator
      const latestIndex = dataToDraw.length - 1;
      const latestPoint = dataToDraw[latestIndex];
      if (latestPoint) {
        const x = padding.left + (latestIndex / (data.length - 1 || 1)) * width;
        const normalizedValue = ((latestPoint[dataKey] as number - displayMin) / displayRange);
        const y = padding.top + height - normalizedValue * height;

        // Latest value badge
        const badgeText = `${latestPoint[dataKey].toFixed(2)}${unit}`;
        ctx.font = "bold 12px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
        const textWidth = ctx.measureText(badgeText).width;

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.roundRect(x - textWidth / 2 - 8, y - 35, textWidth + 16, 24, 6);
        ctx.fill();

        ctx.fillStyle = "#0f172a";
        ctx.fillText(badgeText, x - textWidth / 2, y - 19);
      }
    };

    let animationProgress = 0;
    const animate = () => {
      animationProgress += 0.05;
      if (animationProgress > 1) animationProgress = 1;

      drawChart(animationProgress);

      if (animationProgress < 1) {
        animationRef.current = requestAnimationFrame(animate);
      }
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [data, dataKey, color, unit]);

  return (
    <div className="relative w-full h-72">
      <canvas
        ref={canvasRef}
        className="w-full h-full rounded-lg"
        style={{ background: "linear-gradient(to bottom, #1e293b, #0f172a)" }}
      />
      {title && (
        <div className="absolute top-4 left-14">
          <h4 className="text-sm font-semibold text-slate-400">{title}</h4>
        </div>
      )}
    </div>
  );
}
