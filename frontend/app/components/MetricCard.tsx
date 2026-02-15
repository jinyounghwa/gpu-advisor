"use client";

import { useEffect, useState } from "react";

interface MetricCardProps {
  title: string;
  value: string;
  unit: string;
  icon: string;
  color: "yellow" | "blue" | "purple" | "red" | "green";
}

const colorConfig = {
  yellow: {
    bg: "from-yellow-500/10 to-amber-600/5",
    border: "border-yellow-500/30",
    icon: "bg-gradient-to-br from-yellow-400 to-amber-500",
    text: "text-yellow-400",
    glow: "shadow-yellow-500/20",
  },
  blue: {
    bg: "from-blue-500/10 to-cyan-600/5",
    border: "border-blue-500/30",
    icon: "bg-gradient-to-br from-blue-400 to-cyan-500",
    text: "text-blue-400",
    glow: "shadow-blue-500/20",
  },
  purple: {
    bg: "from-purple-500/10 to-violet-600/5",
    border: "border-purple-500/30",
    icon: "bg-gradient-to-br from-purple-400 to-violet-500",
    text: "text-purple-400",
    glow: "shadow-purple-500/20",
  },
  red: {
    bg: "from-red-500/10 to-rose-600/5",
    border: "border-red-500/30",
    icon: "bg-gradient-to-br from-red-400 to-rose-500",
    text: "text-red-400",
    glow: "shadow-red-500/20",
  },
  green: {
    bg: "from-green-500/10 to-emerald-600/5",
    border: "border-green-500/30",
    icon: "bg-gradient-to-br from-green-400 to-emerald-500",
    text: "text-green-400",
    glow: "shadow-green-500/20",
  },
};

export default function MetricCard({
  title,
  value,
  unit,
  icon,
  color,
}: MetricCardProps) {
  const [animate, setAnimate] = useState(false);

  useEffect(() => {
    setAnimate(true);
  }, [value]);

  const config = colorConfig[color];

  return (
    <div
      className={`relative bg-gradient-to-br ${config.bg} ${config.border} border backdrop-blur-xl rounded-2xl p-6 transition-all duration-500 hover:scale-105 hover:shadow-2xl ${config.glow}`}
    >
      <div className="flex items-start justify-between mb-4">
        <h3 className="text-slate-300 font-medium text-sm uppercase tracking-wide">{title}</h3>
        <div className={`w-12 h-12 rounded-xl ${config.icon} flex items-center justify-center text-2xl shadow-lg ${config.glow}`}>
          {icon}
        </div>
      </div>
      <div className="flex items-baseline gap-2">
        <span
          className={`text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r ${config.text} transition-all duration-300 ${animate ? "scale-105" : ""}`}
        >
          {value}
        </span>
        {unit && <span className="text-slate-400 text-sm">{unit}</span>}
      </div>
      <div className="absolute top-4 right-4 w-20 h-20 rounded-full blur-3xl opacity-30 bg-gradient-to-br from-white/20 to-transparent pointer-events-none" />
    </div>
  );
}
