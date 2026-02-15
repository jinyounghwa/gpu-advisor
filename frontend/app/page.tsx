"use client";

import { useState, useRef, useEffect } from "react";
import dynamic from "next/dynamic";

const TrainingDashboard = dynamic(
  () => import("./components/TrainingDashboard"),
  { ssr: false }
);

interface GPUResult {
  title: string;
  summary: string;
  specs: string;
  usage: string;
  recommendation: string;
}

const QUICK_CHIPS = ["RTX 4090", "RTX 4080", "H100", "M4 Mac"];

// ===== SVG Icons =====
function GpuIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <rect x="4" y="4" width="16" height="16" rx="2" />
      <rect x="8" y="8" width="8" height="8" rx="1" />
      <line x1="2" y1="9" x2="4" y2="9" />
      <line x1="2" y1="15" x2="4" y2="15" />
      <line x1="20" y1="9" x2="22" y2="9" />
      <line x1="20" y1="15" x2="22" y2="15" />
      <line x1="9" y1="2" x2="9" y2="4" />
      <line x1="15" y1="2" x2="15" y2="4" />
      <line x1="9" y1="20" x2="9" y2="22" />
      <line x1="15" y1="20" x2="15" y2="22" />
    </svg>
  );
}

function SearchIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  );
}

function LoadingSpinner() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="animate-spin">
      <path d="M21 12a9 9 0 1 1-6.219-8.56" />
    </svg>
  );
}

function SpecsIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="3" width="20" height="14" rx="2" />
      <line x1="8" y1="21" x2="16" y2="21" />
      <line x1="12" y1="17" x2="12" y2="21" />
    </svg>
  );
}

function UsageIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2L2 7l10 5 10-5-10-5z" />
      <path d="M2 17l10 5 10-5" />
      <path d="M2 12l10 5 10-5" />
    </svg>
  );
}

function LightbulbIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 18h6" />
      <path d="M10 22h4" />
      <path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8 6 6 0 0 0 6 8c0 1 .23 2.23 1.5 3.5A4.61 4.61 0 0 1 8.91 14" />
    </svg>
  );
}

function ChartIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 3v18h18" />
      <path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3" />
    </svg>
  );
}

type TabKey = "advisor" | "training";

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabKey>("advisor");
  const [input, setInput] = useState("");
  const [result, setResult] = useState<GPUResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [serverStatus, setServerStatus] = useState<"checking" | "online" | "offline">("checking");
  const inputRef = useRef<HTMLInputElement>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  // Server health check
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch("http://localhost:8000/");
        if (res.ok) {
          setServerStatus("online");
        } else {
          setServerStatus("offline");
        }
      } catch {
        setServerStatus("offline");
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  // Scroll to result
  useEffect(() => {
    if (result && resultRef.current) {
      resultRef.current.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [result]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    await searchGPU(input.trim());
  };

  const searchGPU = async (query: string) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch("http://localhost:8000/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: query }),
      });

      if (!res.ok) throw new Error("서버 통신 오류");

      const data = await res.json();
      setResult(data);
    } catch {
      setError("백엔드 서버와 연결할 수 없습니다. 서버 상태를 확인해주세요.");
    } finally {
      setLoading(false);
    }
  };

  const handleChipClick = (chip: string) => {
    setInput(chip);
    searchGPU(chip);
  };

  const tabs: { key: TabKey; label: string; icon: React.ReactNode }[] = [
    {
      key: "advisor",
      label: "GPU Advisor",
      icon: <SearchIcon />,
    },
    {
      key: "training",
      label: "AI Training",
      icon: <ChartIcon />,
    },
  ];

  return (
    <div className="relative min-h-screen flex flex-col">
      {/* Ambient glow backgrounds */}
      <div
        className="pointer-events-none fixed"
        style={{
          top: "-20%",
          left: "-10%",
          width: "50%",
          height: "50%",
          background: "radial-gradient(ellipse, rgba(99,102,241,0.08) 0%, transparent 70%)",
          filter: "blur(60px)",
        }}
      />
      <div
        className="pointer-events-none fixed"
        style={{
          bottom: "-20%",
          right: "-10%",
          width: "50%",
          height: "50%",
          background: "radial-gradient(ellipse, rgba(139,92,246,0.06) 0%, transparent 70%)",
          filter: "blur(60px)",
        }}
      />

      {/* Header */}
      <header className="relative z-10 border-b border-[#1a1d24] px-6 py-3">
        <div className="mx-auto flex max-w-6xl items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 text-white shadow-lg shadow-indigo-500/20">
              <GpuIcon />
            </div>
            <span className="text-base font-bold tracking-tight text-white">
              GPU Advisor
            </span>
          </div>

          {/* Tab Navigation */}
          <nav className="flex items-center rounded-xl border border-[#1e2030] bg-[#0d0f14]/80 p-1 backdrop-blur-sm">
            {tabs.map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-all duration-200 ${
                  activeTab === tab.key
                    ? "bg-gradient-to-r from-indigo-500/20 to-purple-500/20 text-white shadow-sm border border-indigo-500/20"
                    : "text-[#64748b] hover:text-[#94a3b8] hover:bg-[#16181f]"
                }`}
                id={`tab-${tab.key}`}
              >
                {tab.icon}
                {tab.label}
              </button>
            ))}
          </nav>

          <div className="flex items-center gap-2">
            <div
              className={`h-2 w-2 rounded-full ${
                serverStatus === "online"
                  ? "bg-emerald-400 shadow-lg shadow-emerald-400/40"
                  : serverStatus === "offline"
                  ? "bg-red-400 shadow-lg shadow-red-400/40"
                  : "bg-amber-400 animate-pulse"
              }`}
            />
            <span className="text-xs text-[#64748b]">
              {serverStatus === "online"
                ? "서버 연결됨"
                : serverStatus === "offline"
                ? "서버 오프라인"
                : "확인 중..."}
            </span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 mx-auto flex w-full max-w-6xl flex-1 flex-col px-6 py-8">
        {/* ===== GPU Advisor Tab ===== */}
        {activeTab === "advisor" && (
          <div className="flex flex-col items-center">
            {/* Hero section */}
            <section className="mb-10 w-full max-w-2xl text-center animate-fade-in-up">
              <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-[#23262f] bg-[#16181f]/60 px-4 py-1.5 text-xs text-[#94a3b8] backdrop-blur-sm">
                <span className="inline-block h-1.5 w-1.5 rounded-full bg-indigo-400" />
                AI 기반 GPU 추천 시스템
              </div>
              <h1 className="mb-3 text-3xl font-extrabold leading-tight tracking-tight sm:text-4xl">
                <span className="gradient-text">최적의 GPU</span>를 찾아보세요
              </h1>
              <p className="mx-auto max-w-lg text-sm leading-relaxed text-[#94a3b8]">
                그래픽카드 모델명을 입력하면 AI가 스펙, 용도, 추천 정보를 분석해 드립니다.
              </p>
            </section>

            {/* Search */}
            <form
              onSubmit={handleSubmit}
              className="animate-fade-in-up stagger-1 mb-5 w-full max-w-2xl"
              style={{ opacity: 0 }}
            >
              <div className="glass-card flex items-center gap-3 p-2 pr-2">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg text-[#64748b]">
                  <SearchIcon />
                </div>
                <input
                  ref={inputRef}
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="GPU 모델명 입력 (예: RTX 4090, H100, M4)"
                  className="flex-1 bg-transparent text-base text-white outline-none placeholder-[#4a5568]"
                  autoFocus
                  id="gpu-search-input"
                />
                <button
                  type="submit"
                  disabled={loading || !input.trim()}
                  className="btn-primary flex items-center gap-2 whitespace-nowrap"
                  id="gpu-search-btn"
                >
                  {loading ? <LoadingSpinner /> : <SearchIcon />}
                  {loading ? "분석 중..." : "검색"}
                </button>
              </div>
            </form>

            {/* Quick chips */}
            <div
              className="animate-fade-in-up stagger-2 mb-8 flex w-full max-w-2xl flex-wrap items-center justify-center gap-2"
              style={{ opacity: 0 }}
            >
              <span className="mr-1 text-xs text-[#4a5568]">빠른 검색:</span>
              {QUICK_CHIPS.map((chip) => (
                <button
                  key={chip}
                  onClick={() => handleChipClick(chip)}
                  className="rounded-full border border-[#23262f] bg-[#16181f]/60 px-4 py-1.5 text-xs font-medium text-[#94a3b8] transition-all hover:border-indigo-500/40 hover:bg-indigo-500/10 hover:text-indigo-300"
                  id={`chip-${chip.replace(/\s/g, "-").toLowerCase()}`}
                >
                  {chip}
                </button>
              ))}
            </div>

            {/* Error */}
            {error && (
              <div className="animate-fade-in mb-8 w-full max-w-2xl rounded-2xl border border-red-500/20 bg-red-500/5 px-6 py-4 text-center text-sm text-red-300">
                <svg
                  className="mb-2 mx-auto"
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <circle cx="12" cy="12" r="10" />
                  <line x1="12" y1="8" x2="12" y2="12" />
                  <line x1="12" y1="16" x2="12.01" y2="16" />
                </svg>
                {error}
              </div>
            )}

            {/* Result */}
            {result && (
              <div ref={resultRef} className="animate-fade-in-up w-full max-w-2xl space-y-4">
                {/* Title card */}
                <div className="glass-card p-6">
                  <div className="mb-1 flex items-center gap-2">
                    <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500/20 to-purple-500/20 text-indigo-300">
                      <GpuIcon />
                    </div>
                    <h2 className="text-xl font-bold text-white">
                      {result.title}
                    </h2>
                  </div>
                  <p className="ml-10 text-sm text-[#94a3b8]">{result.summary}</p>
                </div>

                {/* Info grid */}
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  <div className="glass-card p-5">
                    <div className="mb-3 flex items-center gap-2 text-indigo-300">
                      <SpecsIcon />
                      <h3 className="text-xs font-semibold uppercase tracking-wider">사양</h3>
                    </div>
                    <p className="text-base font-mono font-semibold text-white leading-relaxed">
                      {result.specs}
                    </p>
                  </div>
                  <div className="glass-card p-5">
                    <div className="mb-3 flex items-center gap-2 text-purple-300">
                      <UsageIcon />
                      <h3 className="text-xs font-semibold uppercase tracking-wider">추천 용도</h3>
                    </div>
                    <p className="text-base text-white leading-relaxed">{result.usage}</p>
                  </div>
                </div>

                {/* Recommendation */}
                <div className="glass-card relative overflow-hidden p-5">
                  <div
                    className="pointer-events-none absolute -right-10 -top-10 h-32 w-32 rounded-full"
                    style={{
                      background: "radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 70%)",
                    }}
                  />
                  <div className="relative">
                    <div className="mb-3 flex items-center gap-2 text-amber-300">
                      <LightbulbIcon />
                      <h3 className="text-xs font-semibold uppercase tracking-wider">AI 추천</h3>
                    </div>
                    <p className="text-sm leading-relaxed text-[#cbd5e1]">
                      {result.recommendation}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Empty state */}
            {!result && !error && !loading && (
              <div
                className="animate-fade-in-up stagger-3 mt-4 flex w-full max-w-2xl flex-col items-center py-12 text-center"
                style={{ opacity: 0 }}
              >
                <div className="animate-float mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-[#16181f] to-[#1a1d24] text-[#334155] shadow-inner border border-[#23262f]">
                  <GpuIcon />
                </div>
                <p className="mb-1 text-sm font-medium text-[#475569]">
                  검색 결과가 여기에 표시됩니다
                </p>
                <p className="text-xs text-[#334155]">
                  위의 검색바에 GPU 모델명을 입력하거나 빠른 검색을 사용하세요
                </p>
              </div>
            )}
          </div>
        )}

        {/* ===== Training Dashboard Tab ===== */}
        {activeTab === "training" && (
          <div className="animate-fade-in-up w-full">
            <div className="mb-6">
              <h1 className="text-2xl font-extrabold tracking-tight text-white sm:text-3xl">
                <span className="gradient-text">AI Training</span> Dashboard
              </h1>
              <p className="mt-1 text-sm text-[#64748b]">
                AlphaZero 기반 강화학습 실시간 지표 모니터링
              </p>
            </div>
            <TrainingDashboard />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="relative z-10 border-t border-[#1a1d24] px-6 py-4">
        <div className="mx-auto flex max-w-6xl items-center justify-between text-xs text-[#334155]">
          <span>© 2026 GPU Advisor</span>
          <span>Powered by FastAPI + Next.js</span>
        </div>
      </footer>
    </div>
  );
}
