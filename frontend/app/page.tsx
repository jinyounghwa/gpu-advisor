"use client";

import { useEffect, useState, useRef } from "react";

interface GPUResult {
  title: string;
  summary: string;
  specs: string;
  usage: string;
  recommendation: string;
  agent_trace?: {
    selected_action: string;
    raw_action?: string;
    confidence: number;
    entropy?: number;
    value: number;
    safe_mode?: boolean;
    safe_reason?: string | null;
    action_probs_text: string;
    expected_rewards_text: string;
  };
}

const QUICK_CHIPS = ["RTX 4090", "RTX 4080", "H100", "M4 Mac"];

function GpuIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="4" y="4" width="16" height="16" rx="2" />
      <rect x="8" y="8" width="8" height="8" rx="1" />
      <line x1="2" y1="9" x2="4" y2="9" />
      <line x1="2" y1="15" x2="4" y2="15" />
      <line x1="20" y1="9" x2="22" y2="9" />
      <line x1="20" y1="15" x2="22" y2="15" />
    </svg>
  );
}

function SearchIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  );
}

function LoadingSpinner() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="animate-spin">
      <path d="M21 12a9 9 0 1 1-6.219-8.56" />
    </svg>
  );
}

function SpecsIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="3" width="20" height="14" rx="2" />
      <line x1="8" y1="21" x2="16" y2="21" />
      <line x1="12" y1="17" x2="12" y2="21" />
    </svg>
  );
}

function UsageIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2L2 7l10 5 10-5-10-5z" />
      <path d="M2 17l10 5 10-5" />
      <path d="M2 12l10 5 10-5" />
    </svg>
  );
}

function LightbulbIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 18h6" />
      <path d="M10 22h4" />
      <path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8 6 6 0 0 0 6 8c0 1 .23 2.23 1.5 3.5A4.61 4.61 0 0 1 8.91 14" />
    </svg>
  );
}

export default function Home() {
  const [input, setInput] = useState("");
  const [result, setResult] = useState<GPUResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [serverStatus, setServerStatus] = useState<"checking" | "online" | "offline">("checking");
  const resultRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch("http://localhost:8000/");
        setServerStatus(res.ok ? "online" : "offline");
      } catch {
        setServerStatus("offline");
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (result && resultRef.current) {
      resultRef.current.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [result]);

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

      if (!res.ok) {
        throw new Error("request failed");
      }

      const data = await res.json();
      setResult(data);
    } catch {
      setError("백엔드 서버와 연결할 수 없습니다. 서버 상태를 확인해주세요.");
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const query = input.trim();
    if (!query) return;
    await searchGPU(query);
  };

  const handleChipClick = (chip: string) => {
    setInput(chip);
    searchGPU(chip);
  };

  return (
    <div className="app-shell min-h-screen">
      <div className="ambient-orb ambient-orb-left" />
      <div className="ambient-orb ambient-orb-right" />
      <div className="grid-overlay" />

      <header className="relative z-10 border-b border-white/10 px-4 py-4 sm:px-6">
        <div className="mx-auto flex w-full max-w-6xl items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-400 to-sky-500 text-white shadow-[0_0_25px_rgba(14,165,233,0.45)]">
              <GpuIcon />
            </div>
            <div>
              <p className="text-[10px] uppercase tracking-[0.24em] text-cyan-200/70">AI Stack</p>
              <p className="text-sm font-semibold text-white">GPU Advisor</p>
            </div>
          </div>

          <div className="rounded-full border border-cyan-300/30 bg-cyan-400/10 px-3 py-1 text-xs text-cyan-100">
            Advisor Mode
          </div>

          <div className="hidden items-center gap-2 rounded-full border border-white/10 bg-black/25 px-3 py-1 text-xs text-slate-300 md:flex">
            <span
              className={`h-2 w-2 rounded-full ${
                serverStatus === "online"
                  ? "bg-emerald-400 shadow-[0_0_16px_rgba(52,211,153,0.9)]"
                  : serverStatus === "offline"
                  ? "bg-rose-400"
                  : "bg-amber-400 animate-pulse"
              }`}
            />
            {serverStatus === "online"
              ? "서버 연결됨"
              : serverStatus === "offline"
              ? "서버 오프라인"
              : "확인 중..."}
          </div>
        </div>
      </header>

      <main className="relative z-10 flex min-h-[calc(100vh-150px)] items-center px-4 py-8 sm:px-6">
        <div className="mx-auto w-full max-w-6xl">
          <section className="glass-panel mx-auto w-full max-w-4xl p-6 sm:p-10">
              <div className="mx-auto flex max-w-2xl flex-col items-center text-center">
                <span className="badge mb-4">AI 기반 GPU 분석 엔진</span>
                <h1 className="headline mb-3">중앙에서 시작하는 GPU 탐색</h1>
                <p className="text-sm leading-relaxed text-slate-300">
                  모델명을 입력하면 성능 포지션, 추천 용도, 의사결정 근거를 한 번에 제공합니다.
                </p>
              </div>

              <form onSubmit={handleSubmit} className="mx-auto mt-8 w-full max-w-2xl">
                <div className="search-shell">
                  <SearchIcon />
                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="GPU 모델명 입력 (예: RTX 4090, H100, M4)"
                    className="search-input"
                    autoFocus
                    id="gpu-search-input"
                  />
                  <button
                    type="submit"
                    disabled={loading || !input.trim()}
                    className="btn-primary"
                    id="gpu-search-btn"
                  >
                    {loading ? <LoadingSpinner /> : <SearchIcon />}
                    {loading ? "분석 중..." : "검색"}
                  </button>
                </div>
              </form>

              <div className="mx-auto mt-4 flex w-full max-w-2xl flex-wrap items-center justify-center gap-2">
                {QUICK_CHIPS.map((chip) => (
                  <button
                    key={chip}
                    onClick={() => handleChipClick(chip)}
                    className="chip-btn"
                    id={`chip-${chip.replace(/\s/g, "-").toLowerCase()}`}
                  >
                    {chip}
                  </button>
                ))}
              </div>

              {error && (
                <div className="mx-auto mt-6 w-full max-w-2xl rounded-2xl border border-rose-400/30 bg-rose-500/10 px-5 py-4 text-center text-sm text-rose-200">
                  {error}
                </div>
              )}

              {result && (
                <div ref={resultRef} className="mx-auto mt-8 w-full max-w-3xl space-y-4">
                  <div className="result-card">
                    <p className="text-xs uppercase tracking-[0.2em] text-cyan-200/70">Result</p>
                    <h2 className="mt-2 text-2xl font-semibold text-white">{result.title}</h2>
                    <p className="mt-2 text-sm leading-relaxed text-slate-300">{result.summary}</p>
                  </div>

                  <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                    <div className="result-card">
                      <div className="mb-2 flex items-center gap-2 text-cyan-300">
                        <SpecsIcon />
                        <p className="text-xs uppercase tracking-wider">사양</p>
                      </div>
                      <p className="font-mono text-lg text-white">{result.specs}</p>
                    </div>
                    <div className="result-card">
                      <div className="mb-2 flex items-center gap-2 text-orange-300">
                        <UsageIcon />
                        <p className="text-xs uppercase tracking-wider">추천 용도</p>
                      </div>
                      <p className="text-sm leading-relaxed text-slate-100">{result.usage}</p>
                    </div>
                  </div>

                  <div className="result-card border-cyan-300/20">
                    <div className="mb-2 flex items-center gap-2 text-cyan-200">
                      <LightbulbIcon />
                      <p className="text-xs uppercase tracking-wider">AI 추천</p>
                    </div>
                    <p className="text-sm leading-relaxed text-slate-200">{result.recommendation}</p>

                    {result.agent_trace && (
                      <div className="mt-4 rounded-xl border border-white/10 bg-black/30 p-3 text-xs text-slate-300">
                        <p>Action: {result.agent_trace.selected_action}</p>
                        <p>
                          Confidence: {(result.agent_trace.confidence * 100).toFixed(1)}% | Value: {result.agent_trace.value.toFixed(3)}
                        </p>
                        {result.agent_trace.safe_mode && (
                          <p className="text-amber-300">
                            Safety Gate: {result.agent_trace.safe_reason || "enabled"} (raw: {result.agent_trace.raw_action})
                          </p>
                        )}
                        <p>Policy: {result.agent_trace.action_probs_text}</p>
                        <p>Expected Reward: {result.agent_trace.expected_rewards_text}</p>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {!result && !loading && !error && (
                <div className="mx-auto mt-10 flex max-w-2xl flex-col items-center rounded-2xl border border-white/10 bg-black/20 px-6 py-8 text-center">
                  <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-white/5 text-slate-200">
                    <GpuIcon />
                  </div>
                  <p className="text-sm text-slate-300">검색 결과가 이 영역에 표시됩니다</p>
                  <p className="mt-1 text-xs text-slate-500">좌상단 정렬 대신 중앙 레이아웃으로 재배치되었습니다.</p>
                </div>
              )}
          </section>
        </div>
      </main>

      <footer className="relative z-10 border-t border-white/10 px-4 py-4 text-center text-xs text-slate-500">
        <p>© 2026 GPU Advisor · FastAPI + Next.js</p>
      </footer>
    </div>
  );
}
