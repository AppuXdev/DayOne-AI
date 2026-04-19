"use client";

import { useEffect, useMemo, useState } from "react";
import type { CSSProperties } from "react";
import { useRouter } from "next/navigation";
import { apiRequest, clearAuthSession, decodeJwt, getStoredToken } from "../../lib/api";
import type { JwtPayload } from "../../lib/api";

type TraceListItem = {
  id: string;
  tenant_id: string;
  query: string;
  trace: TracePayload;
  created_at: string;
};

type TracePayload = {
  trace_id?: string;
  query?: string;
  tenant_id?: string;
  query_type?: string;
  route?: string;
  retrieval?: {
    bm25_topk?: Array<Record<string, unknown>>;
    dense_topk?: Array<Record<string, unknown>>;
    rrf_fused?: Array<Record<string, unknown>>;
    reranked?: Array<Record<string, unknown>>;
  };
  final_context?: Array<Record<string, unknown>>;
  verification?: Record<string, unknown>;
  abstained?: boolean;
  abstain_reason?: string | null;
  latency_ms?: number;
  confidence?: number;
};

type EvalAbstentionModeMetrics = {
  abstention_precision: number;
  abstention_recall: number;
  abstention_f1: number;
  false_abstentions: number;
  false_abstention_rate: number;
};

type EvalAbstentionMetricsResponse = {
  source_file: string;
  generated_at: string;
  release_tag?: string | null;
  model_version?: string | null;
  git_commit?: string | null;
  modes: Record<string, EvalAbstentionModeMetrics>;
};

type EvalAbstentionMetricsHistoryResponse = {
  items: EvalAbstentionMetricsResponse[];
};

type StageName = "bm25" | "dense" | "rrf" | "rerank";

type RankMovementRow = {
  chunkId: string;
  source: string;
  snippet: string;
  bm25Rank?: number;
  denseRank?: number;
  rrfRank?: number;
  rerankRank?: number;
};

type TrendState = "improving" | "degrading" | "flat" | "insufficient";
type StabilityState = "stable" | "moderate" | "noisy" | "insufficient";

type StatusDescriptor = {
  key: string;
  label: string;
  color: string;
  background: string;
  border: string;
  severity: number;
  isIssue: boolean;
};

type ModeAssessment = {
  mode: string;
  label: string;
  metrics: EvalAbstentionModeMetrics;
  previous?: EvalAbstentionModeMetrics;
  series: number[];
  direction: { state: TrendState; label: string; color: string; deltaPp?: string };
  stdDev?: number;
  volatility: { state: StabilityState; label: string; color: string };
  status: StatusDescriptor;
  tooltip: string;
  artifactSourceFile?: string;
};

function statusBadgeStyle(kind: "green" | "red" | "yellow") {
  if (kind === "green") {
    return {
      color: "#86efac",
      background: "rgba(34, 197, 94, 0.14)",
      border: "1px solid rgba(34, 197, 94, 0.25)",
    };
  }
  if (kind === "red") {
    return {
      color: "#fda4af",
      background: "rgba(244, 63, 94, 0.14)",
      border: "1px solid rgba(244, 63, 94, 0.25)",
    };
  }
  return {
    color: "#fde68a",
    background: "rgba(234, 179, 8, 0.14)",
    border: "1px solid rgba(234, 179, 8, 0.25)",
  };
}

function asPrettyJson(value: unknown): string {
  try {
    return JSON.stringify(value ?? {}, null, 2);
  } catch {
    return "{}";
  }
}

function getNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function getText(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function chunkKey(item: Record<string, unknown>): string {
  const source = getText(item.source) || "unknown";
  const page = String(item.page ?? "-");
  const row = String(item.row ?? "-");
  const snippet = getText(item.snippet).slice(0, 80);
  return `${source}|${page}|${row}|${snippet}`;
}

function normalizeStageItems(stage: Array<Record<string, unknown>>): Array<Record<string, unknown>> {
  return stage
    .map((item) => ({
      ...item,
      rank: getNumber(item.rank),
      source: getText(item.source) || "unknown",
      snippet: getText(item.snippet),
    }))
    .filter((item) => typeof item.rank === "number");
}

function buildRankMovementRows(trace: TracePayload | undefined): RankMovementRow[] {
  const bm25 = normalizeStageItems(trace?.retrieval?.bm25_topk ?? []);
  const dense = normalizeStageItems(trace?.retrieval?.dense_topk ?? []);
  const rrf = normalizeStageItems(trace?.retrieval?.rrf_fused ?? []);
  const rerank = normalizeStageItems(trace?.retrieval?.reranked ?? []);

  const all = new Map<string, RankMovementRow>();

  const upsert = (stage: StageName, item: Record<string, unknown>) => {
    const key = chunkKey(item);
    const existing = all.get(key) ?? {
      chunkId: key,
      source: getText(item.source) || "unknown",
      snippet: getText(item.snippet),
    };
    const rank = getNumber(item.rank);
    if (stage === "bm25") existing.bm25Rank = rank;
    if (stage === "dense") existing.denseRank = rank;
    if (stage === "rrf") existing.rrfRank = rank;
    if (stage === "rerank") existing.rerankRank = rank;
    all.set(key, existing);
  };

  bm25.forEach((item) => upsert("bm25", item));
  dense.forEach((item) => upsert("dense", item));
  rrf.forEach((item) => upsert("rrf", item));
  rerank.forEach((item) => upsert("rerank", item));

  const rows = Array.from(all.values());
  rows.sort((a, b) => {
    const rankA = a.rerankRank ?? a.rrfRank ?? a.denseRank ?? a.bm25Rank ?? 9999;
    const rankB = b.rerankRank ?? b.rrfRank ?? b.denseRank ?? b.bm25Rank ?? 9999;
    return rankA - rankB;
  });
  return rows;
}

function movementDelta(previous?: number, current?: number): number | undefined {
  if (typeof previous !== "number" || typeof current !== "number") return undefined;
  return previous - current;
}

function movementStyle(delta?: number): CSSProperties {
  if (typeof delta !== "number") {
    return { color: "#94a3b8", background: "rgba(100, 116, 139, 0.12)", border: "1px solid rgba(100, 116, 139, 0.2)" };
  }
  if (delta > 0) {
    return { color: "#86efac", background: "rgba(34, 197, 94, 0.12)", border: "1px solid rgba(34, 197, 94, 0.2)" };
  }
  if (delta < 0) {
    return { color: "#fda4af", background: "rgba(244, 63, 94, 0.12)", border: "1px solid rgba(244, 63, 94, 0.2)" };
  }
  return { color: "#cbd5e1", background: "rgba(148, 163, 184, 0.15)", border: "1px solid rgba(148, 163, 184, 0.25)" };
}

function isInterestingRow(row: RankMovementRow): boolean {
  const deltas = [
    movementDelta(row.bm25Rank, row.denseRank),
    movementDelta(row.denseRank, row.rrfRank),
    movementDelta(row.rrfRank, row.rerankRank),
  ].filter((value): value is number => typeof value === "number");

  if (deltas.length === 0) return false;
  return deltas.some((delta) => Math.abs(delta) >= 2);
}

function formatDelta(delta?: number): string {
  if (typeof delta !== "number") return "-";
  if (delta > 0) return `+${delta}`;
  return `${delta}`;
}

function sparklineCoordinates(values: number[], width = 110, height = 32, padding = 3): Array<{ x: number; y: number }> {
  if (values.length === 0) return [];
  if (values.length === 1) {
    const y = height - padding - Math.min(1, Math.max(0, values[0])) * (height - padding * 2);
    return [
      { x: padding, y },
      { x: width - padding, y },
    ];
  }

  const usableWidth = width - padding * 2;
  const usableHeight = height - padding * 2;
  return values.map((value, index) => {
    const x = padding + (index / (values.length - 1)) * usableWidth;
    const clamped = Math.min(1, Math.max(0, value));
    const y = height - padding - clamped * usableHeight;
    return { x, y };
  });
}

function sparklinePoints(points: Array<{ x: number; y: number }>): string {
  return points.map((point) => `${point.x.toFixed(2)},${point.y.toFixed(2)}`).join(" ");
}

function modeF1Series(mode: string, history: EvalAbstentionMetricsResponse[]): number[] {
  const oldestToNewest = [...history].reverse();
  const series = oldestToNewest
    .map((item) => item.modes?.[mode]?.abstention_f1)
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  return series;
}

function modeLabel(mode: string): string {
  const labels: Record<string, string> = {
    reranker_on: "pgvector + reranker",
    reranker_off: "pgvector no reranker",
    faiss_baseline: "faiss baseline",
    no_reranker: "no reranker",
  };
  return labels[mode] ?? mode.replaceAll("_", " ");
}

function trendDirection(series: number[], minRuns = 3): { state: TrendState; label: string; color: string; deltaPp?: string } {
  if (series.length < minRuns) {
    return { state: "insufficient", label: `insufficient data (${series.length}/${minRuns})`, color: "#64748b" };
  }
  const first = series[0];
  const last = series[series.length - 1];
  const delta = last - first;
  const deltaPp = `${delta > 0 ? "+" : ""}${(delta * 100).toFixed(1)}pp`;
  if (Math.abs(delta) < 0.0001) {
    return { state: "flat", label: "flat", color: "#cbd5e1", deltaPp };
  }
  if (delta > 0) {
    return { state: "improving", label: "improving", color: "#86efac", deltaPp };
  }
  return { state: "degrading", label: "degrading", color: "#fda4af", deltaPp };
}

function f1DeltaStdDev(series: number[]): number | undefined {
  if (series.length < 3) return undefined;
  const deltas: number[] = [];
  for (let i = 1; i < series.length; i += 1) {
    deltas.push(series[i] - series[i - 1]);
  }
  if (deltas.length < 2) return undefined;
  const mean = deltas.reduce((sum, value) => sum + value, 0) / deltas.length;
  const variance = deltas.reduce((sum, value) => sum + (value - mean) ** 2, 0) / deltas.length;
  return Math.sqrt(variance);
}

function volatilityLabel(stdDev: number | undefined): { state: StabilityState; label: string; color: string } {
  if (typeof stdDev !== "number") {
    return { state: "insufficient", label: "insufficient", color: "#64748b" };
  }
  const pp = stdDev * 100;
  if (pp <= 1.0) {
    return { state: "stable", label: "stable", color: "#86efac" };
  }
  if (pp <= 2.5) {
    return { state: "moderate", label: "moderate", color: "#fde68a" };
  }
  return { state: "noisy", label: "noisy", color: "#fda4af" };
}

function combinedStatus(
  trend: { state: TrendState; deltaPp?: string },
  stability: { state: StabilityState },
): StatusDescriptor {
  if (trend.state === "insufficient" || stability.state === "insufficient") {
    return {
      key: "insufficient_data",
      label: "insufficient data",
      color: "#cbd5e1",
      background: "rgba(148, 163, 184, 0.15)",
      border: "1px solid rgba(148, 163, 184, 0.3)",
      severity: 1,
      isIssue: false,
    };
  }

  // Regression alert: consistent degradation with low variance.
  if (trend.state === "degrading" && stability.state === "stable") {
    return {
      key: "regression_gate_risk",
      label: "regression (gate risk)",
      color: "#fecaca",
      background: "rgba(239, 68, 68, 0.2)",
      border: "1px solid rgba(239, 68, 68, 0.45)",
      severity: 5,
      isIssue: true,
    };
  }

  if (trend.state === "improving" && stability.state === "stable") {
    return {
      key: "improving_stable",
      label: "improving",
      color: "#86efac",
      background: "rgba(34, 197, 94, 0.14)",
      border: "1px solid rgba(34, 197, 94, 0.32)",
      severity: 0,
      isIssue: false,
    };
  }
  if (trend.state === "improving" && stability.state === "moderate") {
    return {
      key: "improving_moderate",
      label: "improving (moderate variance)",
      color: "#86efac",
      background: "rgba(34, 197, 94, 0.14)",
      border: "1px solid rgba(34, 197, 94, 0.32)",
      severity: 1,
      isIssue: false,
    };
  }
  if (trend.state === "improving" && stability.state === "noisy") {
    return {
      key: "improving_unstable",
      label: "improving but unstable",
      color: "#fde68a",
      background: "rgba(234, 179, 8, 0.16)",
      border: "1px solid rgba(234, 179, 8, 0.35)",
      severity: 2,
      isIssue: true,
    };
  }

  if (trend.state === "degrading" && stability.state === "moderate") {
    return {
      key: "degrading_moderate",
      label: "degrading (moderate variance)",
      color: "#fca5a5",
      background: "rgba(239, 68, 68, 0.16)",
      border: "1px solid rgba(239, 68, 68, 0.35)",
      severity: 4,
      isIssue: true,
    };
  }
  if (trend.state === "degrading" && stability.state === "noisy") {
    return {
      key: "degrading_unstable",
      label: "degrading and unstable",
      color: "#fde68a",
      background: "rgba(234, 179, 8, 0.16)",
      border: "1px solid rgba(234, 179, 8, 0.35)",
      severity: 4,
      isIssue: true,
    };
  }

  if (trend.state === "flat" && stability.state === "stable") {
    return {
      key: "flat_stable",
      label: "stable",
      color: "#e2e8f0",
      background: "rgba(148, 163, 184, 0.15)",
      border: "1px solid rgba(148, 163, 184, 0.3)",
      severity: 0,
      isIssue: false,
    };
  }
  if (trend.state === "flat" && stability.state === "noisy") {
    return {
      key: "flat_noisy",
      label: "unstable",
      color: "#fde68a",
      background: "rgba(234, 179, 8, 0.16)",
      border: "1px solid rgba(234, 179, 8, 0.35)",
      severity: 2,
      isIssue: true,
    };
  }

  return {
    key: "fallback_stable",
    label: "stable",
    color: "#e2e8f0",
    background: "rgba(148, 163, 184, 0.15)",
    border: "1px solid rgba(148, 163, 184, 0.3)",
    severity: 0,
    isIssue: false,
  };
}

function buildModeAssessments(history: EvalAbstentionMetricsResponse[]): ModeAssessment[] {
  const latestModes = history[0]?.modes ?? {};
  const previousModes = history[1]?.modes ?? {};

  return Object.entries(latestModes)
    .map(([mode, metrics]) => {
      const previous = previousModes[mode];
      const series = modeF1Series(mode, history);
      const direction = trendDirection(series, 3);
      const stdDev = f1DeltaStdDev(series);
      const volatility = volatilityLabel(stdDev);
      const status = combinedStatus(direction, volatility);
      const artifactSourceFile = history.find((item) => item.modes?.[mode])?.source_file;
      const tooltip = `Trend: ${direction.label}${direction.deltaPp ? ` (${direction.deltaPp})` : ""}\nVolatility: ${volatility.label}${typeof stdDev === "number" ? ` (${(stdDev * 100).toFixed(2)}pp std)` : ""}\nLast ${series.length} runs: [${series.map((v) => v.toFixed(2)).join(", ")}]`;

      return {
        mode,
        label: modeLabel(mode),
        metrics,
        previous,
        series,
        direction,
        stdDev,
        volatility,
        status,
        tooltip,
        artifactSourceFile,
      } satisfies ModeAssessment;
    })
    .sort((a, b) => {
      if (b.status.severity !== a.status.severity) return b.status.severity - a.status.severity;
      return a.label.localeCompare(b.label);
    });
}

export default function AdminDebugPanel() {
  const router = useRouter();
  const [token, setToken] = useState<string | null>(null);
  const [profile, setProfile] = useState<JwtPayload | null>(null);
  const [items, setItems] = useState<TraceListItem[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [queryTypeFilter, setQueryTypeFilter] = useState<string>("");
  const [abstainedFilter, setAbstainedFilter] = useState<string>("all");
  const [lowConfidenceFilter, setLowConfidenceFilter] = useState(false);
  const [showTopKEvolution, setShowTopKEvolution] = useState(true);
  const [onlyInterestingMovements, setOnlyInterestingMovements] = useState(false);
  const [showOnlyIssues, setShowOnlyIssues] = useState(false);
  const [focusedMode, setFocusedMode] = useState<string | null>(null);
  const [evalMetrics, setEvalMetrics] = useState<EvalAbstentionMetricsResponse | null>(null);
  const [evalHistory, setEvalHistory] = useState<EvalAbstentionMetricsResponse[]>([]);
  const [evalError, setEvalError] = useState<string | null>(null);

  useEffect(() => {
    const stored = getStoredToken();
    if (!stored) {
      router.replace("/login");
      return;
    }
    const decoded = decodeJwt(stored);
    if (!decoded || decoded.role !== "admin") {
      router.replace(decoded?.role === "employee" ? "/chat" : "/login");
      return;
    }
    setToken(stored);
    setProfile(decoded);
  }, [router]);

  async function loadTraces() {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams();
      params.set("limit", "50");
      if (queryTypeFilter) params.set("query_type", queryTypeFilter);
      if (abstainedFilter !== "all") params.set("abstained", abstainedFilter);
      if (lowConfidenceFilter) params.set("low_confidence", "true");

      const data = await apiRequest<TraceListItem[]>({
        url: `/api/admin/traces?${params.toString()}`,
        method: "GET",
      });
      setItems(data);
      setSelectedId((current) => {
        if (current && data.some((trace) => trace.id === current)) return current;
        return data.length > 0 ? data[0].id : null;
      });
    } catch (fetchError) {
      setError(fetchError instanceof Error ? fetchError.message : "Failed to load traces.");
    } finally {
      setLoading(false);
    }
  }

  async function loadEvalAbstentionMetrics() {
    setEvalError(null);
    try {
      const [latest, history] = await Promise.all([
        apiRequest<EvalAbstentionMetricsResponse>({
          url: "/api/admin/eval/abstention-metrics",
          method: "GET",
        }),
        apiRequest<EvalAbstentionMetricsHistoryResponse>({
          url: "/api/admin/eval/abstention-metrics/history?limit=5",
          method: "GET",
        }),
      ]);
      setEvalMetrics(latest);
      setEvalHistory(history.items ?? []);
    } catch (fetchError) {
      setEvalMetrics(null);
      setEvalHistory([]);
      setEvalError(fetchError instanceof Error ? fetchError.message : "Failed to load abstention metrics.");
    }
  }

  function trendValue(current: number, previous: number): string {
    const delta = current - previous;
    if (Math.abs(delta) < 0.0001) return "0.0pp";
    const sign = delta > 0 ? "+" : "";
    return `${sign}${(delta * 100).toFixed(1)}pp`;
  }

  function trendColor(current: number, previous: number): string {
    const delta = current - previous;
    if (Math.abs(delta) < 0.0001) return "#cbd5e1";
    return delta > 0 ? "#86efac" : "#fda4af";
  }

  function formatModeLabel(mode: string): string {
    return modeLabel(mode);
  }

  useEffect(() => {
    if (!token) return;
    void loadTraces();
    void loadEvalAbstentionMetrics();
  }, [token]);

  const selected = useMemo(
    () => items.find((item) => item.id === selectedId) ?? null,
    [items, selectedId],
  );

  const rankMovementRows = useMemo(() => {
    const rows = buildRankMovementRows(selected?.trace);
    if (!onlyInterestingMovements) return rows;
    return rows.filter((row) => isInterestingRow(row));
  }, [onlyInterestingMovements, selected]);

  const modeAssessments = useMemo(() => buildModeAssessments(evalHistory), [evalHistory]);
  const issueAssessments = useMemo(
    () => modeAssessments.filter((assessment) => assessment.status.isIssue),
    [modeAssessments],
  );
  const displayedAssessments = useMemo(
    () => (showOnlyIssues ? issueAssessments : modeAssessments),
    [showOnlyIssues, issueAssessments, modeAssessments],
  );

  const sinceTag = useMemo(() => {
    const previous = evalHistory[1];
    if (!previous) return null;
    if (previous.release_tag) return `since ${previous.release_tag}`;
    if (previous.model_version) return `since ${previous.model_version}`;
    if (previous.git_commit) return `since ${previous.git_commit.slice(0, 8)}`;
    return `since ${new Date(previous.generated_at).toLocaleString()}`;
  }, [evalHistory]);

  function focusMode(mode: string) {
    setFocusedMode(mode);
    if (showOnlyIssues) {
      const isIssue = issueAssessments.some((assessment) => assessment.mode === mode);
      if (!isIssue) return;
    }
    if (typeof document !== "undefined") {
      document.getElementById(`mode-row-${mode}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }

  function signOut() {
    clearAuthSession();
    router.replace("/login");
  }

  if (!token) {
    return <main style={{ minHeight: "100vh", background: "#020617" }} />;
  }

  return (
    <main style={{ minHeight: "100vh", background: "#020617", color: "#e2e8f0", fontFamily: "ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif" }}>
      <div style={{ maxWidth: 1400, margin: "0 auto", padding: "1.25rem" }}>
        <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem", gap: "0.75rem", flexWrap: "wrap" }}>
          <div>
            <p style={{ margin: 0, color: "#38bdf8", fontSize: "0.8rem", textTransform: "uppercase", letterSpacing: "0.12em", fontWeight: 700 }}>
              Admin Debug
            </p>
            <h1 style={{ margin: "0.35rem 0 0", fontSize: "1.8rem", color: "#f8fafc" }}>
              Retrieval Trace Viewer
            </h1>
            <p style={{ margin: "0.4rem 0 0", color: "#94a3b8", fontSize: "0.9rem" }}>
              Tenant: <span style={{ color: "#cbd5e1" }}>{profile?.organization ?? "unknown"}</span>
            </p>
          </div>
          <div style={{ display: "flex", gap: "0.6rem", flexWrap: "wrap" }}>
            <button
              onClick={() => router.push("/admin")}
              style={{ borderRadius: 10, border: "1px solid #334155", background: "#0f172a", color: "#cbd5e1", padding: "0.58rem 0.9rem", cursor: "pointer" }}
            >
              Back to Admin
            </button>
            <button
              onClick={signOut}
              style={{ borderRadius: 10, border: "1px solid #475569", background: "#020617", color: "#94a3b8", padding: "0.58rem 0.9rem", cursor: "pointer" }}
            >
              Sign out
            </button>
          </div>
        </header>

        <section style={{ border: "1px solid rgba(51, 65, 85, 0.75)", borderRadius: 14, background: "rgba(15, 23, 42, 0.55)", padding: "0.85rem", marginBottom: "1rem", display: "flex", gap: "0.7rem", flexWrap: "wrap", alignItems: "center" }}>
          <select value={queryTypeFilter} onChange={(event) => setQueryTypeFilter(event.target.value)} style={{ borderRadius: 8, border: "1px solid #334155", background: "#0f172a", color: "#e2e8f0", padding: "0.5rem 0.65rem" }}>
            <option value="">All query types</option>
            <option value="factual">factual</option>
            <option value="ambiguous">ambiguous</option>
            <option value="multi_hop">multi_hop</option>
            <option value="exception">exception</option>
          </select>
          <select value={abstainedFilter} onChange={(event) => setAbstainedFilter(event.target.value)} style={{ borderRadius: 8, border: "1px solid #334155", background: "#0f172a", color: "#e2e8f0", padding: "0.5rem 0.65rem" }}>
            <option value="all">All abstention states</option>
            <option value="true">Abstained only</option>
            <option value="false">Non-abstained only</option>
          </select>
          <label style={{ display: "inline-flex", alignItems: "center", gap: "0.45rem", color: "#cbd5e1", fontSize: "0.9rem" }}>
            <input type="checkbox" checked={lowConfidenceFilter} onChange={(event) => setLowConfidenceFilter(event.target.checked)} />
            Low confidence only
          </label>
          <label style={{ display: "inline-flex", alignItems: "center", gap: "0.45rem", color: "#cbd5e1", fontSize: "0.9rem" }}>
            <input type="checkbox" checked={showTopKEvolution} onChange={(event) => setShowTopKEvolution(event.target.checked)} />
            Show Top-K Evolution
          </label>
          <label style={{ display: "inline-flex", alignItems: "center", gap: "0.45rem", color: "#cbd5e1", fontSize: "0.9rem" }}>
            <input type="checkbox" checked={onlyInterestingMovements} onChange={(event) => setOnlyInterestingMovements(event.target.checked)} />
            Only interesting movements
          </label>
          <button
            onClick={() => void loadTraces()}
            disabled={loading}
            style={{ borderRadius: 8, border: "none", background: "#0ea5e9", color: "#020617", padding: "0.5rem 0.85rem", fontWeight: 700, cursor: "pointer" }}
          >
            {loading ? "Refreshing..." : "Refresh"}
          </button>
        </section>

        {error ? (
          <div style={{ border: "1px solid rgba(244, 63, 94, 0.35)", borderRadius: 10, background: "rgba(244, 63, 94, 0.12)", color: "#fecdd3", padding: "0.75rem 0.9rem", marginBottom: "1rem" }}>
            {error}
          </div>
        ) : null}

        <section style={{ border: "1px solid rgba(51, 65, 85, 0.8)", borderRadius: 14, background: "rgba(15, 23, 42, 0.75)", padding: "0.85rem", marginBottom: "1rem" }}>
          <p style={{ margin: 0, color: "#38bdf8", fontSize: "0.76rem", textTransform: "uppercase", letterSpacing: "0.1em", fontWeight: 700 }}>
            Abstention Metrics
          </p>
          {evalMetrics ? (
            <>
              <p style={{ margin: "0.45rem 0 0", color: "#94a3b8", fontSize: "0.8rem" }}>
                Source: <span style={{ color: "#cbd5e1" }}>{evalMetrics.source_file}</span> · Updated {new Date(evalMetrics.generated_at).toLocaleString()}
              </p>
              <div style={{ marginTop: "0.7rem", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "0.7rem" }}>
                {Object.entries(evalMetrics.modes).map(([mode, metrics]) => (
                  <div key={mode} style={{ borderRadius: 10, border: "1px solid rgba(51, 65, 85, 0.6)", background: "rgba(2, 6, 23, 0.45)", padding: "0.7rem" }}>
                    <p style={{ margin: 0, color: "#a5f3fc", fontSize: "0.8rem", fontWeight: 700 }}>{formatModeLabel(mode)}</p>
                    <div style={{ marginTop: "0.5rem", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.45rem", fontSize: "0.8rem" }}>
                      <span style={{ color: "#94a3b8" }}>Precision</span>
                      <span style={{ color: "#f8fafc", fontWeight: 600 }}>{(metrics.abstention_precision * 100).toFixed(1)}%</span>
                      <span style={{ color: "#94a3b8" }}>Recall</span>
                      <span style={{ color: "#f8fafc", fontWeight: 600 }}>{(metrics.abstention_recall * 100).toFixed(1)}%</span>
                      <span style={{ color: "#94a3b8" }}>F1</span>
                      <span style={{ color: "#f8fafc", fontWeight: 600 }}>{(metrics.abstention_f1 * 100).toFixed(1)}%</span>
                      <span style={{ color: "#94a3b8" }}>False abstentions</span>
                      <span style={{ color: "#f8fafc", fontWeight: 600 }}>{metrics.false_abstentions}</span>
                      <span style={{ color: "#94a3b8" }}>False abstention rate</span>
                      <span style={{ color: "#f8fafc", fontWeight: 600 }}>{(metrics.false_abstention_rate * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                ))}
              </div>
              <p style={{ margin: "0.8rem 0 0", color: "#93c5fd", fontSize: "0.78rem", fontWeight: 700, letterSpacing: "0.04em", textTransform: "uppercase" }}>
                Recent Trend (latest vs previous)
              </p>
              <div style={{ marginTop: "0.55rem", display: "flex", justifyContent: "space-between", alignItems: "center", gap: "0.75rem", flexWrap: "wrap" }}>
                <label style={{ display: "inline-flex", alignItems: "center", gap: "0.45rem", color: "#cbd5e1", fontSize: "0.83rem" }}>
                  <input type="checkbox" checked={showOnlyIssues} onChange={(event) => setShowOnlyIssues(event.target.checked)} />
                  Show only issues
                </label>
                {sinceTag ? <span style={{ color: "#94a3b8", fontSize: "0.78rem" }}>{sinceTag}</span> : null}
              </div>

              {issueAssessments.length > 0 ? (
                <div style={{ marginTop: "0.55rem", display: "flex", gap: "0.45rem", flexWrap: "wrap" }}>
                  {issueAssessments.slice(0, 5).map((assessment) => (
                    <div
                      key={`issue-${assessment.mode}`}
                      title={assessment.tooltip}
                      style={{
                        display: "inline-flex",
                        alignItems: "center",
                        gap: "0.35rem",
                        borderRadius: 999,
                        padding: "0.2rem 0.4rem 0.2rem 0.2rem",
                        color: assessment.status.color,
                        background: assessment.status.background,
                        border: assessment.status.border,
                      }}
                    >
                      <button
                        onClick={() => focusMode(assessment.mode)}
                        style={{
                          borderRadius: 999,
                          padding: "0.12rem 0.42rem",
                          fontSize: "0.74rem",
                          fontWeight: 700,
                          color: assessment.status.color,
                          background: "transparent",
                          border: "none",
                          cursor: "pointer",
                        }}
                      >
                        {assessment.label}: {assessment.status.label}
                      </button>
                      {assessment.status.key === "regression_gate_risk" && assessment.artifactSourceFile ? (
                        <a
                          href={`/api/admin/eval/artifact?source_file=${encodeURIComponent(assessment.artifactSourceFile)}`}
                          target="_blank"
                          rel="noreferrer"
                          style={{ color: "#fde68a", fontSize: "0.72rem", textDecoration: "underline" }}
                        >
                          view eval
                        </a>
                      ) : null}
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ marginTop: "0.55rem", borderRadius: 10, border: "1px solid rgba(34, 197, 94, 0.3)", background: "rgba(34, 197, 94, 0.12)", color: "#86efac", padding: "0.5rem 0.65rem", fontSize: "0.84rem", fontWeight: 700 }}>
                  All modes stable
                </div>
              )}

              {evalHistory.length > 0 ? (
                <div style={{ marginTop: "0.5rem", borderRadius: 10, border: "1px solid rgba(51, 65, 85, 0.6)", overflow: "hidden" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.78rem" }}>
                    <thead>
                      <tr style={{ background: "rgba(15, 23, 42, 0.8)", color: "#94a3b8" }}>
                        <th style={{ textAlign: "left", padding: "0.55rem", borderBottom: "1px solid rgba(51, 65, 85, 0.6)" }}>Mode</th>
                        <th style={{ textAlign: "left", padding: "0.55rem", borderBottom: "1px solid rgba(51, 65, 85, 0.6)" }}>Status</th>
                        <th style={{ textAlign: "left", padding: "0.55rem", borderBottom: "1px solid rgba(51, 65, 85, 0.6)" }}>Precision</th>
                        <th style={{ textAlign: "left", padding: "0.55rem", borderBottom: "1px solid rgba(51, 65, 85, 0.6)" }}>Recall</th>
                        <th style={{ textAlign: "left", padding: "0.55rem", borderBottom: "1px solid rgba(51, 65, 85, 0.6)" }}>F1</th>
                        <th style={{ textAlign: "left", padding: "0.55rem", borderBottom: "1px solid rgba(51, 65, 85, 0.6)" }}>F1 vs prev</th>
                        <th style={{ textAlign: "left", padding: "0.55rem", borderBottom: "1px solid rgba(51, 65, 85, 0.6)" }}>Trend</th>
                        <th style={{ textAlign: "left", padding: "0.55rem", borderBottom: "1px solid rgba(51, 65, 85, 0.6)" }}>Volatility</th>
                        <th style={{ textAlign: "left", padding: "0.55rem", borderBottom: "1px solid rgba(51, 65, 85, 0.6)" }}>F1 (5-run)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {displayedAssessments.map((assessment) => {
                        const coordinates = sparklineCoordinates(assessment.series);
                        const points = sparklinePoints(coordinates);
                        const latestPoint = coordinates[coordinates.length - 1];
                        const isFocused = focusedMode === assessment.mode;
                        return (
                          <tr id={`mode-row-${assessment.mode}`} key={assessment.mode} style={{ borderBottom: "1px solid rgba(30, 41, 59, 0.75)", background: isFocused ? "rgba(56, 189, 248, 0.12)" : "transparent" }}>
                            <td style={{ padding: "0.55rem", color: "#e2e8f0", fontWeight: 600 }}>{assessment.label}</td>
                            <td style={{ padding: "0.55rem" }}>
                              <span title={assessment.tooltip} style={{ display: "inline-block", borderRadius: 999, padding: "0.16rem 0.6rem", fontSize: "0.72rem", fontWeight: 700, color: assessment.status.color, background: assessment.status.background, border: assessment.status.border }}>
                                {assessment.status.label}
                              </span>
                            </td>
                            <td style={{ padding: "0.55rem", color: "#f8fafc" }}>{(assessment.metrics.abstention_precision * 100).toFixed(1)}%</td>
                            <td style={{ padding: "0.55rem", color: "#f8fafc" }}>{(assessment.metrics.abstention_recall * 100).toFixed(1)}%</td>
                            <td style={{ padding: "0.55rem", color: "#f8fafc" }}>{(assessment.metrics.abstention_f1 * 100).toFixed(1)}%</td>
                            <td style={{ padding: "0.55rem", color: assessment.previous ? trendColor(assessment.metrics.abstention_f1, assessment.previous.abstention_f1) : "#64748b", fontWeight: 700 }}>
                              {assessment.previous ? trendValue(assessment.metrics.abstention_f1, assessment.previous.abstention_f1) : "-"}
                            </td>
                            <td style={{ padding: "0.55rem", color: assessment.direction.color, fontWeight: 700 }}>
                              {assessment.direction.label}
                              {assessment.direction.deltaPp ? ` (${assessment.direction.deltaPp})` : ""}
                            </td>
                            <td style={{ padding: "0.55rem", color: assessment.volatility.color, fontWeight: 700 }}>
                              {assessment.volatility.label}
                              {typeof assessment.stdDev === "number" ? ` (${(assessment.stdDev * 100).toFixed(2)}pp)` : ""}
                            </td>
                            <td style={{ padding: "0.45rem 0.55rem", color: "#f8fafc" }}>
                              {assessment.series.length >= 3 ? (
                                <svg width="110" height="32" viewBox="0 0 110 32" role="img" aria-label={`${assessment.mode} F1 trend`}>
                                  <title>{assessment.series.map((value, index) => `Run ${index + 1}: F1 = ${value.toFixed(2)}`).join("\n")}</title>
                                  <rect x="0" y="0" width="110" height="32" rx="5" fill="rgba(2, 6, 23, 0.6)" />
                                  <polyline fill="none" stroke="rgba(56, 189, 248, 0.95)" strokeWidth="1.8" points={points} />
                                  {coordinates.map((point, index) => (
                                    <circle key={`${assessment.mode}-pt-${index}`} cx={point.x} cy={point.y} r={index === coordinates.length - 1 ? 2.4 : 1.6} fill="rgba(125, 211, 252, 1)">
                                      <title>{`Run ${index + 1}: F1 = ${assessment.series[index].toFixed(2)}`}</title>
                                    </circle>
                                  ))}
                                  {latestPoint ? <circle cx={latestPoint.x} cy={latestPoint.y} r="2.8" fill="rgba(186, 230, 253, 0.25)" /> : null}
                                </svg>
                              ) : (
                                <span style={{ color: "#64748b" }}>Need at least 3 runs</span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                      {displayedAssessments.length === 0 ? (
                        <tr>
                          <td colSpan={9} style={{ padding: "0.7rem", color: "#64748b" }}>
                            No rows match current filter.
                          </td>
                        </tr>
                      ) : null}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p style={{ margin: "0.55rem 0 0", color: "#64748b", fontSize: "0.85rem" }}>
                  No recent eval history found.
                </p>
              )}
            </>
          ) : (
            <p style={{ margin: "0.55rem 0 0", color: "#64748b", fontSize: "0.85rem" }}>
              {evalError ?? "No evaluation artifact loaded yet. Run eval.py to generate metrics."}
            </p>
          )}
        </section>

        <section style={{ display: "grid", gridTemplateColumns: "minmax(320px, 430px) 1fr", gap: "0.9rem" }}>
          <aside style={{ border: "1px solid rgba(51, 65, 85, 0.8)", borderRadius: 14, background: "rgba(15, 23, 42, 0.75)", minHeight: 640, overflow: "hidden" }}>
            <div style={{ padding: "0.75rem 0.85rem", borderBottom: "1px solid rgba(51, 65, 85, 0.6)", color: "#cbd5e1", fontWeight: 700, fontSize: "0.9rem" }}>
              Recent Queries ({items.length})
            </div>
            <div style={{ maxHeight: 760, overflowY: "auto" }}>
              {items.map((item) => {
                const abstained = Boolean(item.trace?.abstained);
                const selectedRow = item.id === selectedId;
                return (
                  <button
                    key={item.id}
                    onClick={() => setSelectedId(item.id)}
                    style={{
                      width: "100%",
                      textAlign: "left",
                      border: "none",
                      borderBottom: "1px solid rgba(30, 41, 59, 0.75)",
                      background: selectedRow ? "rgba(56, 189, 248, 0.1)" : "transparent",
                      color: "inherit",
                      padding: "0.75rem 0.85rem",
                      cursor: "pointer",
                    }}
                  >
                    <p style={{ margin: 0, color: "#f8fafc", fontSize: "0.88rem", fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {item.query}
                    </p>
                    <div style={{ marginTop: "0.45rem", display: "flex", gap: "0.4rem", flexWrap: "wrap", alignItems: "center" }}>
                      <span style={{ borderRadius: 999, padding: "0.12rem 0.5rem", fontSize: "0.72rem", color: "#93c5fd", border: "1px solid rgba(59, 130, 246, 0.3)", background: "rgba(59, 130, 246, 0.12)" }}>
                        {item.trace?.query_type ?? "unknown"}
                      </span>
                      <span style={{ borderRadius: 999, padding: "0.12rem 0.5rem", fontSize: "0.72rem", color: "#a5f3fc", border: "1px solid rgba(34, 211, 238, 0.3)", background: "rgba(34, 211, 238, 0.12)" }}>
                        {item.trace?.route ?? "route?"}
                      </span>
                      <span style={{ borderRadius: 999, padding: "0.12rem 0.5rem", fontSize: "0.72rem", ...(abstained ? statusBadgeStyle("red") : statusBadgeStyle("green")) }}>
                        {abstained ? "abstained" : "answered"}
                      </span>
                    </div>
                    <p style={{ margin: "0.45rem 0 0", color: "#64748b", fontSize: "0.72rem" }}>
                      {new Date(item.created_at).toLocaleString()}
                    </p>
                  </button>
                );
              })}
              {items.length === 0 && !loading ? (
                <div style={{ padding: "1rem", color: "#64748b", fontSize: "0.9rem" }}>No traces found for current filters.</div>
              ) : null}
            </div>
          </aside>

          <article style={{ border: "1px solid rgba(51, 65, 85, 0.8)", borderRadius: 14, background: "rgba(15, 23, 42, 0.75)", minHeight: 640, padding: "0.95rem" }}>
            {!selected ? (
              <p style={{ margin: 0, color: "#64748b" }}>Select a query to inspect trace details.</p>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: "0.8rem" }}>
                <section style={{ borderRadius: 12, border: "1px solid rgba(51, 65, 85, 0.6)", background: "rgba(2, 6, 23, 0.45)", padding: "0.8rem" }}>
                  <p style={{ margin: 0, color: "#64748b", fontSize: "0.72rem", letterSpacing: "0.08em", textTransform: "uppercase" }}>Query</p>
                  <p style={{ margin: "0.35rem 0 0", fontSize: "1rem", color: "#f8fafc", fontWeight: 600 }}>{selected.query}</p>
                </section>

                <details open style={{ borderRadius: 12, border: "1px solid rgba(51, 65, 85, 0.6)", background: "rgba(2, 6, 23, 0.45)", padding: "0.75rem" }}>
                  <summary style={{ cursor: "pointer", color: "#cbd5e1", fontWeight: 700 }}>Routing</summary>
                  <div style={{ marginTop: "0.65rem", display: "flex", gap: "0.6rem", flexWrap: "wrap" }}>
                    <span style={{ borderRadius: 999, padding: "0.15rem 0.55rem", ...statusBadgeStyle("yellow") }}>
                      Query Type: {selected.trace?.query_type ?? "unknown"}
                    </span>
                    <span style={{ borderRadius: 999, padding: "0.15rem 0.55rem", ...statusBadgeStyle("green") }}>
                      Route: {selected.trace?.route ?? "unknown"}
                    </span>
                  </div>
                </details>

                {showTopKEvolution ? (
                  <details open style={{ borderRadius: 12, border: "1px solid rgba(51, 65, 85, 0.6)", background: "rgba(2, 6, 23, 0.45)", padding: "0.75rem" }}>
                    <summary style={{ cursor: "pointer", color: "#cbd5e1", fontWeight: 700 }}>Retrieval Top-K Evolution</summary>
                    <div style={{ marginTop: "0.75rem", borderRadius: 10, border: "1px solid rgba(51, 65, 85, 0.6)", overflow: "hidden" }}>
                      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.78rem" }}>
                        <thead>
                          <tr style={{ background: "rgba(15, 23, 42, 0.8)", color: "#94a3b8" }}>
                            <th style={{ textAlign: "left", padding: "0.55rem", borderBottom: "1px solid rgba(51, 65, 85, 0.6)" }}>Chunk</th>
                            <th style={{ textAlign: "left", padding: "0.55rem", borderBottom: "1px solid rgba(51, 65, 85, 0.6)" }}>BM25</th>
                            <th style={{ textAlign: "left", padding: "0.55rem", borderBottom: "1px solid rgba(51, 65, 85, 0.6)" }}>Dense</th>
                            <th style={{ textAlign: "left", padding: "0.55rem", borderBottom: "1px solid rgba(51, 65, 85, 0.6)" }}>RRF</th>
                            <th style={{ textAlign: "left", padding: "0.55rem", borderBottom: "1px solid rgba(51, 65, 85, 0.6)" }}>Rerank</th>
                          </tr>
                        </thead>
                        <tbody>
                          {rankMovementRows.map((row) => {
                            const deltaDense = movementDelta(row.bm25Rank, row.denseRank);
                            const deltaRrf = movementDelta(row.denseRank, row.rrfRank);
                            const deltaRerank = movementDelta(row.rrfRank, row.rerankRank);
                            return (
                              <tr key={row.chunkId} style={{ borderBottom: "1px solid rgba(30, 41, 59, 0.75)" }}>
                                <td style={{ padding: "0.55rem", color: "#cbd5e1", maxWidth: 310 }}>
                                  <div style={{ fontWeight: 600, color: "#f8fafc", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{row.source}</div>
                                  <div style={{ marginTop: "0.2rem", color: "#64748b", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{row.snippet || "-"}</div>
                                </td>
                                <td style={{ padding: "0.55rem", color: "#cbd5e1" }}>{row.bm25Rank ?? "-"}</td>
                                <td style={{ padding: "0.55rem", color: "#cbd5e1" }}>
                                  <div>{row.denseRank ?? "-"}</div>
                                  <span style={{ display: "inline-block", marginTop: "0.2rem", borderRadius: 999, padding: "0.05rem 0.42rem", fontSize: "0.68rem", ...movementStyle(deltaDense) }}>
                                    {formatDelta(deltaDense)}
                                  </span>
                                </td>
                                <td style={{ padding: "0.55rem", color: "#cbd5e1" }}>
                                  <div>{row.rrfRank ?? "-"}</div>
                                  <span style={{ display: "inline-block", marginTop: "0.2rem", borderRadius: 999, padding: "0.05rem 0.42rem", fontSize: "0.68rem", ...movementStyle(deltaRrf) }}>
                                    {formatDelta(deltaRrf)}
                                  </span>
                                </td>
                                <td style={{ padding: "0.55rem", color: "#cbd5e1" }}>
                                  <div>{row.rerankRank ?? "-"}</div>
                                  <span style={{ display: "inline-block", marginTop: "0.2rem", borderRadius: 999, padding: "0.05rem 0.42rem", fontSize: "0.68rem", ...movementStyle(deltaRerank) }}>
                                    {formatDelta(deltaRerank)}
                                  </span>
                                </td>
                              </tr>
                            );
                          })}
                          {rankMovementRows.length === 0 ? (
                            <tr>
                              <td colSpan={5} style={{ padding: "0.7rem", color: "#64748b" }}>
                                {onlyInterestingMovements ? "No significant movements for this trace." : "No rank movement data available."}
                              </td>
                            </tr>
                          ) : null}
                        </tbody>
                      </table>
                    </div>
                    <div style={{ marginTop: "0.7rem", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.7rem" }}>
                      <section>
                        <p style={{ margin: "0 0 0.3rem", color: "#93c5fd", fontSize: "0.8rem", fontWeight: 700 }}>BM25</p>
                        <pre style={{ margin: 0, overflowX: "auto", whiteSpace: "pre-wrap", wordBreak: "break-word", borderRadius: 10, padding: "0.6rem", background: "#020617", border: "1px solid #1e293b", color: "#cbd5e1", fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", fontSize: "0.74rem" }}>{asPrettyJson(selected.trace?.retrieval?.bm25_topk ?? [])}</pre>
                      </section>
                      <section>
                        <p style={{ margin: "0 0 0.3rem", color: "#93c5fd", fontSize: "0.8rem", fontWeight: 700 }}>Dense (pgvector)</p>
                        <pre style={{ margin: 0, overflowX: "auto", whiteSpace: "pre-wrap", wordBreak: "break-word", borderRadius: 10, padding: "0.6rem", background: "#020617", border: "1px solid #1e293b", color: "#cbd5e1", fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", fontSize: "0.74rem" }}>{asPrettyJson(selected.trace?.retrieval?.dense_topk ?? [])}</pre>
                      </section>
                      <section>
                        <p style={{ margin: "0 0 0.3rem", color: "#93c5fd", fontSize: "0.8rem", fontWeight: 700 }}>RRF Fused</p>
                        <pre style={{ margin: 0, overflowX: "auto", whiteSpace: "pre-wrap", wordBreak: "break-word", borderRadius: 10, padding: "0.6rem", background: "#020617", border: "1px solid #1e293b", color: "#cbd5e1", fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", fontSize: "0.74rem" }}>{asPrettyJson(selected.trace?.retrieval?.rrf_fused ?? [])}</pre>
                      </section>
                      <section>
                        <p style={{ margin: "0 0 0.3rem", color: "#93c5fd", fontSize: "0.8rem", fontWeight: 700 }}>Reranked</p>
                        <pre style={{ margin: 0, overflowX: "auto", whiteSpace: "pre-wrap", wordBreak: "break-word", borderRadius: 10, padding: "0.6rem", background: "#020617", border: "1px solid #1e293b", color: "#cbd5e1", fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", fontSize: "0.74rem" }}>{asPrettyJson(selected.trace?.retrieval?.reranked ?? [])}</pre>
                      </section>
                    </div>
                  </details>
                ) : null}

                <details open style={{ borderRadius: 12, border: "1px solid rgba(51, 65, 85, 0.6)", background: "rgba(2, 6, 23, 0.45)", padding: "0.75rem" }}>
                  <summary style={{ cursor: "pointer", color: "#cbd5e1", fontWeight: 700 }}>Final Context</summary>
                  <pre style={{ margin: "0.7rem 0 0", overflowX: "auto", whiteSpace: "pre-wrap", wordBreak: "break-word", borderRadius: 10, padding: "0.6rem", background: "#020617", border: "1px solid #1e293b", color: "#cbd5e1", fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", fontSize: "0.74rem" }}>{asPrettyJson(selected.trace?.final_context ?? [])}</pre>
                </details>

                <details open style={{ borderRadius: 12, border: "1px solid rgba(51, 65, 85, 0.6)", background: "rgba(2, 6, 23, 0.45)", padding: "0.75rem" }}>
                  <summary style={{ cursor: "pointer", color: "#cbd5e1", fontWeight: 700 }}>Verification</summary>
                  <pre style={{ margin: "0.7rem 0 0", overflowX: "auto", whiteSpace: "pre-wrap", wordBreak: "break-word", borderRadius: 10, padding: "0.6rem", background: "#020617", border: "1px solid #1e293b", color: "#cbd5e1", fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", fontSize: "0.74rem" }}>{asPrettyJson(selected.trace?.verification ?? {})}</pre>
                </details>

                <details open style={{ borderRadius: 12, border: "1px solid rgba(51, 65, 85, 0.6)", background: "rgba(2, 6, 23, 0.45)", padding: "0.75rem" }}>
                  <summary style={{ cursor: "pointer", color: "#cbd5e1", fontWeight: 700 }}>Abstention and Timing</summary>
                  <div style={{ marginTop: "0.7rem", display: "flex", gap: "0.6rem", flexWrap: "wrap" }}>
                    <span style={{ borderRadius: 999, padding: "0.15rem 0.55rem", ...(selected.trace?.abstained ? statusBadgeStyle("red") : statusBadgeStyle("green")) }}>
                      Abstained: {selected.trace?.abstained ? "Yes" : "No"}
                    </span>
                    <span style={{ borderRadius: 999, padding: "0.15rem 0.55rem", ...statusBadgeStyle("yellow") }}>
                      Reason: {selected.trace?.abstain_reason || "-"}
                    </span>
                    <span style={{ borderRadius: 999, padding: "0.15rem 0.55rem", border: "1px solid rgba(148, 163, 184, 0.3)", background: "rgba(148, 163, 184, 0.15)", color: "#cbd5e1" }}>
                      Latency: {selected.trace?.latency_ms ?? "-"} ms
                    </span>
                  </div>
                </details>
              </div>
            )}
          </article>
        </section>
      </div>

      <style jsx>{`
        @media (max-width: 1100px) {
          section {
            grid-template-columns: 1fr !important;
          }
        }
      `}</style>
    </main>
  );
}
