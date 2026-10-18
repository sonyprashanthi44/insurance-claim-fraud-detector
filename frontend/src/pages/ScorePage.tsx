import { useEffect, useMemo, useState, type FormEvent } from "react";
import {
  API_URL,
  explainClaim,
  fetchSamples,
  predictClaim,
  type PredictResponse,
  type Sample,
} from "../api";
import { RiskGauge } from "../components/RiskGauge";
import { ShapChart } from "../components/ShapChart";

const FIELD_ORDER = [
  "Month","WeekOfMonth","DayOfWeek","Make","AccidentArea","DayOfWeekClaimed","MonthClaimed",
  "WeekOfMonthClaimed","Sex","MaritalStatus","Age","Fault","PolicyType","VehicleCategory",
  "VehiclePrice","Deductible","DriverRating","Days_Policy_Accident","Days_Policy_Claim",
  "PastNumberOfClaims","AgeOfVehicle","PoliceReportFiled","WitnessPresent","AgentType",
  "NumberOfSuppliments","AddressChange_Claim","NumberOfCars","Year","BasePolicy",
];

const emptyClaim = (): Record<string, string> =>
  Object.fromEntries(FIELD_ORDER.map((k) => [k, ""]));

export function ScorePage() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [claim, setClaim] = useState<Record<string, string>>(emptyClaim);
  const [sampleIdx, setSampleIdx] = useState("");
  const [loadingSamples, setLoadingSamples] = useState(true);
  const [scoring, setScoring] = useState(false);
  const [explaining, setExplaining] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [explanation, setExplanation] = useState<string | null>(null);
  const [explainSource, setExplainSource] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        setLoadingSamples(true);
        const s = await fetchSamples();
        if (!cancelled) setSamples(s);
      } catch (e) {
        if (!cancelled) setError(`Could not load samples from ${API_URL}: ${(e as Error).message}`);
      } finally {
        if (!cancelled) setLoadingSamples(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const sampleOptions = useMemo(
    () =>
      samples.map((s, i) => ({
        value: String(i),
        label: `${s.band} · p=${(s.probability * 100).toFixed(1)}% · actual=${s.actual_fraud ? "fraud" : "legit"}`,
      })),
    [samples],
  );

  function loadSample(idx: string) {
    setSampleIdx(idx);
    const s = samples[Number(idx)];
    if (!s) return;
    const next = emptyClaim();
    for (const [k, v] of Object.entries(s.claim)) {
      if (k in next) next[k] = v == null ? "" : String(v);
    }
    setClaim(next);
    setResult(null);
    setExplanation(null);
    setExplainSource(null);
    setError(null);
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setExplanation(null);
    setExplainSource(null);
    setScoring(true);
    try {
      const payload: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(claim)) {
        if (v === "") continue;
        if (["Age","Year","WeekOfMonth","WeekOfMonthClaimed","Deductible","DriverRating"].includes(k)) {
          const n = Number(v);
          payload[k] = Number.isFinite(n) ? n : v;
        } else {
          payload[k] = v;
        }
      }
      const pred = await predictClaim(payload);
      setResult(pred);
      setScoring(false);
      setExplaining(true);
      try {
        const exp = await explainClaim({
          probability: pred.probability,
          band: pred.band,
          top_contributions: pred.top_contributions,
          claim: payload,
        });
        setExplanation(exp.explanation);
        setExplainSource(exp.source);
      } catch (expErr) {
        setExplanation(`Explanation unavailable: ${(expErr as Error).message}`);
        setExplainSource("error");
      } finally {
        setExplaining(false);
      }
    } catch (err) {
      setResult(null);
      setError((err as Error).message);
      setScoring(false);
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <section className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold text-white">Claim form</h2>
            <p className="text-sm text-slate-400">Score a claim or load a held-out sample.</p>
          </div>
          <select
            className="min-w-[16rem] rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm"
            value={sampleIdx}
            disabled={loadingSamples || !samples.length}
            onChange={(e) => loadSample(e.target.value)}
          >
            <option value="">{loadingSamples ? "Loading samples…" : "Load sample claim"}</option>
            {sampleOptions.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </div>
        <form onSubmit={onSubmit} className="space-y-4">
          <div className="grid max-h-[28rem] grid-cols-1 gap-3 overflow-y-auto pr-1 sm:grid-cols-2">
            {FIELD_ORDER.map((field) => (
              <label key={field} className="block text-xs text-slate-400">
                {field}
                <input
                  className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100"
                  value={claim[field] ?? ""}
                  onChange={(e) => setClaim((c) => ({ ...c, [field]: e.target.value }))}
                />
              </label>
            ))}
          </div>
          <button
            type="submit"
            disabled={scoring}
            className="rounded-xl bg-indigo-500 px-4 py-2.5 text-sm font-semibold text-white hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {scoring ? "Scoring…" : "Score claim"}
          </button>
        </form>
        {error && (
          <div className="mt-4 rounded-xl border border-red-900/60 bg-red-950/40 px-3 py-2 text-sm text-red-200">{error}</div>
        )}
      </section>
      <section className="space-y-4">
        {result ? (
          <>
            <RiskGauge probability={result.probability} band={result.band} />
            <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
              <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-300">Top SHAP contributions</h3>
              <ShapChart contributions={result.top_contributions} />
            </div>
            <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
              <div className="mb-2 flex items-center justify-between gap-2">
                <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-300">AI explanation</h3>
                {explainSource && (
                  <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[11px] text-slate-400">source: {explainSource}</span>
                )}
              </div>
              {explaining ? (
                <p className="text-sm text-slate-400">Generating explanation…</p>
              ) : (
                <p className="text-sm leading-relaxed text-slate-200">{explanation ?? "No explanation yet."}</p>
              )}
            </div>
          </>
        ) : (
          <div className="rounded-2xl border border-dashed border-slate-700 bg-slate-900/30 p-8 text-center text-sm text-slate-400">
            Submit a claim to see risk band, SHAP drivers, and an explanation.
          </div>
        )}
      </section>
    </div>
  );
}
