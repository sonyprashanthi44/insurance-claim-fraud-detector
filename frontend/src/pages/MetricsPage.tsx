import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { API_URL, fetchMetrics, type MetricsPayload } from "../api";

function pct(n: number) {
  return `${(n * 100).toFixed(1)}%`;
}

export function MetricsPage() {
  const [data, setData] = useState<MetricsPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        setLoading(true);
        const m = await fetchMetrics();
        if (!cancelled) setData(m);
      } catch (e) {
        if (!cancelled) setError(`Could not load metrics from ${API_URL}: ${(e as Error).message}`);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const prPoints = useMemo(() => {
    if (!data) return [];
    const { precision, recall } = data.pr_curve;
    return precision.map((p, i) => ({ precision: p, recall: recall[i] ?? 0 }));
  }, [data]);

  if (loading) {
    return <p className="text-sm text-slate-400">Loading metrics…</p>;
  }
  if (error) {
    return (
      <div className="rounded-xl border border-red-900/60 bg-red-950/40 px-3 py-2 text-sm text-red-200">
        {error}
      </div>
    );
  }
  if (!data) return null;

  const m = data.metrics;
  const cm = m.medium_band_metrics.confusion_matrix;

  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {[
          { label: "Recall (medium band)", value: pct(m.medium_band_metrics.recall) },
          { label: "Precision (medium band)", value: pct(m.medium_band_metrics.precision) },
          { label: "Test PR-AUC", value: m.test_pr_auc.toFixed(3) },
          { label: "Baseline PR-AUC", value: m.baseline_pr_auc.toFixed(3) },
        ].map((card) => (
          <div key={card.label} className="rounded-2xl border border-slate-800 bg-slate-900/50 p-4">
            <p className="text-xs uppercase tracking-wide text-slate-400">{card.label}</p>
            <p className="mt-2 text-2xl font-semibold text-white">{card.value}</p>
          </div>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
          <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-300">
            Confusion matrix (medium threshold)
          </h3>
          <div className="overflow-hidden rounded-xl border border-slate-800">
            <table className="w-full text-sm">
              <thead className="bg-slate-950 text-slate-400">
                <tr>
                  <th className="px-3 py-2 text-left"> </th>
                  <th className="px-3 py-2 text-left">Pred 0</th>
                  <th className="px-3 py-2 text-left">Pred 1</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-t border-slate-800">
                  <td className="px-3 py-2 text-slate-400">Actual 0</td>
                  <td className="px-3 py-2">TN {cm.tn}</td>
                  <td className="px-3 py-2">FP {cm.fp}</td>
                </tr>
                <tr className="border-t border-slate-800">
                  <td className="px-3 py-2 text-slate-400">Actual 1</td>
                  <td className="px-3 py-2">FN {cm.fn}</td>
                  <td className="px-3 py-2">TP {cm.tp}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs text-slate-500">
            Medium thr={m.medium_threshold.toFixed(4)} · High thr={m.high_threshold.toFixed(4)} · CV
            PR-AUC={m.cv_pr_auc.toFixed(3)}
          </p>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
          <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-300">
            Precision–Recall curve
          </h3>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={prPoints}>
                <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
                <XAxis
                  dataKey="recall"
                  type="number"
                  domain={[0, 1]}
                  stroke="#94a3b8"
                  tick={{ fontSize: 11 }}
                  label={{ value: "Recall", position: "insideBottom", offset: -2, fill: "#64748b" }}
                />
                <YAxis
                  dataKey="precision"
                  type="number"
                  domain={[0, 1]}
                  stroke="#94a3b8"
                  tick={{ fontSize: 11 }}
                />
                <Tooltip
                  contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 8 }}
                />
                <Line type="monotone" dataKey="precision" stroke="#818cf8" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-2 text-xs text-slate-500">
            Model PR-AUC {m.test_pr_auc.toFixed(3)} vs baseline {m.baseline_pr_auc.toFixed(3)} (positive
            rate).
          </p>
        </div>
      </div>

      <div className="rounded-2xl border border-amber-900/40 bg-amber-950/20 p-5">
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-amber-200">
          Why not accuracy?
        </h3>
        <p className="text-sm leading-relaxed text-amber-100/90">{m.why_not_accuracy}</p>
      </div>
    </div>
  );
}
