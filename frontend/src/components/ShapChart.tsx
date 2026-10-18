import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Contribution } from "../api";

type Props = {
  contributions: Contribution[];
};

export function ShapChart({ contributions }: Props) {
  const data = [...contributions]
    .map((c) => ({
      name: c.feature.length > 28 ? `${c.feature.slice(0, 28)}…` : c.feature,
      full: c.feature,
      value: c.value,
      shap: Number(c.shap),
    }))
    .sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));

  if (!data.length) {
    return <p className="text-sm text-slate-400">No SHAP contributions yet.</p>;
  }

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 16, right: 16, top: 8, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis type="number" stroke="#94a3b8" tick={{ fontSize: 11 }} />
          <YAxis type="category" dataKey="name" width={120} stroke="#94a3b8" tick={{ fontSize: 11 }} />
          <Tooltip
            contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 8 }}
            labelStyle={{ color: "#e2e8f0" }}
            formatter={(val, _name, item) => {
              const row = item?.payload as { value?: unknown; full?: string };
              return [`${Number(val).toFixed(4)} (value=${String(row?.value ?? "")})`, "SHAP"];
            }}
          />
          <ReferenceLine x={0} stroke="#64748b" />
          <Bar dataKey="shap" radius={[0, 4, 4, 0]}>
            {data.map((d) => (
              <Cell key={d.full} fill={d.shap >= 0 ? "#ef4444" : "#3b82f6"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="mt-2 text-xs text-slate-500">
        Red bars push toward fraud; blue bars push away from fraud.
      </p>
    </div>
  );
}
