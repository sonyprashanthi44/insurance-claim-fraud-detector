type Props = {
  probability: number;
  band: string;
};

const bandColor: Record<string, string> = {
  Low: "#22c55e",
  Medium: "#f59e0b",
  High: "#ef4444",
};

export function RiskGauge({ probability, band }: Props) {
  const pct = Math.max(0, Math.min(1, probability)) * 100;
  const color = bandColor[band] ?? "#94a3b8";

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
      <div className="mb-3 flex items-end justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-wide text-slate-400">Fraud risk</p>
          <p className="text-3xl font-semibold" style={{ color }}>
            {band}
          </p>
        </div>
        <p className="text-2xl font-semibold text-slate-100">{pct.toFixed(1)}%</p>
      </div>
      <div className="h-3 w-full overflow-hidden rounded-full bg-slate-800">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <div className="mt-2 flex justify-between text-[11px] text-slate-500">
        <span>Low</span>
        <span>Medium</span>
        <span>High</span>
      </div>
    </div>
  );
}
