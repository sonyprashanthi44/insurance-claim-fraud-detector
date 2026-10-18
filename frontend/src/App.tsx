import { NavLink, Route, Routes } from "react-router-dom";
import { API_URL } from "./api";
import { MetricsPage } from "./pages/MetricsPage";
import { ScorePage } from "./pages/ScorePage";

export default function App() {
  return (
    <div className="mx-auto min-h-screen max-w-6xl px-4 py-6 sm:px-6">
      <header className="mb-8 flex flex-wrap items-end justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-indigo-300">SIU screening aid</p>
          <h1 className="text-2xl font-semibold text-white sm:text-3xl">Insurance Claim Fraud Detector</h1>
          <p className="mt-1 text-sm text-slate-400">LightGBM + SHAP risk bands. Not an automated decision-maker.</p>
        </div>
        <nav className="flex gap-2">
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              `rounded-lg px-3 py-2 text-sm ${isActive ? "bg-indigo-500 text-white" : "bg-slate-900 text-slate-300 hover:bg-slate-800"}`
            }
          >
            Score
          </NavLink>
          <NavLink
            to="/metrics"
            className={({ isActive }) =>
              `rounded-lg px-3 py-2 text-sm ${isActive ? "bg-indigo-500 text-white" : "bg-slate-900 text-slate-300 hover:bg-slate-800"}`
            }
          >
            Metrics
          </NavLink>
        </nav>
      </header>
      <Routes>
        <Route path="/" element={<ScorePage />} />
        <Route path="/metrics" element={<MetricsPage />} />
      </Routes>
      <footer className="mt-10 border-t border-slate-800 pt-4 text-xs text-slate-500">
        API: {API_URL} · Screening aid only · Demographic features require fairness review before production use
      </footer>
    </div>
  );
}
