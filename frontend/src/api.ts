const API_URL = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") || "http://127.0.0.1:8000";

export type Contribution = {
  feature: string;
  value: unknown;
  shap: number;
};

export type PredictResponse = {
  probability: number;
  band: "Low" | "Medium" | "High" | string;
  top_contributions: Contribution[];
};

export type Sample = {
  band: string;
  probability: number;
  actual_fraud: number;
  claim: Record<string, unknown>;
};

export type MetricsPayload = {
  metrics: {
    n_rows: number;
    n_train: number;
    n_test: number;
    positive_rate: number;
    baseline_pr_auc: number;
    cv_pr_auc: number;
    test_pr_auc: number;
    medium_threshold: number;
    high_threshold: number;
    medium_band_metrics: {
      recall: number;
      precision: number;
      confusion_matrix: {
        tn: number;
        fp: number;
        fn: number;
        tp: number;
        matrix: number[][];
      };
    };
    high_band_metrics: {
      recall: number;
      precision: number;
      confusion_matrix: {
        tn: number;
        fp: number;
        fn: number;
        tp: number;
      };
    };
    why_not_accuracy: string;
  };
  pr_curve: {
    precision: number[];
    recall: number[];
    thresholds: Array<number | null>;
    baseline_pr_auc: number;
    test_pr_auc: number;
  };
  thresholds: {
    high: number;
    medium: number;
    rules?: Record<string, string>;
  };
};

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export async function fetchHealth(): Promise<{ status: string }> {
  return handle(await fetch(`${API_URL}/health`));
}

export async function fetchSamples(): Promise<Sample[]> {
  const data = await handle<{ samples: Sample[] }>(await fetch(`${API_URL}/samples`));
  return data.samples;
}

export async function fetchMetrics(): Promise<MetricsPayload> {
  return handle(await fetch(`${API_URL}/metrics`));
}

export async function predictClaim(claim: Record<string, unknown>): Promise<PredictResponse> {
  return handle(
    await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(claim),
    }),
  );
}

export async function explainClaim(body: {
  probability: number;
  band: string;
  top_contributions: Contribution[];
  claim?: Record<string, unknown>;
}): Promise<{ explanation: string; source: string }> {
  return handle(
    await fetch(`${API_URL}/explain`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
}

export { API_URL };
