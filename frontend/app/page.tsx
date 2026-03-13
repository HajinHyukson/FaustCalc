"use client";

import { FormEvent, useState } from "react";

type ApiSuccess = {
  exitCode: number;
  report: string;
  stderr: string;
};

type ApiError = {
  error: string;
  details?: string;
  exitCode?: number;
};

async function readApiPayload(response: Response): Promise<{
  json: Record<string, unknown> | null;
  text: string;
}> {
  const text = await response.text();
  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.toLowerCase().includes("application/json")) {
    return { json: null, text };
  }

  try {
    const parsed = JSON.parse(text) as Record<string, unknown>;
    return { json: parsed, text };
  } catch {
    return { json: null, text };
  }
}

export default function Page() {
  const [requiredTickers, setRequiredTickers] = useState("SPY,QQQ");
  const [optionalTickers, setOptionalTickers] = useState("AAPL,MSFT,NVDA,AMZN");
  const [years, setYears] = useState(3);
  const [freq, setFreq] = useState<"weekly" | "daily">("weekly");
  const [cashInput, setCashInput] = useState("100000.00");
  const [cache, setCache] = useState(true);
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState("");
  const [error, setError] = useState("");

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setReport("");

    try {
      const response = await fetch("/api/portfolio", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          requiredTickers,
          optionalTickers,
          years,
          freq,
          cash: Number(cashInput),
          cache,
          logLevel: "INFO",
        }),
      });

      const payload = await readApiPayload(response);

      if (!response.ok) {
        const apiError = payload.json as ApiError | null;
        const message = [
          apiError?.error ?? `Request failed with HTTP ${response.status}.`,
          apiError?.details ?? (!apiError ? payload.text.slice(0, 800) : ""),
        ]
          .filter(Boolean)
          .join("\n");
        throw new Error(message || "Request failed.");
      }

      if (!payload.json) {
        throw new Error("API returned a non-JSON success response.");
      }

      const apiSuccess = payload.json as ApiSuccess;
      setReport(apiSuccess.report || "(No output)");
      if (apiSuccess.stderr) {
        setReport((prev) => `${prev}\n\n[stderr]\n${apiSuccess.stderr}`);
      }
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "Unknown request failure.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <h1>FaustCalc on Vercel</h1>
      <p className="muted">
        This page calls <code>/api/portfolio</code>, which runs the portfolio engine in Python on the server.
      </p>

      <section className="card" style={{ marginTop: "1rem" }}>
        <form onSubmit={onSubmit} className="grid">
          <div>
            <label htmlFor="required">Required tickers (comma-separated)</label>
            <input
              id="required"
              value={requiredTickers}
              onChange={(e) => setRequiredTickers(e.target.value)}
              required
            />
          </div>

          <div>
            <label htmlFor="optional">Optional tickers (comma-separated)</label>
            <input id="optional" value={optionalTickers} onChange={(e) => setOptionalTickers(e.target.value)} />
          </div>

          <div className="grid two">
            <div>
              <label htmlFor="years">Years</label>
              <input
                id="years"
                type="number"
                min={1}
                max={20}
                value={years}
                onChange={(e) => setYears(Number(e.target.value))}
                required
              />
            </div>
            <div>
              <label htmlFor="freq">Frequency</label>
              <select id="freq" value={freq} onChange={(e) => setFreq(e.target.value as "weekly" | "daily")}>
                <option value="weekly">weekly</option>
                <option value="daily">daily</option>
              </select>
            </div>
          </div>

          <div className="grid two">
            <div>
              <label htmlFor="cash">Cash (USD)</label>
              <input
                id="cash"
                type="number"
                min={1}
                step="0.01"
                value={cashInput}
                onChange={(e) => setCashInput(e.target.value)}
                required
              />
            </div>
            <div>
              <label htmlFor="cache">Cache mode</label>
              <select id="cache" value={String(cache)} onChange={(e) => setCache(e.target.value === "true")}>
                <option value="true">Use cache</option>
                <option value="false">Fresh pull</option>
              </select>
            </div>
          </div>

          <button type="submit" disabled={loading}>
            {loading ? "Running model..." : "Run Portfolio Model"}
          </button>
        </form>
      </section>

      <section className="card" style={{ marginTop: "1rem" }}>
        <h2 style={{ marginTop: 0 }}>Report</h2>
        {error ? <pre style={{ color: "#b91c1c" }}>{error}</pre> : <pre>{report || "No run yet."}</pre>}
      </section>
    </main>
  );
}
