import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

type RequestPayload = {
  requiredTickers: string;
  optionalTickers?: string;
  years?: number;
  freq?: "weekly" | "daily";
  cash: number;
  cache?: boolean;
  logLevel?: "INFO" | "DEBUG";
};

const TICKER_RE = /^[A-Z0-9.-]+$/;

function normalizeTickers(raw: string): string[] {
  const tickers = raw
    .split(",")
    .map((t) => t.trim().toUpperCase())
    .filter(Boolean);
  const unique = [...new Set(tickers)];
  if (!unique.every((ticker) => TICKER_RE.test(ticker))) {
    throw new Error("Tickers may only contain A-Z, 0-9, '.', and '-'.");
  }
  return unique;
}

function findRepoRoot(): string | null {
  const candidates = [
    process.cwd(),
    path.resolve(process.cwd(), ".."),
    path.resolve(process.cwd(), "../.."),
  ];
  for (const candidate of candidates) {
    const cliPath = path.join(candidate, "src", "cli.py");
    if (fs.existsSync(cliPath)) {
      return candidate;
    }
  }
  return null;
}

async function runCli(command: string, args: string[], cwd: string): Promise<{ code: number; stdout: string; stderr: string }> {
  return await new Promise((resolve) => {
    const child = spawn(command, args, {
      cwd,
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("close", (code) => {
      resolve({ code: code ?? 1, stdout, stderr });
    });
    child.on("error", (error) => {
      resolve({ code: 1, stdout, stderr: `${stderr}\n${error.message}`.trim() });
    });
  });
}

export async function POST(request: NextRequest) {
  try {
    const body = (await request.json()) as RequestPayload;

    const required = normalizeTickers(body.requiredTickers ?? "");
    const optional = normalizeTickers(body.optionalTickers ?? "");
    const years = Number(body.years ?? 3);
    const freq = body.freq === "daily" ? "daily" : "weekly";
    const cash = Number(body.cash);
    const cache = body.cache === true;
    const logLevel = body.logLevel === "DEBUG" ? "DEBUG" : "INFO";

    if (required.length === 0) {
      return NextResponse.json({ error: "requiredTickers must include at least one ticker." }, { status: 400 });
    }
    if (!Number.isFinite(cash) || cash <= 0) {
      return NextResponse.json({ error: "cash must be a positive number." }, { status: 400 });
    }
    if (!Number.isInteger(years) || years < 1 || years > 20) {
      return NextResponse.json({ error: "years must be an integer between 1 and 20." }, { status: 400 });
    }

    const root = findRepoRoot();
    if (!root) {
      return NextResponse.json(
        { error: "Unable to find src/cli.py in deployment bundle. Verify Vercel Root Directory and included files." },
        { status: 500 }
      );
    }

    const args = [
      "-m",
      "src.cli",
      "--required-tickers",
      required.join(","),
      "--years",
      String(years),
      "--freq",
      freq,
      "--cash",
      String(cash),
      "--log-level",
      logLevel,
      "--no-plot-frontier",
      cache ? "--cache" : "--no-cache",
    ];

    if (optional.length > 0) {
      args.push("--optional-tickers", optional.join(","));
    }

    const pythonCommand = process.env.PYTHON_PATH || "python";
    const result = await runCli(pythonCommand, args, root);

    if (result.code !== 0) {
      return NextResponse.json(
        {
          error: "Portfolio engine failed.",
          details: result.stderr || result.stdout || "Unknown Python execution error.",
          exitCode: result.code,
        },
        { status: 500 }
      );
    }

    return NextResponse.json({
      exitCode: result.code,
      report: result.stdout.trim(),
      stderr: result.stderr.trim(),
    });
  } catch (error) {
    return NextResponse.json(
      { error: "Invalid request payload.", details: error instanceof Error ? error.message : "Unknown error." },
      { status: 400 }
    );
  }
}
