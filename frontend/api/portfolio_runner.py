import json
import os
import re
import subprocess
import sys
from http.server import BaseHTTPRequestHandler


TICKER_RE = re.compile(r"^[A-Z0-9.-]+$")


def _repo_root() -> tuple[str | None, list[str]]:
    file_dir = os.path.abspath(os.path.dirname(__file__))
    cwd = os.path.abspath(os.getcwd())
    candidates: list[str] = []

    def add(path: str) -> None:
        normalized = os.path.abspath(path)
        if normalized not in candidates:
            candidates.append(normalized)

    for base in (file_dir, cwd, "/var/task"):
        add(base)
        add(os.path.join(base, ".."))
        add(os.path.join(base, "..", ".."))
        add(os.path.join(base, "..", "..", ".."))
        add(os.path.join(base, ".next", "server"))
        add(os.path.join(base, ".next", "server", "app"))

    for candidate in candidates:
        cli_path = os.path.join(candidate, "src", "cli.py")
        if os.path.exists(cli_path):
            return candidate, candidates

    return None, candidates


def _normalize_tickers(raw: str) -> list[str]:
    tickers = [ticker.strip().upper() for ticker in (raw or "").split(",") if ticker.strip()]
    unique: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        if ticker not in seen:
            seen.add(ticker)
            unique.append(ticker)
    if not all(TICKER_RE.fullmatch(ticker) for ticker in unique):
        raise ValueError("Tickers may only contain A-Z, 0-9, '.', and '-'.")
    return unique


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("content-length", "0"))
            raw_body = self.rfile.read(length) if length > 0 else b"{}"
            body = json.loads(raw_body.decode("utf-8"))
        except Exception as exc:
            _json_response(self, 400, {"error": "Invalid request payload.", "details": str(exc)})
            return

        try:
            required = _normalize_tickers(str(body.get("requiredTickers", "")))
            optional = _normalize_tickers(str(body.get("optionalTickers", "")))
            years = int(body.get("years", 3))
            freq = "daily" if body.get("freq") == "daily" else "weekly"
            cash = float(body.get("cash", 0))
            cache = body.get("cache") is True
            log_level = "DEBUG" if body.get("logLevel") == "DEBUG" else "INFO"
        except (TypeError, ValueError) as exc:
            _json_response(self, 400, {"error": "Invalid request payload.", "details": str(exc)})
            return

        if not required:
            _json_response(self, 400, {"error": "requiredTickers must include at least one ticker."})
            return
        if years < 1 or years > 20:
            _json_response(self, 400, {"error": "years must be an integer between 1 and 20."})
            return
        if cash <= 0:
            _json_response(self, 400, {"error": "cash must be a positive number."})
            return

        root, searched = _repo_root()
        cli_path = os.path.join(root, "src", "cli.py") if root else ""
        requirements_path = os.path.join(root, "requirements.txt") if root else ""
        frontend_requirements_path = os.path.join(root, "frontend", "requirements.txt") if root else ""
        if not os.path.exists(cli_path):
            _json_response(
                self,
                500,
                {
                    "error": "Unable to find src/cli.py in deployment bundle. Verify Vercel includeFiles configuration.",
                    "details": (
                        f"cwd={os.getcwd()}; file_dir={os.path.dirname(__file__)}; "
                        f"searched={','.join(searched)}"
                    ),
                },
            )
            return

        if not (os.path.exists(requirements_path) or os.path.exists(frontend_requirements_path)):
            _json_response(
                self,
                500,
                {
                    "error": "Unable to find requirements.txt in deployment bundle.",
                    "details": (
                        f"checked={requirements_path},{frontend_requirements_path}; "
                        f"cwd={os.getcwd()}; file_dir={os.path.dirname(__file__)}"
                    ),
                },
            )
            return

        args = [
            sys.executable,
            "-m",
            "src.cli",
            "--required-tickers",
            ",".join(required),
            "--years",
            str(years),
            "--freq",
            freq,
            "--cash",
            str(cash),
            "--log-level",
            log_level,
            "--no-plot-frontier",
            "--cache" if cache else "--no-cache",
        ]
        if optional:
            args.extend(["--optional-tickers", ",".join(optional)])

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("FAUSTCALC_CACHE_DIR", "/tmp/faustcalc-cache")

        try:
            result = subprocess.run(
                args,
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=55,
                check=False,
            )
        except subprocess.TimeoutExpired:
            _json_response(
                self,
                504,
                {"error": "Portfolio engine timed out.", "details": "The Python function exceeded the execution timeout."},
            )
            return
        except Exception as exc:
            _json_response(self, 500, {"error": "Portfolio engine failed.", "details": str(exc)})
            return

        if result.returncode != 0:
            _json_response(
                self,
                500,
                {
                    "error": "Portfolio engine failed.",
                    "details": (result.stderr or result.stdout or "Python process failed.").strip(),
                    "exitCode": result.returncode,
                },
            )
            return

        _json_response(
            self,
            200,
            {
                "exitCode": result.returncode,
                "report": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            },
        )

    def log_message(self, format: str, *args) -> None:
        return
