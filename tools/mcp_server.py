#!/usr/bin/env python3
"""Tiny stdio MCP server with Cortex-specific helper tools.

No dependencies: only the Python standard library. Register in opencode.json:

    "mcp": { "servers": { "cortex": {
        "type": "local", "command": ["python3", "tools/mcp_server.py"], "cwd": "."
    } } }

Tools are intentionally read-only except `build`/`test`, which only write
Rust build artifacts. Never force-push or rewrite history from here.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "/tmp/qwen2-0.5b-q4km.gguf"
BENCH_RE = re.compile(r"throughput:\s+([\d.]+)\s+tok/s")


def run(cmd, timeout=900, env=None):
    try:
        p = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT after %ss: %s" % (timeout, " ".join(cmd))
    tail = p.stdout[-6000:] + (("\n[stderr]\n" + p.stderr[-2000:]) if p.stderr else "")
    return "[exit %d]\n%s" % (p.returncode, tail)


def tool_status(_args):
    return run(["git", "status", "--short"]) + "\n" + run(["git", "log", "-1", "--oneline"])


def tool_test(_args):
    return run(["cargo", "test"], timeout=900)


def tool_build(args):
    cmd = ["cargo", "build", "--release"]
    if args.get("examples", True):
        cmd += ["--examples"]
    return run(cmd, timeout=1200)


def tool_bench(args):
    model = args.get("model", DEFAULT_MODEL)
    prompt = args.get("prompt", "Once upon a time")
    n = int(args.get("n", 40))
    device = args.get("device", "cuda")
    threads = int(args.get("threads", 0))
    graph = bool(args.get("graph", False))
    cmd = [
        str(ROOT / "target/release/examples/bench"),
        model, prompt, str(n),
        "--device", device,
        "--threads", str(threads),
    ]
    env = os.environ.copy()
    if graph:
        env["CORTEX_CUDA_GRAPH"] = "1"
    out = run(cmd, timeout=1200, env=env)
    m = BENCH_RE.search(out)
    speed = ("%.2f tok/s" % float(m.group(1))) if m else "throughput not found"
    return "graph=%s %s\n%s" % (graph, speed, out)


def tool_parity(args):
    """Compare greedy token sequences from CPU and CUDA for the same prompt."""
    model = args.get("model", DEFAULT_MODEL)
    prompt = args.get("prompt", "Once upon a time")
    n = int(args.get("n", 8))
    results = {}
    for dev in ("cpu", "cuda"):
        cmd = [
            str(ROOT / "target/release/examples/generate"),
            model, prompt, "--temp", "0", "--max", str(n), "--device", dev,
            "--threads", str(int(args.get("threads", 4))),
        ]
        results[dev] = run(cmd, timeout=1200)
    same = results["cpu"].strip() == results["cuda"].strip()
    return "identical=%s\n\n[CPU]\n%s\n[CUDA]\n%s" % (
        same, results["cpu"][-2000:], results["cuda"][-2000:]
    )


def tool_ggml_bench(args):
    """Benchmark the optional llama.cpp CUDA FFI backend."""
    model = args.get("model", DEFAULT_MODEL)
    n = int(args.get("n", 128))
    context = int(args.get("context", 1024))
    cmd = [str(ROOT / "target/release/examples/ggml_bench"), model,
           args.get("prompt", "Once upon a time"), str(n), str(context)]
    env = os.environ.copy()
    if args.get("dir"):
        env["CORTEX_GGML_DIR"] = str(args["dir"])
    if args.get("ngl") is not None:
        env["CORTEX_GGML_NGL"] = str(int(args["ngl"]))
    if args.get("chat"):
        env["CORTEX_GGML_CHAT"] = "1"
    return run(cmd, timeout=1200, env=env)


TOOLS = {
    "cortex_status": ("Repository status and HEAD", {}, tool_status),
    "cortex_test": ("Run cargo test", {}, tool_test),
    "cortex_build": ("Build release binaries", {"examples": "boolean"}, tool_build),
    "cortex_bench": (
        "Benchmark decode throughput (tok/s). Set graph=true to test CUDA Graph.",
        {
            "model": "string", "prompt": "string", "n": "integer",
            "device": "string", "threads": "integer", "graph": "boolean",
        },
        tool_bench,
    ),
    "cortex_ggml_bench": (
        "Benchmark decode throughput via llama.cpp CUDA FFI (opt-in backend).",
        {"model": "string", "prompt": "string", "n": "integer", "context": "integer",
         "dir": "string", "ngl": "integer", "chat": "boolean"},
        tool_ggml_bench,
    ),
    "cortex_parity": (
        "Compare greedy CPU vs CUDA output for identical prompt/tokens.",
        {"model": "string", "prompt": "string", "n": "integer", "threads": "integer"},
        tool_parity,
    ),
}


def respond(req_id, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": req_id}
    if error is not None:
        msg["error"] = {"code": -32000, "message": str(error)}
    else:
        msg["result"] = result
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:
            respond(None, error="invalid JSON: %s" % exc)
            continue
        method = req.get("method")
        rid = req.get("id")
        params = req.get("params") or {}
        if method == "initialize":
            respond(rid, {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "cortex-helper", "version": "0.1.0"},
                "instructions": (
                    "Cortex helpers: benchmark decode speed, check CPU/CUDA parity, "
                    "and run tests. Never claim 300+ tok/s without a measured, "
                    "correct-logits result. Keep model name 'ornith 1.5 9b'."
                ),
            })
        elif method == "notifications/initialized" or method.startswith("notifications/"):
            continue
        elif method == "ping":
            respond(rid, {})
        elif method == "tools/list":
            respond(rid, {
                "tools": [
                    {
                        "name": name,
                        "description": desc,
                        "inputSchema": {
                            "type": "object",
                            "properties": {k: {"type": t} for k, t in fields.items()},
                            "additionalProperties": False,
                        },
                    }
                    for name, (desc, fields, _fn) in TOOLS.items()
                ]
            })
        elif method == "tools/call":
            name = params.get("name")
            entry = TOOLS.get(name)
            if not entry:
                respond(rid, error="unknown tool: %s" % name)
                continue
            try:
                text = entry[2](params.get("arguments") or {})
                respond(rid, {"content": [{"type": "text", "text": str(text)}]})
            except Exception as exc:
                respond(rid, {"content": [{"type": "text", "text": "ERROR: %s" % exc}], "isError": True})
        else:
            respond(rid, error="method not found: %s" % method)


if __name__ == "__main__":
    main()
