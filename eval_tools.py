"""Measure how often Jarvis picks the right tool.

Sends every request in evals/tool-choice/cases.json to Claude with Jarvis's real
instructions, tools and settings (imported from jarvis11.py), and checks which tool
Claude chose. The tools are never actually run, so nothing on your Mac changes.

    python eval_tools.py            # each case once
    python eval_tools.py --reps 2   # each case twice, for a more reliable score
    python eval_tools.py --limit 3  # just the first 3 cases, to check the cost first
"""
import argparse
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

import jarvis11 as jarvis

FLOW = Path(__file__).parent / "evals" / "tool-choice"
OUT = FLOW / "baseline"
lock = threading.Lock()


def tool_definitions() -> list:
    # Jarvis's @beta_tool functions become plain definitions; the server tools already are
    return [tool if isinstance(tool, dict) else tool.to_dict() for tool in jarvis.TOOLS]


def append(path: Path, row: dict) -> None:
    with lock, path.open("a") as file:  # write each result as soon as it is ready
        file.write(json.dumps(row) + "\n")


def run_case(case: dict, rep: int) -> None:
    started = time.monotonic()
    try:
        response = jarvis.client.with_options(timeout=120).beta.messages.create(
            **jarvis.REQUEST_OPTIONS,
            tools=tool_definitions(),
            messages=[{"role": "user", "content": case["prompt"]}],
        )
    except anthropic.APIError as error:
        append(OUT / "errors.jsonl", {"prompt_id": case["id"], "rep": rep, "failure": "api_error", "error": str(error)})
        return
    latency = round(time.monotonic() - started, 2)
    if not response.model.startswith(jarvis.MODEL):
        append(OUT / "errors.jsonl", {"prompt_id": case["id"], "rep": rep, "failure": "wrong_model", "model": response.model})
        return

    called = [block.name for block in response.content if block.type in ("tool_use", "server_tool_use")]
    expected = case["expected"]
    if expected:
        correct = any(name in expected for name in called)
    else:
        correct = not called  # "no tool" cases pass only if Claude just answers

    usage = response.usage
    trace = [{"role": "system", "content": jarvis.SYSTEM}, {"role": "user", "content": case["prompt"]}]
    for block in response.content:
        if block.type in ("tool_use", "server_tool_use"):
            trace.append({"role": "tool_call", "name": block.name, "content": json.dumps(block.input, indent=2)})
        elif block.type.endswith("_tool_result"):
            trace.append({"role": "tool_result", "content": f"({block.type} from Anthropic's servers)"})
        elif block.type == "text":
            trace.append({"role": "assistant", "content": block.text})
    (OUT / "traces").mkdir(parents=True, exist_ok=True)
    (OUT / "traces" / f"{case['id']}_rep{rep}.json").write_text(json.dumps(trace, indent=2))

    append(OUT / "results.jsonl", {
        "prompt_id": case["id"],
        "prompt": case["prompt"],
        "tags": case["tags"],
        "rep": rep,
        "model": response.model,
        "stop_reason": response.stop_reason,
        "status": "truncated" if response.stop_reason == "max_tokens" else "ok",
        "grade": {"correct_tool": 1 if correct else 0},
        "latency_s": latency,
        "tool_calls": len(called),
        "web_searches": called.count("web_search"),
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
            "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0,
        },
        "meta": {"expected": expected or ["(no tool)"], "called": called or ["(no tool)"]},
    })


def summarize() -> None:
    rows = [json.loads(line) for line in (OUT / "results.jsonl").read_text().splitlines()]
    scored = [row for row in rows if row["status"] == "ok"]
    n = len(scored)
    passed = sum(row["grade"]["correct_tool"] for row in scored)
    accuracy = passed / n
    margin = 1.96 * math.sqrt(accuracy * (1 - accuracy) / n)  # 95% confidence range
    print(f"\nRight tool: {passed}/{n} = {accuracy:.0%} (give or take {margin:.0%})")
    for row in scored:
        if not row["grade"]["correct_tool"]:
            print(f"  MISSED {row['prompt_id']}: expected {row['meta']['expected']}, Claude used {row['meta']['called']}")
    # Cost at Claude Opus 5.5 prices: $4 per million input tokens, $20 per million output,
    # cache writes 1.25x and cache reads 0.05x the input price, plus $0.01 per web search
    cost = sum(
        (row["usage"]["input_tokens"] * 4 + row["usage"]["output_tokens"] * 20
         + row["usage"]["cache_creation_input_tokens"] * 5 + row["usage"]["cache_read_input_tokens"] * 0.2) / 1e6
        + row["web_searches"] * 0.01
        for row in rows
    )
    errors = OUT / "errors.jsonl"
    print(f"Cost of this run: about ${cost:.2f}")
    if errors.exists():
        print(f"{len(errors.read_text().splitlines())} attempts failed and were not scored; see {errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cases = json.loads((FLOW / "cases.json").read_text())[:args.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if (OUT / "results.jsonl").exists():  # skip cases already finished, if a run was interrupted
        done = {(row["prompt_id"], row["rep"]) for row in map(json.loads, (OUT / "results.jsonl").read_text().splitlines())}
    todo = [(case, rep) for rep in range(args.reps) for case in cases if (case["id"], rep) not in done]
    print(f"Running {len(todo)} requests...")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda job: run_case(*job), todo))
    summarize()
