"""``voicedesk-evals``: run the multilingual conversation evals.

voicedesk-evals synth                        # (re)generate audio fixtures (needs internet)
voicedesk-evals run --mode text              # agent only: tool calls, language, task success
voicedesk-evals run --mode audio             # STT on fixtures -> agent (full understanding path)
voicedesk-evals run --mode audio --stage stt # STT + language ID only, no LLM (cheap CI gate)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from voicedesk.config import Settings
from voicedesk.evals.runner import aggregate, result_rows, run_scenario
from voicedesk.evals.scoring import load_suite
from voicedesk.evals.synth import synthesize_fixtures
from voicedesk.providers.registry import build_llm, build_stt

ROOT = Path(__file__).resolve().parents[4]
EVALS = ROOT / "evals"


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _fmt(v: Any) -> str:
    if isinstance(v, float) and v <= 1.0:
        return f"{v:.0%}"
    return "–" if v is None else str(v)


def to_markdown(report: dict[str, Any]) -> str:
    s, cfg = report["summary"], report["config"]
    lines = [
        f"# Eval run {report['run_id']}",
        "",
        f"mode `{cfg['mode']}` · stage `{cfg['stage']}` · stt `{cfg.get('stt')}`"
        f" · llm `{cfg.get('llm')}` · commit `{cfg.get('commit')}`",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for key, value in s.items():
        if key != "by_group":
            lines.append(f"| {key} | {_fmt(value)} |")
    cols = list(next(iter(s["by_group"].values())).keys())
    lines += ["", "| group | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for g, row in s["by_group"].items():
        lines.append(f"| {g} | " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    failed = [r for r in report["scenarios"] if not r["success"]]
    if failed and cfg["stage"] == "all":
        lines += ["", "## Failed scenarios", ""]
        for r in failed:
            for t in r["turns"]:
                calls = (
                    ", ".join(
                        f"{c['name']}({json.dumps(c['arguments'], ensure_ascii=False)})"
                        for c in t["tool_calls"]
                    )
                    or "no tool"
                )
                lines.append(
                    f"- `{r['id']}#{t['index']}` lang {t['detected_lang']}/{t['expected_lang']}"
                    f" tools_ok={t['tools_ok']} answer_ok={t['answer_ok']} · {calls}"
                    f" · reply: {t['reply'][:120]!r}"
                )
    return "\n".join(lines) + "\n"


async def _run(args: argparse.Namespace) -> int:
    suite = load_suite(args.scenarios)
    if args.only:
        suite.scenarios = [s for s in suite.scenarios if s.id in set(args.only.split(","))]
    settings = Settings()
    stt = build_stt(settings) if args.mode == "audio" else None
    llm = build_llm(settings) if args.stage == "all" else None
    if stt:
        await stt.warmup()
    if llm:
        await llm.warmup()
    t0 = time.time()
    results = []
    for sc in suite.scenarios:
        res = await run_scenario(sc, suite, llm, stt, args.audio_dir)
        mark = "ok " if res.success else "FAIL"
        print(f"  [{mark}] {sc.id}", file=sys.stderr, flush=True)
        results.append(res)
    summary = aggregate(results, with_agent=llm is not None)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    report = {
        "run_id": run_id,
        "duration_s": round(time.time() - t0, 1),
        "config": {
            "mode": args.mode,
            "stage": args.stage,
            "stt": stt.name if stt else None,
            "llm": llm.name if llm else None,
            "commit": _git_sha(),
        },
        "summary": summary,
        "scenarios": result_rows(results),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    name = f"{args.mode}-{args.stage}" + (f"-{args.tag}" if args.tag else "")
    (args.out / f"{name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    md = to_markdown(report)
    (args.out / f"{name}.md").write_text(md)
    print(md)
    gate = summary.get("task_success_rate") if llm else summary.get("language_id_accuracy")
    return 1 if args.min_score is not None and (gate or 0) < args.min_score else 0


def main() -> None:
    p = argparse.ArgumentParser(prog="voicedesk-evals")
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--mode", choices=["text", "audio"], default="text")
    run.add_argument("--stage", choices=["all", "stt"], default="all")
    run.add_argument("--only", help="comma-separated scenario ids")
    run.add_argument("--tag", help="suffix for the result file name, e.g. the model")
    run.add_argument("--min-score", type=float, help="exit 1 if the gate metric is below this")
    synth = sub.add_parser("synth")
    synth.add_argument("--force", action="store_true")
    for sp in (run, synth):
        sp.add_argument("--scenarios", type=Path, default=EVALS / "scenarios.yaml")
        sp.add_argument("--audio-dir", type=Path, default=EVALS / "audio")
    run.add_argument("--out", type=Path, default=EVALS / "results")
    args = p.parse_args()
    if args.cmd == "synth":
        written = asyncio.run(
            synthesize_fixtures(load_suite(args.scenarios), args.audio_dir, args.force)
        )
        print(f"wrote {len(written)} fixtures to {args.audio_dir}")
        return
    if args.stage == "stt" and args.mode != "audio":
        p.error("--stage stt requires --mode audio")
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
