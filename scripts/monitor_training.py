#!/usr/bin/env python3
"""
Live training monitor for SC-GWT runs.
Alerts when critical metrics deviate from expected ranges.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Mapping
from typing import Any

import wandb
from wandb.apis.public import Run


MetricRuleValue = float | int | tuple[float, float]
MetricRules = Mapping[str, MetricRuleValue]
ThresholdMap = Mapping[str, MetricRules]


THRESHOLDS: dict[str, dict[str, MetricRuleValue]] = {
    "step/reward_competence": {"min": -0.05, "warn_below": 0.0, "step_threshold": 15000},
    "step/reward_empowerment": {"min": -0.1, "warn_below": 0.05, "step_threshold": 10000},
    "dream/explore_raw": {"min": 0.0, "warn_below": 0.1, "step_threshold": 30000},
    "debug/empowerment_accuracy": {"min": 0.3, "max": 0.95, "optimal": (0.6, 0.8)},
    "step/episode_steps": {"min": 50, "warn_below": 100, "step_threshold": 20000},
}


def check_metrics(run: Run, thresholds: ThresholdMap) -> list[str]:
    """Poll wandb run and check against thresholds."""
    history = run.scan_history()
    alerts: list[str] = []

    for row in history:
        if not isinstance(row, Mapping):
            continue
        step_raw = row.get("step/total_steps", 0)
        step = int(step_raw) if isinstance(step_raw, (int, float)) else 0

        for metric_name, rules in thresholds.items():
            value_raw = row.get(metric_name)
            if not isinstance(value_raw, (int, float)):
                continue
            value = float(value_raw)

            step_threshold_raw = rules.get("step_threshold")
            if (
                isinstance(step_threshold_raw, (int, float))
                and step > int(step_threshold_raw)
            ):
                warn_below_raw = rules.get("warn_below")
                if isinstance(warn_below_raw, (int, float)) and value < float(
                    warn_below_raw
                ):
                    alerts.append(
                        f"⚠️  Step {step}: {metric_name}={value:.3f} below warning threshold "
                        f"{float(warn_below_raw):.3f}"
                    )

            min_raw = rules.get("min")
            if isinstance(min_raw, (int, float)) and value < float(min_raw):
                alerts.append(
                    f"🚨 Step {step}: {metric_name}={value:.3f} below minimum {float(min_raw):.3f}"
                )

            max_raw = rules.get("max")
            if isinstance(max_raw, (int, float)) and value > float(max_raw):
                alerts.append(
                    f"🚨 Step {step}: {metric_name}={value:.3f} above maximum {float(max_raw):.3f}"
                )

    return alerts


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor DTC-Agent training run")
    parser.add_argument("--run-id", required=True, help="W&B run ID")
    parser.add_argument("--project", default="dtc-agent-crafter", help="W&B project name")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    args = parser.parse_args()

    api = wandb.Api()
    run = api.run(f"{args.project}/{args.run_id}")

    print(f"Monitoring run: {run.name} ({run.id})")
    print(f"Checking every {args.interval} seconds...")
    print("=" * 80)

    last_step = 0
    while True:
        try:
            run.update()
            summary: Mapping[str, Any] = run.summary
            current_step_raw = summary.get("step/total_steps", 0)
            current_step = (
                int(current_step_raw)
                if isinstance(current_step_raw, (int, float))
                else 0
            )

            if current_step > last_step:
                print(f"\n[Step {current_step}] Checking metrics...")
                alerts = check_metrics(run, THRESHOLDS)

                if alerts:
                    print("\n".join(alerts))
                else:
                    print("✓ All metrics within expected ranges")

                last_step = current_step

            time.sleep(args.interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
            break
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Error: {exc}")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
