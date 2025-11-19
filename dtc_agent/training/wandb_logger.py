from __future__ import annotations

import threading
import time
import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, MutableMapping

import numpy as np
import wandb


def _make_serializable(obj: Any) -> Any:
    """Convert numpy types and nested containers into JSON-serializable data."""

    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: _make_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    if isinstance(obj, tuple):
        return [_make_serializable(item) for item in obj]
    return obj


class WandBLogger:
    """Thread-safe helper for aggregating and logging metrics to W&B."""

    def __init__(
        self,
        max_pending_steps: int = 100,
        enable_video_logging: bool = True,
        publish_interval: int | None = None,
        video_frame_stride: int = 1,
        video_max_frames: int = 400,
        log_workers: int = 3,
        log_timeout: float = 30.0,
    ) -> None:
        """Create a logger that buffers metrics before publishing to W&B.

        Args:
            max_pending_steps: Maximum number of steps buffered before stalling producers.
            enable_video_logging: Whether to log videos alongside metrics.
            publish_interval: Step interval for pushing metrics to W&B.
            video_frame_stride: Subsampling stride applied to video frames.
            video_max_frames: Maximum frames per logged video clip.
            log_workers: Number of background worker threads used for logging.
            log_timeout: Seconds to wait for an individual logging job.
        """

        self.latest_step_logged = 0
        self.pending_metrics: Dict[int, List[Dict]] = {}
        self.pending_training: Dict[int, Dict[str, float]] = {}
        self.pending_videos: Dict[int, List[Dict]] = {}
        self.latest_training_loss: float | None = None
        self.lock = threading.Lock()
        self._step_condition = threading.Condition(self.lock)
        self.max_pending_steps = max(0, int(max_pending_steps))
        self._enable_video_logging = bool(enable_video_logging)
        self._shutdown = False
        self._log_queue: Queue = Queue()
        self._log_timeout = max(1.0, float(log_timeout))
        worker_count = max(1, int(log_workers))
        self._log_executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="wandb-log")
        now = time.time()
        self._last_worker_heartbeat = now
        self._last_worker_warning = 0.0
        self._queue_warning_interval = 15.0
        self._worker_warning_interval = 60.0
        self._last_queue_size = 0
        self._queue_stall_since: float | None = None
        self._last_queue_warning = 0.0
        self._post_lock_messages: List[str] = []
        self._log_thread = threading.Thread(
            target=self._logging_worker,
            name="wandb-log-consumer",
            daemon=True,
        )
        self._log_thread.start()
        run = wandb.run
        if run is not None:
            try:
                existing_step = int(run.summary.get("step/total_steps", 0) or 0)
            except (TypeError, ValueError):
                existing_step = 0
            self.latest_step_logged = max(0, existing_step)

        if run is not None and getattr(run, "dir", None):
            local_root = Path(run.dir) / "offline_logs"
        else:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            local_root = Path.cwd() / "offline_logs" / timestamp
        try:
            local_root.mkdir(parents=True, exist_ok=True)
        except Exception:
            local_root = None
        self._local_root = local_root
        self._metrics_file_path = None
        self._video_metadata_path = None
        self._videos_dir = None
        self._local_file_lock = threading.Lock()
        if self._local_root is not None:
            self._metrics_file_path = self._local_root / "metrics.jsonl"
            self._video_metadata_path = self._local_root / "videos.jsonl"
            self._videos_dir = self._local_root / "videos"
            try:
                self._videos_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                self._videos_dir = None
        if publish_interval is None:
            publish_interval = 500
        self._wandb_publish_interval = max(0, int(publish_interval))
        self._last_wandb_publish_step = -self._wandb_publish_interval
        self._video_frame_stride = max(1, int(video_frame_stride))
        self._video_max_frames = max(1, int(video_max_frames))

    # ------------------------------------------------------------------
    # Queue processing
    # ------------------------------------------------------------------
    def process_queue(self, metrics_queue: Queue) -> bool:
        """Drain the queue and stage metrics grouped by step.

        Args:
            metrics_queue: Queue populated by producer threads with metric payloads.

        Returns:
            ``True`` if any metrics were staged, otherwise ``False``.
        """

        drained: List[Dict] = []
        while True:
            try:
                drained.append(metrics_queue.get_nowait())
            except Empty:
                break

        if not drained:
            return False

        with self.lock:
            for message in drained:
                kind = message.get("kind")
                if kind == "step":
                    step = int(message["step"])
                    self.pending_metrics.setdefault(step, []).append(message)
                elif kind == "video" and self._enable_video_logging:
                    step = int(message.get("step", 0))
                    self.pending_videos.setdefault(step, []).append(message)
                elif kind == "training":
                    step = int(message["step"])
                    metrics = message.get("metrics", {})
                    if isinstance(metrics, MutableMapping):
                        self.pending_training[step] = dict(metrics)

        return True

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def flush_pending(self) -> bool:
        """Flush staged metrics in step order.

        Returns:
            ``True`` if any logging jobs were scheduled, otherwise ``False``.
        """

        step_batches: List[tuple[Dict, List[Dict]]] = []
        with self.lock:
            if self.pending_metrics:
                ready_steps = sorted(
                    step for step in self.pending_metrics if step > self.latest_step_logged
                )

                for step in ready_steps:
                    payload = self._flush_step_locked(step)
                    if payload is not None:
                        videos = (
                            self.pending_videos.pop(step, [])
                            if self._enable_video_logging
                            else []
                        )
                        step_batches.append((payload, videos))

                if ready_steps:
                    self._cleanup_stale_metrics_locked()
                    self._enforce_pending_limits_locked()
        self._emit_post_lock_messages()

        if not step_batches:
            return False

        scheduled = False
        for payload, videos in step_batches:
            scheduled |= self._enqueue_log_job(self._log_metrics_payload, payload)
            if self._enable_video_logging:
                for video_message in videos:
                    scheduled |= self._enqueue_log_job(self._log_video, video_message)

        return scheduled

    def flush_training_only(self) -> bool:
        """Flush outstanding training metrics without paired step data.

        Returns:
            ``True`` if any standalone training metrics were logged.
        """

        step_batches: List[tuple[int, Dict[str, float], List[Dict]]] = []
        with self.lock:
            if self.pending_training:
                for step in sorted(self.pending_training.keys()):
                    metrics = self.pending_training.pop(step)
                    if step <= self.latest_step_logged:
                        continue
                    self.latest_training_loss = metrics.get("train/total_loss")
                    self.latest_step_logged = step
                    self._step_condition.notify_all()
                    videos = (
                        self.pending_videos.pop(step, [])
                        if self._enable_video_logging
                        else []
                    )
                    step_batches.append((step, metrics, videos))
                self._cleanup_stale_metrics_locked()
                self._enforce_pending_limits_locked()
        self._emit_post_lock_messages()

        if not step_batches:
            return False

        scheduled = False
        for step, metrics, videos in step_batches:
            scheduled |= self._enqueue_log_job(self._log_metrics_only, step, metrics)
            if self._enable_video_logging:
                for video_message in videos:
                    scheduled |= self._enqueue_log_job(self._log_video, video_message)

        return scheduled

    def add_training_metrics(self, metrics: Dict[str, float], target_step: int | None = None) -> None:
        """Stage training metrics to be flushed on a future step.

        Args:
            metrics: Mapping of metric names to values.
            target_step: Optional step index to associate with the metrics.
        """

        with self.lock:
            if target_step is not None:
                target_step = int(target_step)
            if target_step is None or target_step <= self.latest_step_logged:
                target_step = self.latest_step_logged + 1
            self.pending_training[target_step] = dict(metrics)

    def has_pending(self) -> bool:
        """Return whether metrics or videos remain to be logged.

        Returns:
            ``True`` if any metrics, videos, or queued jobs remain.
        """
        queue_size = self._log_queue.qsize()
        with self.lock:
            pending_state = bool(self.pending_metrics or self.pending_training)
            if self._enable_video_logging:
                pending_state = pending_state or bool(self.pending_videos)
            self._monitor_queue_locked(queue_size)
        self._emit_post_lock_messages()
        return pending_state or queue_size > 0

    def _enqueue_log_job(self, func, *args, **kwargs) -> bool:
        with self.lock:
            if self._shutdown:
                return False
        self._log_queue.put((func, args, kwargs))
        return True

    def _logging_worker(self) -> None:
        while True:
            job = self._log_queue.get()
            if job is None:
                self._log_queue.task_done()
                break

            func, args, kwargs = job
            future = None
            try:
                future = self._log_executor.submit(func, *args, **kwargs)
                future.result(timeout=self._log_timeout)
                with self.lock:
                    self._last_worker_heartbeat = time.time()
                    self._last_worker_warning = 0.0
            except TimeoutError:
                description = self._describe_job(func, args, kwargs)
                self._emit_console_warning(
                    f"[W&B] Logging job '{description}' exceeded "
                    f"{self._log_timeout:.1f}s; continuing without waiting."
                )
                if future is not None:
                    future.cancel()
                with self.lock:
                    self._last_worker_warning = time.time()
            except Exception as exc:  # pragma: no cover - defensive logging
                description = self._describe_job(func, args, kwargs)
                self._emit_console_warning(f"[W&B] Logging job '{description}' failed: {exc}")
                with self.lock:
                    self._last_worker_heartbeat = time.time()
                    self._last_worker_warning = 0.0
            finally:
                self._log_queue.task_done()

    def _log_metrics_payload(self, payload: Dict) -> None:
        step = int(payload["step"])
        metrics = payload["metrics"]
        self._write_metrics_local(payload)
        if not self._should_publish_step(step):
            return
        wandb.log(metrics, step=step)
        self._last_wandb_publish_step = step
        achievements = payload.get("achievements_count", 0)
        if achievements:
            wandb.log({"episode/final_achievements": int(achievements)}, step=step)
        if payload.get("log_progress"):
            worker_id = payload.get("worker")
            self._print_progress(step, metrics, worker_id=worker_id)

    def _log_metrics_only(self, step: int, metrics: Dict[str, float]) -> None:
        self._write_metrics_local({"step": step, "metrics": metrics})
        if not self._should_publish_step(step):
            return
        wandb.log(metrics, step=step)
        self._last_wandb_publish_step = step

    def _log_video(self, message: Dict) -> None:
        if not self._enable_video_logging:
            return

        frames = message.get("frames")
        if not isinstance(frames, np.ndarray) or frames.shape[0] <= 1:
            return

        step = int(message.get("step", 0))
        deadline = time.time() + self._log_timeout

        skip_due_to_timeout = False
        with self._step_condition:
            while (
                not self._shutdown
                and step > self.latest_step_logged
            ):
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                self._step_condition.wait(timeout=remaining)

            if self._shutdown:
                return

            if step > self.latest_step_logged:
                self._push_post_lock_message_locked(
                    f"[W&B] Skipping video for step {step}: timed out waiting for metrics flush"
                )
                skip_due_to_timeout = True
        self._emit_post_lock_messages()
        if skip_due_to_timeout:
            return

        label = "episode/video_truncated" if message.get("truncated") else "episode/video"
        caption = (
            f"Worker {message.get('worker')} Episode {message.get('episode')} "
            f"(info: {message.get('info')})"
        )

        stride = self._video_frame_stride
        if stride > 1:
            frames = frames[::stride]
        frame_count = frames.shape[0]
        max_frames = self._video_max_frames
        if frame_count > max_frames:
            sampling = int(np.ceil(frame_count / max_frames))
            frames = frames[::sampling]
        frames = np.ascontiguousarray(frames)
        local_path = self._save_video_locally(message, frames)
        if not self._should_publish_step(step) and step != self._last_wandb_publish_step:
            return
        try:
            if local_path is not None:
                video_payload = wandb.Video(str(local_path), fps=8, format="mp4", caption=caption)
            else:
                video_payload = wandb.Video(frames, fps=8, format="mp4", caption=caption)
            wandb.log({label: video_payload}, step=step)
        except Exception as exc:  # pragma: no cover - defensive logging
            self._emit_console_warning(f"[W&B] Failed to log video at step {step}: {exc}")
        else:
            self._last_wandb_publish_step = step

    # ------------------------------------------------------------------
    # Internal helpers (locked context)
    # ------------------------------------------------------------------
    def _flush_step_locked(self, step: int) -> Dict | None:
        messages = self.pending_metrics.pop(step, [])
        if not messages:
            return None
        if step <= self.latest_step_logged:
            return None

        primary = messages[0]
        metrics = self._aggregate_step_metrics(step, messages)

        training_metrics = self.pending_training.pop(step, None)
        if training_metrics:
            metrics.update(training_metrics)
            self.latest_training_loss = training_metrics.get("train/total_loss")

        self.latest_step_logged = step
        self._step_condition.notify_all()

        log_progress = any(msg.get("log") for msg in messages)
        worker_id = None
        for msg in messages:
            worker_val = msg.get("worker")
            if worker_val is not None:
                worker_id = int(worker_val)
                break
        achievements_count = 0
        achievements_candidates = [
            msg.get("achievements_count")
            for msg in messages
            if isinstance(msg.get("achievements_count"), (int, float))
        ]
        if achievements_candidates:
            achievements_count = int(max(achievements_candidates))
        else:
            unlocked_metric = metrics.get("crafter_stats/achievements_unlocked")
            if unlocked_metric is not None:
                achievements_count = int(unlocked_metric)

        return {
            "step": step,
            "metrics": metrics,
            "log_progress": log_progress,
            "worker": worker_id,
            "achievements_count": achievements_count,
        }

    def _aggregate_step_metrics(self, step: int, messages: List[Dict]) -> Dict[str, float]:
        primary = messages[0]
        metrics: Dict[str, float] = {
            "step/total_steps": float(step),
            "step/episode": float(primary.get("episode", 0)),
            "step/episode_steps": float(primary.get("episode_steps", 0)),
        }

        scalar_map = {
            "intrinsic": "step/intrinsic_reward",
            "novelty": "step/avg_slot_novelty",
            "entropy": "step/observation_entropy",
            "env_reward": "step/env_reward",
            "epistemic_novelty": "step/epistemic_novelty",
            "competence_progress": "debug/competence_progress",
            "competence_penalty": "debug/competence_penalty",
            "competence_ema_prev": "debug/competence_ema_prev",
            "competence_ema_current": "debug/competence_ema_current",
            "real_action_entropy": "debug/actor_real_entropy",
        }

        for source_key, target_key in scalar_map.items():
            values = [msg.get(source_key) for msg in messages if isinstance(msg.get(source_key), (int, float))]
            if values:
                metrics[target_key] = float(sum(values) / len(values))

        reward_components = self._aggregate_components(messages, "reward_components")
        raw_reward_components = self._aggregate_components(messages, "raw_reward_components")

        if reward_components:
            metrics.update(
                {
                    "step/reward_competence": reward_components.get("competence", 0.0),
                    "step/reward_empowerment": reward_components.get("empowerment", 0.0),
                }
            )
            explore_value = reward_components.get("explore", 0.0)
            raw_explore_value = raw_reward_components.get("explore", explore_value)
            metrics["step/reward_explore"] = max(explore_value, 0.0)
            metrics["step/reward_explore_raw"] = raw_explore_value

        self_states = [msg.get("self_state") for msg in messages if isinstance(msg.get("self_state"), list)]
        if self_states:
            averaged_state = self._average_lists(self_states)
            state_names = ["health_norm", "food_norm", "energy_step", "is_sleeping"]
            for index, value in enumerate(averaged_state):
                name = state_names[index] if index < len(state_names) else f"feature_{index}"
                metrics[f"self_state/{name}"] = float(value)

        info_dicts = [
            message.get("info") for message in messages if isinstance(message.get("info"), dict)
        ]

        merged_info: Dict[str, float] = {}
        merged_stats: Dict[str, float] = {}
        achievements_union: Dict[str, float] = {}
        inventory_max: Dict[str, float] = {}

        for info in info_dicts:
            for key, value in info.items():
                if key in {"stats", "achievements", "inventory"}:
                    continue
                if isinstance(value, (int, float, bool)):
                    merged_info[key] = float(value)
            stats = info.get("stats")
            if isinstance(stats, dict):
                for key, value in stats.items():
                    if isinstance(value, (int, float, bool)):
                        merged_stats[key] = float(value)
            achievements = info.get("achievements")
            if isinstance(achievements, dict):
                for key, value in achievements.items():
                    if isinstance(value, (int, float, bool)):
                        achievements_union[key] = max(achievements_union.get(key, 0.0), float(value))
            inventory = info.get("inventory")
            if isinstance(inventory, dict):
                for key, value in inventory.items():
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        continue
                    inventory_max[key] = max(inventory_max.get(key, 0.0), numeric)

        def _info_lookup(key: str) -> float | None:
            if key in merged_info:
                return merged_info[key]
            if key in merged_stats:
                return merged_stats[key]
            return None

        stat_mappings = {
            "health": "health",
            "food": "food",
            "drink": "drink",
            "energy": "energy",
            "sleep": "is_sleeping",
            "sleeping": "is_sleeping",
        }
        for source_key, target_name in stat_mappings.items():
            value = _info_lookup(source_key)
            if value is not None:
                metrics[f"crafter_stats/{target_name}"] = value

        if achievements_union:
            unlocked = sum(1.0 for value in achievements_union.values() if value >= 1.0)
            metrics["crafter_stats/achievements_unlocked"] = unlocked
            for name, value in achievements_union.items():
                metrics[f"crafter_achievements/{name}"] = float(value)

        if inventory_max:
            for name, value in inventory_max.items():
                metrics[f"crafter_inventory/{name}"] = value

        return metrics

    def _aggregate_components(self, messages: List[Dict], field: str) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}

        for message in messages:
            components = message.get(field)
            if not isinstance(components, dict):
                continue
            for name, value in components.items():
                try:
                    value_f = float(value)
                except (TypeError, ValueError):
                    continue
                totals[name] = totals.get(name, 0.0) + value_f
                counts[name] = counts.get(name, 0) + 1

        return {name: totals[name] / counts[name] for name in totals if counts.get(name)}

    def _average_lists(self, values: List[List[float]]) -> List[float]:
        if not values:
            return []

        max_len = max(len(item) for item in values)
        sums = [0.0] * max_len
        counts = [0] * max_len

        for item in values:
            for index, value in enumerate(item):
                try:
                    sums[index] += float(value)
                    counts[index] += 1
                except (TypeError, ValueError):
                    continue

        return [sums[i] / counts[i] if counts[i] else 0.0 for i in range(max_len)]

    def _monitor_queue_locked(self, queue_size: int) -> None:
        now = time.time()
        if queue_size > 0:
            if queue_size == self._last_queue_size:
                if self._queue_stall_since is None:
                    self._queue_stall_since = now
                elif now - self._queue_stall_since >= self._queue_warning_interval:
                    if now - self._last_queue_warning >= self._queue_warning_interval:
                        duration = now - self._queue_stall_since
                        self._push_post_lock_message_locked(
                            f"[W&B] Logging queue stalled with {queue_size} pending jobs "
                            f"for {duration:.1f}s."
                        )
                        self._last_queue_warning = now
            else:
                self._queue_stall_since = None
            self._last_queue_size = queue_size

            if now - self._last_worker_heartbeat >= self._worker_warning_interval:
                if now - self._last_worker_warning >= self._worker_warning_interval:
                    elapsed = now - self._last_worker_heartbeat
                    self._push_post_lock_message_locked(
                        f"[W&B] Logging worker has not completed a job in "
                        f"{elapsed:.1f}s; check W&B connectivity."
                    )
                    self._last_worker_warning = now
        else:
            self._queue_stall_since = None
            self._last_queue_warning = 0.0
            self._last_queue_size = 0

    def _should_publish_step(self, step: int) -> bool:
        interval = self._wandb_publish_interval
        if interval <= 0:
            return True
        last = self._last_wandb_publish_step
        return step - last >= interval

    def _write_metrics_local(self, payload: Dict) -> None:
        if self._metrics_file_path is None:
            return
        record = {
            "timestamp": time.time(),
            "step": int(payload.get("step", 0)),
            "metrics": payload.get("metrics", {}),
        }
        try:
            text = json.dumps(record, ensure_ascii=False)
        except Exception as exc:
            self._emit_console_warning(f"[LocalLog] Failed to serialize metrics: {exc}")
            return
        with self._local_file_lock:
            try:
                with self._metrics_file_path.open("a", encoding="utf-8") as handle:
                    handle.write(text + "\n")
            except Exception as exc:
                self._emit_console_warning(f"[LocalLog] Failed to write metrics: {exc}")

    def _record_video_metadata(self, metadata: Dict) -> None:
        if self._video_metadata_path is None:
            return
        try:
            clean_metadata = _make_serializable(metadata)
            text = json.dumps(clean_metadata, ensure_ascii=False)
        except Exception as exc:
            self._emit_console_warning(f"[LocalLog] Failed to serialize video metadata: {exc}")
            return
        with self._local_file_lock:
            try:
                with self._video_metadata_path.open("a", encoding="utf-8") as handle:
                    handle.write(text + "\n")
            except Exception as exc:
                self._emit_console_warning(f"[LocalLog] Failed to write video metadata: {exc}")

    def _save_video_locally(self, message: Dict, frames: np.ndarray) -> Path | None:
        if self._videos_dir is None:
            return None
        step = int(message.get("step", 0))
        worker = message.get("worker")
        worker_part = f"worker{worker}" if worker is not None else "worker"
        filename = f"step_{step:06d}_{worker_part}.mp4"
        target_path = self._videos_dir / filename
        try:
            import imageio.v2 as imageio  # type: ignore[import]
        except Exception as exc:
            self._emit_console_warning(f"[LocalLog] imageio unavailable, skipping local video save: {exc}")
            return

        data = np.asarray(frames)
        if data.ndim == 4 and data.shape[1] in (1, 3):
            data = np.transpose(data, (0, 2, 3, 1))
        if data.dtype != np.uint8:
            data = np.clip(data, 0, 255).astype(np.uint8)

        try:
            imageio.mimwrite(target_path, data, fps=8, quality=8)
        except Exception as exc:
            self._emit_console_warning(f"[LocalLog] Failed to save video {target_path.name}: {exc}")
            return None

        metadata = dict(message)
        metadata.pop("frames", None)
        metadata["local_path"] = str(target_path)
        metadata["timestamp"] = time.time()
        self._record_video_metadata(metadata)
        return target_path

    def _describe_job(self, func, args, kwargs) -> str:
        name = getattr(func, "__name__", repr(func))
        step = None
        if args:
            first = args[0]
            if isinstance(first, dict):
                step = first.get("step")
            elif isinstance(first, int):
                step = first
        if step is None and isinstance(kwargs, dict):
            if "payload" in kwargs and isinstance(kwargs["payload"], dict):
                step = kwargs["payload"].get("step")
            else:
                step = kwargs.get("step")
        if step is not None:
            return f"{name} (step={step})"
        return name

    def _cleanup_stale_metrics_locked(self) -> None:
        cutoff = self.latest_step_logged - self.max_pending_steps
        if self.max_pending_steps <= 0:
            return

        stale_metric_steps = [step for step in self.pending_metrics if step <= cutoff]
        for step in stale_metric_steps:
            del self.pending_metrics[step]

        stale_training_steps = [step for step in self.pending_training if step <= cutoff]
        for step in stale_training_steps:
            del self.pending_training[step]

        if self._enable_video_logging:
            stale_video_steps = [step for step in self.pending_videos if step <= cutoff]
            for step in stale_video_steps:
                del self.pending_videos[step]

    def _enforce_pending_limits_locked(self) -> None:
        if self.max_pending_steps <= 0:
            return

        removed_steps: set[int] = set()

        metric_steps = sorted(self.pending_metrics)
        excess_metrics = len(metric_steps) - self.max_pending_steps
        if excess_metrics > 0:
            drop_steps = metric_steps[:excess_metrics]
            for step in drop_steps:
                self.pending_metrics.pop(step, None)
                self.pending_training.pop(step, None)
                if self._enable_video_logging:
                    self.pending_videos.pop(step, None)
            removed_steps.update(drop_steps)

        training_steps = sorted(self.pending_training)
        excess_training = len(training_steps) - self.max_pending_steps
        if excess_training > 0:
            drop_steps = training_steps[:excess_training]
            for step in drop_steps:
                self.pending_training.pop(step, None)
                if self._enable_video_logging:
                    self.pending_videos.pop(step, None)
            removed_steps.update(drop_steps)

        if self._enable_video_logging:
            video_steps = sorted(self.pending_videos)
            excess_videos = len(video_steps) - self.max_pending_steps
            if excess_videos > 0:
                drop_steps = video_steps[:excess_videos]
                for step in drop_steps:
                    self.pending_videos.pop(step, None)
                removed_steps.update(drop_steps)

        if removed_steps:
            highest = max(removed_steps)
            self._push_post_lock_message_locked(
                f"[W&B] Dropped {len(removed_steps)} pending batches up to step {highest} "
                f"to honor max_pending_steps={self.max_pending_steps}. "
                "Consider increasing the limit or investigating slow logging."
            )

    def _push_post_lock_message_locked(self, message: str) -> None:
        self._post_lock_messages.append(message)

    def _emit_post_lock_messages(self) -> None:
        with self.lock:
            if not self._post_lock_messages:
                return
            messages = self._post_lock_messages
            self._post_lock_messages = []
        for message in messages:
            self._emit_console_warning(message)

    def _print_progress(self, step: int, metrics: Dict[str, float], worker_id: int | None) -> None:
        intrinsic = metrics.get("step/intrinsic_reward", 0.0)
        novelty = metrics.get("step/avg_slot_novelty", 0.0)
        entropy = metrics.get("step/observation_entropy", 0.0)
        loss_str = f"{self.latest_training_loss:.4f}" if self.latest_training_loss is not None else "n/a"
        worker_prefix = f"worker {worker_id} " if worker_id is not None else ""
        self._emit_console_info(
            f"[{worker_prefix}step {step:05d}] "
            f"intrinsic={intrinsic:.4f} "
            f"novelty={novelty:.4f} "
            f"entropy={entropy:.4f} "
            f"loss={loss_str}"
        )

    def close(self) -> None:
        with self.lock:
            self._shutdown = True
            self._step_condition.notify_all()
        self._emit_post_lock_messages()
        pending_jobs = self._log_queue.qsize()
        if pending_jobs:
            self._emit_console_warning(f"[W&B] Waiting for {pending_jobs} logging jobs during shutdown.")
        self._log_queue.put(None)
        self._log_queue.join()
        self._log_thread.join(timeout=5.0)
        if self._log_thread.is_alive():
            self._emit_console_warning("[W&B] Logging worker did not shut down within 5s; forcing executor shutdown.")
        try:
            self._log_executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self._log_executor.shutdown(wait=False)

    def _emit_console_warning(self, message: str) -> None:
        self._safe_console_write(message, stream=sys.stderr)

    def _emit_console_info(self, message: str) -> None:
        self._safe_console_write(message, stream=sys.stdout)

    def _safe_console_write(self, message: str, stream) -> None:
        data = (message + "\n").encode("utf-8", "replace")
        fileno = getattr(stream, "fileno", None)
        if callable(fileno):
            try:
                os.write(fileno(), data)
                return
            except OSError:
                pass
        try:
            stream.write(message + "\n")
        except Exception:
            pass
