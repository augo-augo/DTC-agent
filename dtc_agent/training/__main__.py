from __future__ import annotations

import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["KMP_BLOCKTIME"] = "0"

import argparse
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Dict, List, Tuple

import crafter
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf

from dtc_agent.config import load_training_config
from dtc_agent.training import TrainingLoop
from dtc_agent.training.wandb_logger import WandBLogger

torch.autograd.set_detect_anomaly(True)


def _frame_to_chw(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[..., None], repeats=3, axis=2)
    if array.shape[-1] == 1:
        array = np.repeat(array, repeats=3, axis=2)
    if array.dtype != np.uint8:
        array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    return array.transpose(2, 0, 1)


def _preprocess_frame(
    frame: np.ndarray, target_shape: Tuple[int, int, int], device: torch.device
) -> torch.Tensor:
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.expand_dims(array, -1)
    tensor = torch.from_numpy(array)
    if tensor.ndim != 3:
        raise ValueError(f"Observation must be [H, W, C] or [C, H, W], got {tensor.shape}")
    if tensor.shape[-1] == target_shape[0]:
        tensor = tensor.permute(2, 0, 1)
    elif tensor.shape[0] != target_shape[0]:
        raise ValueError(f"Incompatible observation shape {tensor.shape} for expected {target_shape}")
    tensor = tensor.to(device=device, dtype=torch.float32, non_blocking=True)
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    tensor = tensor.unsqueeze(0)
    spatial_size = (target_shape[1], target_shape[2])
    if tensor.shape[-2:] != spatial_size:
        tensor = F.interpolate(tensor, size=spatial_size, mode="bilinear", align_corners=False)
    if tensor.shape[1] != target_shape[0]:
        if tensor.shape[1] == 1 and target_shape[0] == 3:
            tensor = tensor.repeat(1, target_shape[0], 1, 1)
        else:
            raise ValueError("Unable to match channel count for observation tensor")
    return tensor.clamp(0.0, 1.0)


def _compute_self_state(
    info: dict | None, step_count: int, horizon: int, state_dim: int
) -> torch.Tensor:
    """Derive self-centric signals from Crafter status fields."""
    if state_dim <= 0:
        return torch.empty(0, dtype=torch.float32)

    source = info or {}

    def _lookup(key: str, default: float) -> float:
        if isinstance(source, dict):
            if key in source:
                return float(source[key])
            stats = source.get("stats")
            if isinstance(stats, dict) and key in stats:
                return float(stats[key])
        return float(default)

    health = _lookup("health", 9.0)
    food = _lookup("food", 9.0)
    health_norm = np.clip(health / 9.0, 0.0, 1.0)
    food_norm = np.clip(food / 9.0, 0.0, 1.0)

    denom = max(1, horizon)
    energy = max(0.0, 1.0 - step_count / denom)
    is_sleeping = _lookup("is_sleeping", source.get("sleep", source.get("sleeping", 0.0)))

    health_critical = 1.0 if health < 3.0 else 0.0
    food_critical = 1.0 if food < 2.0 else 0.0

    features: List[float] = [
        health_norm,
        food_norm,
        energy,
        is_sleeping,
        health_critical,
        food_critical,
    ]
    if state_dim <= len(features):
        selected = features[:state_dim]
    else:
        selected = features + [0.0] * (state_dim - len(features))
    return torch.tensor(selected, dtype=torch.float32)


def _select_env_action(action_tensor: torch.Tensor, action_space_n: int) -> int:
    if action_tensor.ndim != 2:
        raise ValueError("Expected batched action tensor from TrainingLoop.step")
    usable = min(action_tensor.shape[-1], action_space_n)
    slice_tensor = action_tensor[0, :usable]
    index = int(torch.argmax(slice_tensor).item())
    return index % action_space_n


def _actor_loop(
    worker_id: int,
    loop: TrainingLoop,
    config,
    runtime_device: torch.device,
    max_steps: int,
    log_interval: int,
    video_step_interval: int,
    steps_lock: threading.Lock,
    shared_state: Dict[str, int],
    stop_event: threading.Event,
    metrics_queue: Queue,
    seed: int,
) -> None:
    env = crafter.Env()
    try:
        if hasattr(env, "seed"):
            env.seed(seed)
        episode_frames: List[np.ndarray] = []
        episode_steps = 0
        observation = env.reset()
        frame = observation
        observation_tensor = _preprocess_frame(frame, config.encoder.observation_shape, runtime_device)
        with steps_lock:
            shared_state["episodes"] += 1
            episode_id = shared_state["episodes"]
        self_state_vec = _compute_self_state(
            info=None,
            step_count=episode_steps,
            horizon=max_steps,
            state_dim=config.self_state_dim,
        ).unsqueeze(0).to(runtime_device)
        episode_frames = [_frame_to_chw(frame)]
        while not stop_event.is_set():
            with torch.no_grad():
                policy_result = loop.agent.act(
                    observation_tensor,
                    self_state=self_state_vec if self_state_vec.numel() > 0 else None,
                    train=False,
                )
            env_action = _select_env_action(policy_result.action, env.action_space.n)
            next_observation, env_reward, terminated, info = env.step(env_action)
            truncated = False
            next_tensor = _preprocess_frame(
                next_observation, config.encoder.observation_shape, runtime_device
            )
            transition_state = policy_result.self_state
            if transition_state is None and self_state_vec.numel() > 0:
                transition_state = self_state_vec
            loop.trainer.store_transition(
                observation_tensor,
                policy_result.action,
                next_tensor,
                transition_state,
            )
            next_episode_steps = episode_steps + 1
            next_self_state_vec = _compute_self_state(
                info,
                next_episode_steps,
                max_steps,
                config.self_state_dim,
            ).unsqueeze(0).to(runtime_device)
            episode_frames.append(_frame_to_chw(next_observation))
            with steps_lock:
                if shared_state["steps"] >= max_steps:
                    stop_event.set()
                    step_index = shared_state["steps"]
                    reached_limit = True
                else:
                    shared_state["steps"] += 1
                    step_index = shared_state["steps"]
                    reached_limit = shared_state["steps"] >= max_steps
                    if reached_limit:
                        stop_event.set()
            info_dict = info if isinstance(info, dict) else {}
            reward_components = {}
            if policy_result.reward_components is not None:
                reward_components = {
                    name: float(value.mean().item())
                    for name, value in policy_result.reward_components.items()
                }
            raw_components = {}
            if policy_result.raw_reward_components is not None:
                raw_components = {
                    name: float(value.mean().item())
                    for name, value in policy_result.raw_reward_components.items()
                }
            self_state_list: List[float] = []
            if next_self_state_vec.numel() > 0:
                self_state_list = [float(x) for x in next_self_state_vec.squeeze(0).tolist()]
            competence_breakdown = policy_result.competence_breakdown or {}
            progress_tensor = competence_breakdown.get("progress")
            penalty_tensor = competence_breakdown.get("penalty")
            ema_prev_tensor = competence_breakdown.get("ema_prev")
            ema_current_tensor = competence_breakdown.get("ema_current")
            competence_progress = (
                float(progress_tensor.mean().item())
                if isinstance(progress_tensor, torch.Tensor)
                else 0.0
            )
            competence_penalty = (
                float(penalty_tensor.mean().item())
                if isinstance(penalty_tensor, torch.Tensor)
                else 0.0
            )
            competence_ema_prev = (
                float(ema_prev_tensor.item())
                if isinstance(ema_prev_tensor, torch.Tensor)
                else 0.0
            )
            competence_ema_current = (
                float(ema_current_tensor.item())
                if isinstance(ema_current_tensor, torch.Tensor)
                else 0.0
            )
            epistemic_value = (
                float(policy_result.epistemic_novelty.mean().item())
                if isinstance(policy_result.epistemic_novelty, torch.Tensor)
                else 0.0
            )
            real_entropy_value = (
                float(policy_result.real_action_entropy)
                if policy_result.real_action_entropy is not None
                else 0.0
            )
            should_log = log_interval > 0 and step_index % log_interval == 0
            achievements = info_dict.get("achievements") if isinstance(info_dict, dict) else None
            achievements_count = len(achievements) if isinstance(achievements, dict) else 0
            metrics_queue.put(
                {
                    "kind": "step",
                    "worker": worker_id,
                    "step": step_index,
                    "episode": episode_id,
                    "episode_steps": next_episode_steps,
                    "intrinsic": float(policy_result.intrinsic_reward.mean().item()),
                    "novelty": float(policy_result.novelty.mean().item()),
                    "entropy": float(policy_result.observation_entropy.mean().item()),
                    "env_reward": float(env_reward),
                    "reward_components": reward_components,
                    "raw_reward_components": raw_components,
                    "self_state": self_state_list,
                    "info": info_dict,
                    "log": should_log,
                    "done": terminated or truncated or reached_limit,
                    "achievements_count": achievements_count,
                    "epistemic_novelty": epistemic_value,
                    "competence_progress": competence_progress,
                    "competence_penalty": competence_penalty,
                    "competence_ema_prev": competence_ema_prev,
                    "competence_ema_current": competence_ema_current,
                    "real_action_entropy": real_entropy_value,
                }
            )
            done = terminated or truncated or reached_limit
            should_upload_video = False
            if done:
                with steps_lock:
                    last_video_step = shared_state.get("last_video_step", 0)
                    current_step = shared_state["steps"]
                    if current_step - last_video_step >= video_step_interval:
                        shared_state["last_video_step"] = current_step
                        should_upload_video = True
            if (
                done
                and should_upload_video
                and episode_frames
                and len(episode_frames) > 1
            ):
                try:
                    video_array = np.stack(episode_frames, axis=0)
                except ValueError:
                    video_array = None
                if video_array is not None:
                    metrics_queue.put(
                        {
                            "kind": "video",
                            "worker": worker_id,
                            "step": step_index,
                            "episode": episode_id,
                            "frames": video_array,
                            "info": info_dict,
                            "truncated": reached_limit and not terminated and not truncated,
                        }
                    )
            if done:
                if shared_state["steps"] >= max_steps:
                    break
                observation = env.reset()
                frame = observation
                observation_tensor = _preprocess_frame(
                    frame, config.encoder.observation_shape, runtime_device
                )
                episode_steps = 0
                with steps_lock:
                    shared_state["episodes"] += 1
                    episode_id = shared_state["episodes"]
                episode_frames = [_frame_to_chw(frame)]
                self_state_vec = _compute_self_state(
                    info=None,
                    step_count=episode_steps,
                    horizon=max_steps,
                    state_dim=config.self_state_dim,
                ).unsqueeze(0).to(runtime_device)
                continue
            observation_tensor = next_tensor
            self_state_vec = next_self_state_vec
            episode_steps = next_episode_steps
    finally:
        if hasattr(env, "close"):
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="SC-GWT training harness (Crafter integration)")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override configuration values (OmegaConf dotlist syntax).",
    )
    parser.add_argument("--device", default=None, help="Runtime device override.")
    parser.add_argument("--seed", type=int, default=0, help="Environment reset seed.")
    parser.add_argument(
        "--max-steps", type=int, default=5000, help="Total environment steps to execute."
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="How frequently to print intrinsic reward diagnostics.",
    )
    parser.add_argument(
        "--actor-workers",
        type=int,
        default=2,
        help="Number of parallel actor threads to use for experience collection.",
    )
    parser.add_argument(
        "--wandb-publish-interval",
        type=int,
        default=1,
        help="Minimum step gap between successive W&B uploads (0 disables throttling).",
    )
    parser.add_argument(
        "--video-step-interval",
        type=int,
        default=250,
        help="Minimum environment step gap between successive W&B video uploads (0 disables throttling).",
    )
    parser.add_argument(
        "--video-frame-stride",
        type=int,
        default=2,
        help="Retain every Nth frame when constructing episode videos (1 keeps all frames).",
    )
    parser.add_argument(
        "--video-max-frames",
        type=int,
        default=240,
        help="Maximum number of frames kept after striding for a single uploaded video.",
    )
    parser.add_argument(
        "--wandb-log-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for an individual W&B logging job before timing out.",
    )
    args = parser.parse_args()

    raw_cfg = OmegaConf.load(args.config)
    if args.override:
        raw_cfg = OmegaConf.merge(raw_cfg, OmegaConf.from_dotlist(list(args.override)))

    config = load_training_config(args.config, overrides=args.override)
    if args.device:
        config.device = args.device
        raw_cfg.device = args.device

    runtime_device = torch.device(config.device)

    import faiss

    faiss.omp_set_num_threads(1)
    dummy_index = faiss.IndexFlatL2(64)
    dummy_vec = np.random.randn(1, 64).astype("float32")
    dummy_index.add(dummy_vec)
    del dummy_index, dummy_vec
    print("[FAISS] Pre-initialized in main thread")

    wandb.init(
        project="dtc-agent-crafter",
        config=OmegaConf.to_container(raw_cfg, resolve=True),
        name=f"crafter_seed{args.seed}",
    )
    wandb.define_metric("step/total_steps", summary="max")
    wandb.define_metric("step/*", step_metric="step/total_steps")
    wandb.define_metric("train/*", step_metric="step/total_steps")
    wandb.define_metric("dream/*", step_metric="step/total_steps")

    loop = TrainingLoop(config)
    num_workers = max(1, args.actor_workers)
    stop_event = threading.Event()
    steps_lock = threading.Lock()
    policy_lock = threading.Lock()
    shared_state: Dict[str, int] = {"steps": 0, "episodes": 0, "last_video_step": 0}
    metrics_queue: Queue = Queue()

    actor_threads = []
    video_step_interval = max(0, args.video_step_interval)
    for worker_id in range(num_workers):
        worker_seed = args.seed + worker_id
        thread = threading.Thread(
            target=_actor_loop,
            args=(
                worker_id,
                loop,
                config,
                runtime_device,
                args.max_steps,
                args.log_interval,
                video_step_interval,
                steps_lock,
                shared_state,
                stop_event,
                metrics_queue,
                worker_seed,
            ),
            daemon=True,
        )
        thread.start()
        actor_threads.append(thread)

    logger = WandBLogger(
        max_pending_steps=100,
        publish_interval=max(0, args.wandb_publish_interval),
        video_frame_stride=max(1, args.video_frame_stride),
        video_max_frames=max(1, args.video_max_frames),
        log_timeout=max(1.0, args.wandb_log_timeout),
    )
    flush_interval = 50
    last_flush_step = -flush_interval
    current_step = 0
    try:
        while True:
            processed = logger.process_queue(metrics_queue)

            with policy_lock:
                optimize_result = loop.train_step()

            with steps_lock:
                current_step = shared_state["steps"]

            if optimize_result:
                step_index, training_metrics = optimize_result
                training_metrics.setdefault(
                    "train/optimization_step", float(step_index)
                )
                logger.add_training_metrics(training_metrics, target_step=current_step)
                processed = True

            flush_due = current_step - last_flush_step >= flush_interval
            should_flush = flush_due or (
                optimize_result and logger.has_pending()
            )
            if should_flush:
                if logger.flush_pending():
                    processed = True
                last_flush_step = current_step

            with policy_lock:
                optimize_result = loop.train_step()
            if optimize_result:
                step_index, training_metrics = optimize_result
                training_metrics.setdefault(
                    "train/optimization_step", float(step_index)
                )
                with steps_lock:
                    current_step = shared_state["steps"]
                logger.add_training_metrics(training_metrics, target_step=current_step)
                if current_step - last_flush_step >= flush_interval:
                    if logger.flush_pending():
                        processed = True
                    last_flush_step = current_step

            if (
                stop_event.is_set()
                and all(not thread.is_alive() for thread in actor_threads)
                and metrics_queue.empty()
            ):
                if logger.flush_pending():
                    processed = True
                    with steps_lock:
                        current_step = shared_state["steps"]
                    last_flush_step = current_step
                if logger.flush_training_only():
                    processed = True
                if not logger.has_pending():
                    break

            if not processed:
                time.sleep(0.001)
    finally:
        stop_event.set()
        for thread in actor_threads:
            thread.join()
        logger.close()
        wandb.finish()


if __name__ == "__main__":
    main()



