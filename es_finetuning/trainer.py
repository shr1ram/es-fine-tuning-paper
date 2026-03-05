"""Core Evolution Strategies training loop.

The ESTrainer does NOT perform inference or scoring — it accepts an
evaluate_fn callable that returns a scalar reward for the current model
state. This keeps all model/task-specific logic in the caller.
"""

import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from .config import ESConfig
from .noise import apply_es_update, perturb_weights, restore_weights
from .utils import force_memory_cleanup, setup_tensorboard


def _save_results_json(path, config_dict, history, baseline_eval,
                       full_eval_result, total_time):
    """Write results.json summarizing the experiment."""
    results = {
        "config": config_dict,
        "training": {
            "total_time_s": round(total_time, 1),
            "total_time_h": round(total_time / 3600, 2),
            "iterations": len(history["mean"]),
            "reward_per_iteration": {
                "mean": history["mean"],
                "min": history["min"],
                "max": history["max"],
                "time_s": history["time"],
            },
        },
    }
    if baseline_eval is not None:
        results["baseline_eval"] = baseline_eval
    if full_eval_result is not None:
        results["full_eval"] = full_eval_result
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {path}")


def _clean_task_name(name):
    """Strip benchmark prefixes like 'lb/' for display."""
    for prefix in ("lb/", "ib/", "cb/"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _save_results_plot(path, history, baseline_eval, full_eval_result):
    """Generate results.png with training curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping results.png")
        return

    iters = list(range(1, len(history["mean"]) + 1))
    means = history["mean"]
    mins = history["min"]
    maxs = history["max"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Per-iteration reward (mean/min/max)
    ax = axes[0]
    ax.fill_between(iters, mins, maxs, alpha=0.2, color="steelblue",
                     label="min–max")
    ax.plot(iters, means, color="steelblue", linewidth=1, alpha=0.4)
    # Rolling average
    window = max(1, len(means) // 10)
    if len(means) >= window:
        rolling = np.convolve(means, np.ones(window) / window, mode="valid")
        ax.plot(range(window, len(means) + 1), rolling, color="steelblue",
                linewidth=2, label=f"mean (rolling {window})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reward (Qasper F1, 0–1)")
    ax.set_title("Training Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Baseline vs fine-tuned (full eval)
    ax = axes[1]
    has_baseline = baseline_eval and baseline_eval.get("scores")
    has_finetuned = full_eval_result and full_eval_result.get("scores")

    if has_baseline or has_finetuned:
        # Collect all task names from both
        all_tasks = []
        if has_finetuned:
            all_tasks = list(full_eval_result["scores"].keys())
        elif has_baseline:
            all_tasks = list(baseline_eval["scores"].keys())

        clean_names = [_clean_task_name(t) for t in all_tasks]
        base_vals = []
        ft_vals = []
        for t in all_tasks:
            base_vals.append(
                baseline_eval["scores"].get(t, 0) if has_baseline else 0)
            ft_vals.append(
                full_eval_result["scores"].get(t, 0) if has_finetuned else 0)

        y = np.arange(len(clean_names))
        bar_h = 0.35

        ax.barh(y + bar_h / 2, base_vals, bar_h, color="lightcoral",
                label="Baseline")
        ax.barh(y - bar_h / 2, ft_vals, bar_h, color="seagreen",
                label="ES fine-tuned")

        # Labels on bars
        for i, (bv, fv) in enumerate(zip(base_vals, ft_vals)):
            ax.text(bv + 0.3, i + bar_h / 2, f"{bv:.2f}",
                    va="center", fontsize=9, color="lightcoral",
                    fontweight="bold")
            ax.text(fv + 0.3, i - bar_h / 2, f"{fv:.2f}",
                    va="center", fontsize=9, color="seagreen",
                    fontweight="bold")

        ax.set_yticks(y)
        ax.set_yticklabels(clean_names)
        ax.set_xlabel("F1 Score (0–100)")
        n_samples = (full_eval_result or baseline_eval).get("num_samples", "?")
        ax.set_title(f"Full Eval ({n_samples} samples)")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No eval data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title("Full Eval")

    fig.suptitle(os.path.basename(os.path.dirname(path)), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {path}")


class ESTrainer:
    """Evolution Strategies trainer for LLM weight optimization.

    Args:
        model: The model whose weights are perturbed and updated.
        param_names: List of parameter names to optimize (e.g. base LLM
            params only, excluding memory-policy params).
        evaluate_fn: Callable(model) -> float. Called after perturbation
            to get a scalar reward. Should run inference + scoring.
        config: ES hyperparameters.
        pre_step_fn: Optional callable() invoked once per iteration before
            evaluating population members. Use this to resample a shared
            data batch so all members are compared on the same inputs.
        full_eval_fn: Optional callable(model) -> dict for full evaluation
            at end of training. Should return {"scores": {"task": value}, ...}.
        metadata: Optional dict of extra info to save in config.json
            alongside ESConfig fields (e.g. namm_checkpoint path,
            run_config name).
    """

    def __init__(self, model, param_names: list[str],
                 evaluate_fn, config: ESConfig,
                 validate_fn=None, pre_step_fn=None,
                 full_eval_fn=None, metadata=None):
        self.model = model
        self.param_names = param_names
        self.evaluate_fn = evaluate_fn
        self.validate_fn = validate_fn
        self.pre_step_fn = pre_step_fn
        self.full_eval_fn = full_eval_fn
        self.config = config
        self.metadata = metadata or {}

        # Verify all param_names exist in model
        model_param_names = {n for n, _ in model.named_parameters()}
        missing = set(param_names) - model_param_names
        if missing:
            raise ValueError(
                f"param_names contains {len(missing)} names not found in "
                f"model.named_parameters(). First 5: {list(missing)[:5]}"
            )

    def train(self):
        """Run the ES optimization loop."""
        cfg = self.config
        np.random.seed(cfg.initial_seed)

        # Setup logging
        log_dir = os.path.join(cfg.log_dir, f"es_s{cfg.sigma}_a{cfg.alpha}_"
                               f"p{cfg.population_size}_n{cfg.num_iterations}_"
                               f"{cfg.noise_mode}")
        writer = setup_tensorboard(log_dir)
        checkpoint_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save full config so we can reproduce this experiment
        config_dict = asdict(cfg)
        config_dict.update(self.metadata)
        config_path = os.path.join(log_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        print(f"Config saved to {config_path}")

        print(f"ES Training: pop={cfg.population_size}, iter={cfg.num_iterations}, "
              f"sigma={cfg.sigma}, alpha={cfg.alpha}, mode={cfg.noise_mode}")
        print(f"Optimizing {len(self.param_names)} parameters")
        print(f"Logging to {log_dir}")

        # Baseline full eval before training
        baseline_eval = None
        if self.full_eval_fn:
            print("\nRunning baseline evaluation (before training)...")
            eval_start = time.time()
            baseline_eval = self.full_eval_fn(self.model)
            eval_time = time.time() - eval_start
            baseline_eval["time_s"] = round(eval_time, 1)
            print(f"Baseline eval complete in {eval_time:.1f}s")
            for k, v in baseline_eval.get("scores", {}).items():
                print(f"  {_clean_task_name(k)}: {v:.2f}")

        # Collect metrics for results.json
        history = {"mean": [], "min": [], "max": [], "time": []}

        training_start = time.time()

        for iteration in range(cfg.num_iterations):
            iter_start = time.time()
            force_memory_cleanup()

            # Generate random seeds for this iteration's population
            seeds = np.random.randint(
                0, 2**30, size=cfg.population_size, dtype=np.int64
            ).tolist()

            # Resample shared data batch for this iteration (if provided)
            if self.pre_step_fn is not None:
                self.pre_step_fn()

            # Evaluate each population member
            rewards = []
            for member_idx, seed in enumerate(seeds):
                perturb_weights(
                    self.model, seed, cfg.sigma,
                    self.param_names, cfg.noise_mode
                )

                reward = self.evaluate_fn(self.model)
                rewards.append(reward)

                restore_weights(
                    self.model, seed, cfg.sigma,
                    self.param_names, cfg.noise_mode
                )

                force_memory_cleanup()

            # Normalize rewards
            rewards_arr = np.array(rewards, dtype=np.float32)
            normalized = (
                (rewards_arr - rewards_arr.mean())
                / (rewards_arr.std() + 1e-8)
            )

            # Apply ES update to base weights
            apply_es_update(
                self.model, seeds, normalized, cfg.sigma, cfg.alpha,
                self.param_names, cfg.population_size, cfg.noise_mode
            )

            iter_time = time.time() - iter_start

            # Log metrics
            mean_r = rewards_arr.mean().item()
            min_r = rewards_arr.min().item()
            max_r = rewards_arr.max().item()

            history["mean"].append(round(mean_r, 6))
            history["min"].append(round(min_r, 6))
            history["max"].append(round(max_r, 6))
            history["time"].append(round(iter_time, 1))

            writer.add_scalar("reward/mean", mean_r, iteration)
            writer.add_scalar("reward/min", min_r, iteration)
            writer.add_scalar("reward/max", max_r, iteration)
            writer.add_scalar("time/iteration_sec", iter_time, iteration)

            if torch.cuda.is_available():
                mem_mb = torch.cuda.memory_allocated() / 1024**2
                writer.add_scalar("gpu/memory_mb", mem_mb, iteration)

            print(f"[{iteration + 1}/{cfg.num_iterations}] "
                  f"mean={mean_r:.4f} min={min_r:.4f} max={max_r:.4f} "
                  f"time={iter_time:.1f}s")

            # Checkpoint
            if (iteration + 1) % cfg.checkpoint_every == 0:
                self._save_checkpoint(checkpoint_dir, iteration + 1)

            # Validation on unperturbed weights (kept for TensorBoard only)
            if self.validate_fn and (iteration + 1) % cfg.eval_every == 0:
                val_start = time.time()
                val_reward = self.validate_fn(self.model)
                val_time = time.time() - val_start
                writer.add_scalar("reward/val", val_reward, iteration)
                print(f"  val={val_reward:.4f} ({val_time:.1f}s)")

        total_time = time.time() - training_start
        print(f"Training complete in {total_time:.1f}s ({total_time / 3600:.1f}h)")

        # Final checkpoint
        self._save_checkpoint(checkpoint_dir, cfg.num_iterations, final=True)
        writer.close()

        # Full evaluation on fine-tuned weights
        full_eval_result = None
        if self.full_eval_fn:
            print("\nRunning full evaluation (after training)...")
            eval_start = time.time()
            full_eval_result = self.full_eval_fn(self.model)
            eval_time = time.time() - eval_start
            full_eval_result["time_s"] = round(eval_time, 1)
            print(f"Full eval complete in {eval_time:.1f}s")
            for k, v in full_eval_result.get("scores", {}).items():
                print(f"  {_clean_task_name(k)}: {v:.2f}")

        # Save results
        results_path = os.path.join(log_dir, "results.json")
        _save_results_json(results_path, config_dict, history, baseline_eval,
                           full_eval_result, total_time)

        plot_path = os.path.join(log_dir, "results.png")
        _save_results_plot(plot_path, history, baseline_eval, full_eval_result)

        print("Done.")

    def _save_checkpoint(self, checkpoint_dir, iteration, final=False):
        """Save model state dict for the optimized parameters."""
        suffix = "final" if final else f"iter{iteration}"
        path = os.path.join(checkpoint_dir, f"es_checkpoint_{suffix}.pt")

        state = {}
        all_params = dict(self.model.named_parameters())
        for name in self.param_names:
            state[name] = all_params[name].detach().cpu()

        torch.save(state, path)
        print(f"  Checkpoint saved: {path}")
