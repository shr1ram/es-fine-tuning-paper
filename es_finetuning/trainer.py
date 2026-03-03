"""Core Evolution Strategies training loop.

The ESTrainer does NOT perform inference or scoring — it accepts an
evaluate_fn callable that returns a scalar reward for the current model
state. This keeps all model/task-specific logic in the caller.
"""

import os
import time

import numpy as np
import torch

from .config import ESConfig
from .noise import apply_es_update, perturb_weights, restore_weights
from .utils import force_memory_cleanup, setup_tensorboard


class ESTrainer:
    """Evolution Strategies trainer for LLM weight optimization.

    Args:
        model: The model whose weights are perturbed and updated.
        param_names: List of parameter names to optimize (e.g. base LLM
            params only, excluding memory-policy params).
        evaluate_fn: Callable(model) -> float. Called after perturbation
            to get a scalar reward. Should run inference + scoring.
        config: ES hyperparameters.
        validate_fn: Optional callable(model) -> float for validation on
            unperturbed weights. Called every config.eval_every iterations.
    """

    def __init__(self, model, param_names: list[str],
                 evaluate_fn, config: ESConfig,
                 validate_fn=None):
        self.model = model
        self.param_names = param_names
        self.evaluate_fn = evaluate_fn
        self.validate_fn = validate_fn
        self.config = config

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

        print(f"ES Training: pop={cfg.population_size}, iter={cfg.num_iterations}, "
              f"sigma={cfg.sigma}, alpha={cfg.alpha}, mode={cfg.noise_mode}")
        print(f"Optimizing {len(self.param_names)} parameters")
        print(f"Logging to {log_dir}")

        training_start = time.time()

        for iteration in range(cfg.num_iterations):
            iter_start = time.time()
            force_memory_cleanup()

            # Generate random seeds for this iteration's population
            seeds = np.random.randint(
                0, 2**30, size=cfg.population_size, dtype=np.int64
            ).tolist()

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

            # Validation on unperturbed weights
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
