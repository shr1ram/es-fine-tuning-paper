from dataclasses import dataclass


@dataclass
class ESConfig:
    """Configuration for Evolution Strategies fine-tuning."""

    sigma: float = 0.001  # Noise scale for weight perturbations
    alpha: float = 0.0005  # Learning rate
    population_size: int = 8  # Population members per iteration
    num_iterations: int = 150  # Total ES iterations
    noise_mode: str = "correlated"  # "correlated" (shared seed) or "iid" (per-param seed)
    initial_seed: int = 33  # Initial random seed for numpy
    mini_batch_size: int = 4  # Samples per population eval (passed to evaluate_fn)
    checkpoint_every: int = 25  # Save model every N iterations
    eval_every: int = 25  # Run validation every N iterations
    log_dir: str = "es_runs"  # Directory for TensorBoard logs and checkpoints
