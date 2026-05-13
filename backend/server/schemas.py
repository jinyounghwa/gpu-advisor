from pydantic import BaseModel

class GPUQuery(BaseModel):
    model_name: str

class TrainingConfig(BaseModel):
    num_steps: int = 500
    learning_rate: float = 1e-4
    batch_size: int = 32
    seed: int = 42

class PipelineRequest(BaseModel):
    target_days: int = 30
    lookback_days: int = 30
    num_steps: int = 500
    batch_size: int = 32
    learning_rate: float = 1e-4
    seed: int = 42
    min_accuracy: float = 0.55
    min_avg_reward: float = 0.0
    max_abstain_ratio: float = 0.85
    max_safe_override_ratio: float = 0.90
    min_action_entropy: float = 0.25
    min_uplift_vs_buy: float = 0.0
    require_30d: bool = True
    run_training: bool = True
