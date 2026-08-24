"""RL configuration for Unitree G1 BRAVE loco-manipulation task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlDistillationRunnerCfg,
  RslRlDistillationAlgorithmCfg,
  RslRlDistillationStudentTeacherCfg,
)


def unitree_g1_distillation_runner_cfg() -> RslRlDistillationRunnerCfg:
  """Create RL runner configuration for Unitree G1 BRAVE loco-manipulation task."""
  return RslRlDistillationRunnerCfg(
    num_steps_per_env=120,
    max_iterations=10_000,
    save_interval=500,
    experiment_name="g1_loco_manip",
    obs_groups=dict({"policy": ["policy"], "teacher": ["policy"]}),
    policy=RslRlDistillationStudentTeacherCfg(
      init_noise_std=0.1,
      noise_std_type="scalar",
      student_obs_normalization=False,
      teacher_obs_normalization=False,
      student_hidden_dims=(128, 128, 128),
      teacher_hidden_dims=(512, 256, 128),
      activation="elu",
    ),
    algorithm=RslRlDistillationAlgorithmCfg(
      num_learning_epochs=2,
      learning_rate=1.0e-3,
      gradient_length=15,
    )
  )
