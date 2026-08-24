
from mjlab.tasks.loco_manip.config.g1.rl_cfg import unitree_g1_distillation_runner_cfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import unitree_g1_loco_manip_env_cfg
from .rl_cfg import unitree_g1_distillation_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Distill-LocoManip-BRAVE-G1",
  env_cfg=unitree_g1_loco_manip_env_cfg(),
  play_env_cfg=unitree_g1_loco_manip_env_cfg(play=True),
  rl_cfg=unitree_g1_distillation_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
