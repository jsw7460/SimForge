"""T1 motion-tracking on ``goal_kick.npz`` (Newton).

Thin wrapper around :func:`rlworld.scripts.t1_tracking._train_motion.train_single_motion`.
Run name: ``T1_Tracking_NT_MLP_goal_kick``.
"""
from rlworld.scripts.t1_tracking._train_motion import train_single_motion


if __name__ == "__main__":
    train_single_motion(sim_type="newton", motion_name="goal_kick")
