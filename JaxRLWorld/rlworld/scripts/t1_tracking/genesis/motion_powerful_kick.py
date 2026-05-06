"""T1 motion-tracking on ``powerful_kick.npz`` (Genesis).

Thin wrapper around :func:`rlworld.scripts.t1_tracking._train_motion.train_single_motion`.
Run name: ``T1_Tracking_GS_MLP_powerful_kick``.
"""

from rlworld.scripts.t1_tracking._train_motion import train_single_motion

if __name__ == "__main__":
    train_single_motion(sim_type="genesis", motion_name="powerful_kick")
