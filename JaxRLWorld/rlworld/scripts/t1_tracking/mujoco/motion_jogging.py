"""T1 motion-tracking on ``jogging.npz`` (MuJoCo (mjlab)).

Thin wrapper around :func:`rlworld.scripts.t1_tracking._train_motion.train_single_motion`.
Run name: ``T1_Tracking_MJ_MLP_jogging``.
"""

from rlworld.scripts.t1_tracking._train_motion import train_single_motion

if __name__ == "__main__":
    train_single_motion(sim_type="mujoco", motion_name="jogging")
