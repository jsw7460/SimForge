from dataclasses import dataclass

from rlworld.rl.configs.observations import ObservationTermConfig
from rlworld.rl.configs.observations.noise import NoiseConfig
from rlworld.rl.envs.mdp.observations.genesis import proprioception, exteroception, state


@dataclass
class LocomotionObservations:
    """Standard observation set for locomotion tasks."""

    base_name: str

    # === Base linear velocity ===
    base_lin_vel_scale: float = 2.0
    base_lin_vel_noise: NoiseConfig | None = None
    base_lin_vel_clip: tuple[float, float] | None = None
    include_base_lin_vel: bool = True

    # === IMU angular velocity ===
    ang_vel_scale: float = 0.25
    ang_vel_noise: NoiseConfig | None = None
    ang_vel_clip: tuple[float, float] | None = None
    ang_vel_history: int = 0

    # === Projected gravity ===
    gravity_scale: float = 1.0
    gravity_noise: NoiseConfig | None = None
    gravity_clip: tuple[float, float] | None = None
    gravity_history: int = 0
    critic_gravity_history: int | None = None

    # === Command ===
    command_scale: float = 1.0

    # === DOF position ===
    dof_pos_scale: float = 1.0
    dof_pos_noise: NoiseConfig | None = None
    dof_pos_clip: tuple[float, float] | None = None
    dof_pos_history: int = 0
    include_dof_pos: bool = False

    # === DOF nominal difference ===
    nominal_difference_scale: float = 1.0
    nominal_difference_noise: NoiseConfig | None = None
    nominal_difference_clip: tuple[float, float] | None = None
    nominal_difference_history: int = 0
    include_nominal_difference: bool = True

    # === DOF velocity ===
    dof_vel_scale: float = 0.05
    dof_vel_noise: NoiseConfig | None = None
    dof_vel_clip: tuple[float, float] | None = None
    dof_vel_history: int = 0

    # === Previous actions ===
    prev_actions_scale: float = 1.0
    prev_actions_noise: NoiseConfig | None = None
    prev_actions_clip: tuple[float, float] | None = None
    prev_actions_history: int = 0

    def to_terms(self) -> list[ObservationTermConfig]:
        """Convert to list of ObservationTermConfig for actor."""
        terms = []

        if self.include_base_lin_vel:
            terms.append(ObservationTermConfig(
                state.base_lin_vel,
                scale=self.base_lin_vel_scale,
                noise=self.base_lin_vel_noise,
                clip=self.base_lin_vel_clip,
            ))

        terms.append(ObservationTermConfig(
                proprioception.imu_ang_vel,
                scale=self.ang_vel_scale,
                noise=self.ang_vel_noise,
                clip=self.ang_vel_clip,
                history_length=self.ang_vel_history,
                params={"base_name": self.base_name},
            ))

        terms.append(ObservationTermConfig(
                proprioception.projected_gravity,
                scale=self.gravity_scale,
                noise=self.gravity_noise,
                clip=self.gravity_clip,
                history_length=self.gravity_history,
            ))

        terms.append(ObservationTermConfig(
                exteroception.command,
                scale=self.command_scale,
            ))


        if self.include_dof_pos:
            terms.append(
                ObservationTermConfig(
                    proprioception.dof_pos,
                    scale=self.dof_pos_scale,
                    noise=self.dof_pos_noise,
                    clip=self.dof_pos_clip,
                    history_length=self.dof_pos_history,
                )
            )

        if self.include_nominal_difference:
            terms.append(
                ObservationTermConfig(
                    proprioception.dof_pos_nominal_difference,
                    scale=self.nominal_difference_scale,
                    noise=self.nominal_difference_noise,
                    clip=self.nominal_difference_clip,
                    history_length=self.nominal_difference_history,
                )
            )

        terms.append(ObservationTermConfig(
                proprioception.dof_vel,
                scale=self.dof_vel_scale,
                noise=self.dof_vel_noise,
                clip=self.dof_vel_clip,
                history_length=self.dof_vel_history,
            ))

        terms.append(ObservationTermConfig(
                proprioception.prev_processed_actions,
                scale=self.prev_actions_scale,
                noise=self.prev_actions_noise,
                clip=self.prev_actions_clip,
                history_length=self.prev_actions_history,
            ))

        return terms

    def to_critic_terms(self) -> list[ObservationTermConfig]:
        """Convert to list of ObservationTermConfig for critic."""
        gravity_hist = (
            self.critic_gravity_history
            if self.critic_gravity_history is not None
            else self.gravity_history
        )

        terms = []

        if self.include_base_lin_vel:
            terms.append(ObservationTermConfig(
                state.base_lin_vel,
                scale=self.base_lin_vel_scale,
                noise=self.base_lin_vel_noise,
                clip=self.base_lin_vel_clip,
            ))

        if self.include_dof_pos:
            terms.append(ObservationTermConfig(
                proprioception.dof_pos,
                scale=self.dof_pos_scale,
                noise=self.dof_pos_noise,
                clip=self.dof_pos_clip,
                history_length=self.dof_pos_history,
            ))

        if self.include_nominal_difference:
            terms.append(ObservationTermConfig(
                proprioception.dof_pos_nominal_difference,
                scale=self.nominal_difference_scale,
                noise=self.nominal_difference_noise,
                clip=self.nominal_difference_clip,
                history_length=self.nominal_difference_history,
            ))

        terms.extend([
            ObservationTermConfig(
                proprioception.imu_ang_vel,
                scale=self.ang_vel_scale,
                noise=self.ang_vel_noise,
                clip=self.ang_vel_clip,
                history_length=self.ang_vel_history,
                params={"base_name": self.base_name},
            ),
            ObservationTermConfig(
                proprioception.projected_gravity,
                scale=self.gravity_scale,
                noise=self.gravity_noise,
                clip=self.gravity_clip,
                history_length=gravity_hist,
            ),
            ObservationTermConfig(
                exteroception.command,
                scale=self.command_scale,
            ),
            ObservationTermConfig(
                proprioception.dof_vel,
                scale=self.dof_vel_scale,
                noise=self.dof_vel_noise,
                clip=self.dof_vel_clip,
                history_length=self.dof_vel_history,
            ),
            ObservationTermConfig(
                proprioception.prev_processed_actions,
                scale=self.prev_actions_scale,
                noise=self.prev_actions_noise,
                clip=self.prev_actions_clip,
                history_length=self.prev_actions_history,
            ),
        ])

        return terms