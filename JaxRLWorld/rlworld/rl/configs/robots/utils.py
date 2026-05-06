def reflected_inertia_simple(rotor_inertia: float, gear_ratio: float) -> float:
    """Compute reflected inertia: I_rotor * gear_ratio^2."""
    return rotor_inertia * gear_ratio**2


def reflected_inertia_from_two_stage_planetary(
    rotor_inertia: tuple[float, float, float],
    gear_ratio: tuple[float, float, float],
) -> float:
    """Compute reflected inertia of a two-stage planetary gearbox."""
    r1 = rotor_inertia[0] * (gear_ratio[1] * gear_ratio[2]) ** 2
    r2 = rotor_inertia[1] * gear_ratio[2] ** 2
    r3 = rotor_inertia[2]
    return r1 + r2 + r3
