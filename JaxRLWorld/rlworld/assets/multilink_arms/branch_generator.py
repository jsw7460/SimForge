def generate_branched_arm_urdf(
    n_branches: int = 4,
    links_per_branch: int = 5,
    link_length: float = 0.1,
    mass_range: tuple[float, float] = (0.5, 2.0),
    branch_angle_deg: float = 45.0,
    output_path: str = "branched_arm.urdf"
):
    """Generate URDF for branched manipulator with N branches, each having M links.

    Structure:
        base_link
            ├── branch_1_link_1 ── branch_1_link_2 ── ... ── branch_1_link_M
            ├── branch_2_link_1 ── branch_2_link_2 ── ... ── branch_2_link_M
            ├── ...
            └── branch_N_link_1 ── branch_N_link_2 ── ... ── branch_N_link_M

    Args:
        n_branches: Number of branches from base.
        links_per_branch: Number of links per branch.
        link_length: Length of each link.
        mass_range: (min_mass, max_mass) for links.
        branch_angle_deg: Angle of each branch from vertical (z-axis).
        output_path: Output URDF file path.
    """
    import numpy as np

    radius = 0.02
    branch_angle_rad = np.radians(branch_angle_deg)

    urdf = f'''<?xml version="1.0"?>
<robot name="branched_arm_{n_branches}x{links_per_branch}">

  <!-- Base link (fixed) -->
  <link name="base_link">
    <visual>
      <geometry><sphere radius="0.05"/></geometry>
    </visual>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
'''

    for branch_idx in range(n_branches):
        # Distribute branches evenly around z-axis
        azimuth = 2 * np.pi * branch_idx / n_branches

        # Branch direction (tilted outward from vertical)
        roll = branch_angle_rad * np.cos(azimuth)
        pitch = branch_angle_rad * np.sin(azimuth)

        # Mass increases along each branch
        masses = np.linspace(mass_range[0], mass_range[1], links_per_branch)

        prev_link = "base_link"

        for link_idx in range(links_per_branch):
            is_end_effector = (link_idx == links_per_branch - 1)

            if is_end_effector:
                link_name = f"branch_{branch_idx + 1}_ee"
            else:
                link_name = f"branch_{branch_idx + 1}_link_{link_idx + 1}"

            joint_name = f"branch_{branch_idx + 1}_joint_{link_idx + 1}"

            mass = masses[link_idx]
            ixx = (1 / 12) * mass * (3 * radius ** 2 + link_length ** 2)
            izz = (1 / 2) * mass * radius ** 2

            # First link of branch: offset from base with branch angle
            # Subsequent links: straight continuation
            if link_idx == 0:
                origin_xyz = "0 0 0"
                origin_rpy = f"{roll:.4f} {pitch:.4f} 0"
            else:
                origin_xyz = f"0 0 {link_length}"
                origin_rpy = "0 0 0"

            # Alternate joint axes within branch
            axis = "1 0 0" if link_idx % 2 == 0 else "0 1 0"

            urdf += f'''
  <!-- Branch {branch_idx + 1}, Link {link_idx + 1} -->
  <link name="{link_name}">
    <visual>
      <origin xyz="0 0 {link_length / 2}"/>
      <geometry><cylinder radius="{radius}" length="{link_length}"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 {link_length / 2}"/>
      <geometry><cylinder radius="{radius}" length="{link_length}"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 {link_length / 2}"/>
      <mass value="{mass:.4f}"/>
      <inertia ixx="{ixx:.6f}" ixy="0" ixz="0" iyy="{ixx:.6f}" iyz="0" izz="{izz:.6f}"/>
    </inertial>
  </link>

  <joint name="{joint_name}" type="revolute">
    <parent link="{prev_link}"/>
    <child link="{link_name}"/>
    <origin xyz="{origin_xyz}" rpy="{origin_rpy}"/>
    <axis xyz="{axis}"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>
'''
            prev_link = link_name

    urdf += '</robot>'

    with open(output_path, 'w') as f:
        f.write(urdf)

    print(f"Generated {n_branches}-branch arm URDF ({links_per_branch} links each): {output_path}")
    return output_path


if __name__ == "__main__":
    generate_branched_arm_urdf(
        n_branches=4,
        links_per_branch=4,
        branch_angle_deg=0.0,
        output_path="branched_4x4_arm.urdf"
    )