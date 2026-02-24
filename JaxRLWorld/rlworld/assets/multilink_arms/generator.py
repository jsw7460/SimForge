def generate_n_link_arm_urdf(
    n_links: int = 30,
    link_length: float = 0.1,
    mass_range: tuple[float, float] = (0.5, 2.0),
    output_path: str = "n_link_arm.urdf"
):
    """Generate URDF for N-link serial manipulator with varying mass/inertia."""

    import numpy as np

    masses = np.linspace(mass_range[0], mass_range[1], n_links)

    urdf = f'''<?xml version="1.0"?>
<robot name="n_link_arm_{n_links}">

  <!-- Base link (fixed) -->
  <link name="base_link">
    <visual>
      <geometry><cylinder radius="0.05" length="0.02"/></geometry>
    </visual>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
'''

    prev_link = "base_link"

    for i in range(n_links):
        mass = masses[i]
        # Cylinder inertia: Ixx = Iyy = (1/12)*m*(3r^2 + h^2), Izz = (1/2)*m*r^2
        radius = 0.02
        ixx = (1 / 12) * mass * (3 * radius ** 2 + link_length ** 2)
        izz = (1 / 2) * mass * radius ** 2

        urdf += f'''
  <!-- Link {i + 1} -->
  <link name="link_{i + 1}">
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

  <joint name="joint_{i + 1}" type="revolute">
    <parent link="{prev_link}"/>
    <child link="link_{i + 1}"/>
    <origin xyz="0 0 {link_length if i > 0 else 0}" rpy="0 0 0"/>
    <axis xyz="{'1 0 0' if i % 2 == 0 else '0 1 0'}"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>
'''
        prev_link = f"link_{i + 1}"

    urdf += '</robot>'

    with open(output_path, 'w') as f:
        f.write(urdf)

    print(f"Generated {n_links}-link arm URDF: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_n_link_arm_urdf(
        n_links=20,
        link_length=0.1,
        output_path="20_link_arm.urdf"
    )