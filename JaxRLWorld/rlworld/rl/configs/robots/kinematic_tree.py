import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jax.numpy as jnp


__all__ = ["KinematicTree"]


class KinematicTree:
    """
    Extract kinematic tree from URDF or MJCF file.
    """

    def __init__(self, urdf_path: str = None, mjcf_path: str = None):
        """
        Args:
            urdf_path: Path to URDF file
            mjcf_path: Path to MJCF file
        """
        if urdf_path is None and mjcf_path is None:
            raise ValueError("Either urdf_path or mjcf_path must be provided")

        self.urdf_path = urdf_path
        self.mjcf_path = mjcf_path
        self.name = "robot"

        self.parent_indices = []
        self.children_indices = []
        self.joints = []
        self.links = []

        self._bottom_up_order = None
        self._depth_cache = None
        self._adjacency_cache = None

        if urdf_path is not None:
            self._parse_urdf()
        else:
            self._parse_mjcf()

    def _parse_mjcf(self):
        """Parse MJCF file and build kinematic tree"""
        tree = ET.parse(self.mjcf_path)
        root = tree.getroot()

        model_elem = root if root.tag == 'mujoco' else root.find('mujoco')
        if model_elem is not None:
            self.name = model_elem.get('model', 'mjcf_robot')

        worldbody = root.find('.//worldbody')
        if worldbody is None:
            raise ValueError("MJCF file has no worldbody")

        # Try MuJoCo first, fallback to XML parsing
        mj_model = None
        try:
            import mujoco
            mj_model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        except (ValueError, Exception):
            pass

        # Collect all bodies
        bodies = []

        def collect_bodies(elem, parent_idx):
            for body_elem in elem.findall('body'):
                body_name = body_elem.get('name', f'body_{len(bodies)}')
                body_idx = len(bodies)
                bodies.append({
                    'name': body_name,
                    'elem': body_elem,
                    'parent_idx': parent_idx
                })
                collect_bodies(body_elem, body_idx)

        collect_bodies(worldbody, -1)

        # Build links with mass
        self.links = []
        for idx, body in enumerate(bodies):
            body_name = body['name']

            if mj_model is not None:
                import mujoco
                mj_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                mass = mj_model.body_mass[mj_body_id] if mj_body_id >= 0 else 1.0
            else:
                mass = self._get_body_mass(body['elem'])

            self.links.append({
                "index": idx,
                "name": body_name,
                "mass": mass
            })

        # Build parent/children indices
        self.parent_indices = [body['parent_idx'] for body in bodies]
        self.children_indices = [[] for _ in range(len(bodies))]
        for child_idx, parent_idx in enumerate(self.parent_indices):
            if parent_idx != -1:
                self.children_indices[parent_idx].append(child_idx)

        # Build joints
        self.joints = []
        for body_idx, body in enumerate(bodies):
            if body['parent_idx'] == -1:
                continue
            for joint_elem in body['elem'].findall('joint'):
                joint_type = joint_elem.get('type', 'hinge')
                if joint_type == 'fixed':
                    continue
                self.joints.append({
                    "index": len(self.joints),
                    "parent_link": body['parent_idx'],
                    "child_link": body_idx,
                    "name": joint_elem.get('name', f'joint_{len(self.joints)}'),
                    "type": joint_type
                })

    def _get_body_mass(self, body_elem) -> float:
        """Extract mass from body element"""
        inertial = body_elem.find('inertial')
        if inertial is not None:
            mass = inertial.get('mass')
            if mass is not None:
                return float(mass)

        total_mass = 0.0
        for geom in body_elem.findall('geom'):
            mass = geom.get('mass')
            if mass is not None:
                total_mass += float(mass)

        return total_mass if total_mass > 0 else 1.0

    def _parse_urdf(self):
        """Parse URDF file and build kinematic tree"""
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()

        robot_name = root.get('name', 'robot')
        self.name = robot_name

        all_links = {link.get('name'): link for link in root.findall('link')}
        joints = list(root.findall('joint'))

        sensor_keywords = [
            'camera', 'depth', 'rgb', 'lidar', 'laser', 'radar',
            'imu', 'gps', 'sonar', 'sensor', 'optical', 'logo', 'd435', 'mid360',
            'contour'
        ]

        links = {}
        for name, link in all_links.items():
            is_sensor = any(keyword in name.lower() for keyword in sensor_keywords)
            if is_sensor:
                continue
            links[name] = link

        # Filter out low-mass links (mass <= 0.01)
        filtered_links = {}
        for name, link_elem in links.items():
            inertial_elem = link_elem.find('inertial')
            mass = 0.0
            if inertial_elem is not None:
                mass_elem = inertial_elem.find('mass')
                if mass_elem is not None:
                    mass = float(mass_elem.get('value', 0))
            if mass > 0.01:
                filtered_links[name] = link_elem

        links = filtered_links

        link_names = list(links.keys())
        link_to_idx = {name: idx for idx, name in enumerate(link_names)}

        self.links = []
        for idx, name in enumerate(link_names):
            link_elem = links[name]
            inertial_elem = link_elem.find('inertial')

            mass = 0.0
            if inertial_elem is not None:
                mass_elem = inertial_elem.find('mass')
                if mass_elem is not None:
                    mass = float(mass_elem.get('value', 0))

            self.links.append({
                "index": idx,
                "name": name,
                "mass": mass
            })

        child_links = set()
        for joint in joints:
            child_elem = joint.find('child')
            parent_elem = joint.find('parent')

            if child_elem is not None and parent_elem is not None:
                child_link = child_elem.get('link')
                parent_link = parent_elem.get('link')

                if child_link in link_to_idx and parent_link in link_to_idx:
                    child_links.add(child_link)

        root_links = [name for name in link_names if name not in child_links]

        if len(root_links) != 1:
            raise ValueError(f"Expected 1 root link, found {len(root_links)}: {root_links}")

        self.parent_indices = [-1] * len(link_names)

        joint_info_list = []

        for joint in joints:
            joint_name = joint.get('name')
            joint_type = joint.get('type')

            parent_elem = joint.find('parent')
            child_elem = joint.find('child')

            if parent_elem is None or child_elem is None:
                continue

            parent_link = parent_elem.get('link')
            child_link = child_elem.get('link')

            if parent_link not in link_to_idx or child_link not in link_to_idx:
                continue

            parent_idx = link_to_idx[parent_link]
            child_idx = link_to_idx[child_link]

            self.parent_indices[child_idx] = parent_idx

            if joint_type != 'fixed':
                joint_info_list.append({
                    "index": len(joint_info_list),
                    "parent_link": parent_idx,
                    "child_link": child_idx,
                    "name": joint_name,
                    "type": joint_type
                })

        self.joints = joint_info_list

        self.children_indices = [[] for _ in range(len(link_names))]
        for child_idx, parent_idx in enumerate(self.parent_indices):
            if parent_idx != -1:
                self.children_indices[parent_idx].append(child_idx)

    # ===== Properties =====
    @property
    def num_bodies(self) -> int:
        return len(self.parent_indices)

    @property
    def root_idx(self) -> int:
        for i, parent in enumerate(self.parent_indices):
            if parent == -1:
                return i
        return 0

    @property
    def num_joints(self) -> int:
        return len(self.joints)

    @property
    def max_joint_idx(self) -> int:
        return len(self.joints) - 1 if self.joints else -1

    # ===== Joint Methods =====

    def get_joint_parent_link(self, joint_idx: int) -> int:
        if joint_idx >= len(self.joints):
            raise ValueError(f"No joint at index {joint_idx}")
        return self.joints[joint_idx]['parent_link']

    def get_joint_child_link(self, joint_idx: int) -> int:
        if joint_idx >= len(self.joints):
            raise ValueError(f"No joint at index {joint_idx}")
        return self.joints[joint_idx]['child_link']

    def get_active_joint_indices(self) -> list[int]:
        return list(range(len(self.joints)))

    # ===== Tree Traversal =====

    def get_parent(self, body_idx: int) -> int | None:
        if body_idx < 0 or body_idx >= len(self.parent_indices):
            return None
        return self.parent_indices[body_idx] if self.parent_indices[body_idx] != -1 else None

    def get_children(self, body_idx: int) -> list[int]:
        if body_idx < 0 or body_idx >= len(self.children_indices):
            return []
        return self.children_indices[body_idx]

    def traverse_bottom_up(self) -> list[int]:
        """Get body indices in bottom-up order (leaves → root)"""
        if self._bottom_up_order is not None:
            return self._bottom_up_order

        visited = [False] * self.num_bodies
        order = []

        def dfs_post_order(node):
            if visited[node]:
                return
            visited[node] = True
            for child in self.children_indices[node]:
                dfs_post_order(child)
            order.append(node)

        dfs_post_order(self.root_idx)
        self._bottom_up_order = order
        return order

    def get_ancestor_chain(self, body_idx: int) -> list[int]:
        """Get ancestor chain from root to body_idx (inclusive)"""
        chain = []
        current = body_idx

        while current != -1:
            chain.append(current)
            current = self.parent_indices[current]

        return list(reversed(chain))

    # ===== Depth Methods =====

    def get_depth(self, body_idx: int) -> int:
        """Get depth level of body in the tree (root = 0)"""
        if self._depth_cache is None:
            self._compute_depths()
        return self._depth_cache[body_idx]

    def _compute_depths(self):
        """Compute and cache depth for all bodies"""
        self._depth_cache = [0] * self.num_bodies

        for body_idx in range(self.num_bodies):
            depth = 0
            current = body_idx
            while self.parent_indices[current] != -1:
                depth += 1
                current = self.parent_indices[current]
            self._depth_cache[body_idx] = depth

    def get_bodies_at_depth(self, depth: int) -> list[int]:
        """Get all body indices at a specific depth level"""
        if self._depth_cache is None:
            self._compute_depths()
        return [i for i in range(self.num_bodies) if self._depth_cache[i] == depth]

    def get_max_depth(self) -> int:
        """Get maximum depth of the tree"""
        if self._depth_cache is None:
            self._compute_depths()
        return max(self._depth_cache)

    def get_depth_groups(self) -> dict[int, list[int]]:
        """Group body indices by depth level"""
        if self._depth_cache is None:
            self._compute_depths()

        depth_groups = {}
        for body_idx in range(self.num_bodies):
            depth = self._depth_cache[body_idx]
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append(body_idx)

        return depth_groups

    # ===== Adjacency Matrix =====

    def get_adjacency_matrix(self) -> "jnp.ndarray":
        """Get bidirectional adjacency matrix as JAX array (cached)."""
        if self._adjacency_cache is not None:
            return self._adjacency_cache

        import jax.numpy as jnp

        num_bodies = self.num_bodies
        adjacency = jnp.zeros((num_bodies, num_bodies), dtype=jnp.float32)

        for child_idx, parent_idx in enumerate(self.parent_indices):
            if parent_idx != -1:
                adjacency = adjacency.at[parent_idx, child_idx].set(1.0)
                adjacency = adjacency.at[child_idx, parent_idx].set(1.0)

        self._adjacency_cache = adjacency
        return adjacency

    # ===== Display Methods =====

    def __repr__(self) -> str:
        source = "URDF" if self.urdf_path else "MJCF"
        return (f"KinematicTree(source='{source}', name='{self.name}', "
                f"num_bodies={self.num_bodies}, "
                f"num_joints={self.num_joints}, "
                f"root={self.root_idx})")

    def print_links(self) -> None:
        """Print links in formatted table"""
        print(f"\n{'=' * 50}")
        print(f"Links ({len(self.links)} total)")
        print(f"{'=' * 50}")
        print(f"{'Index':<8}{'Name':<30}{'Mass':<12}")
        print(f"{'-' * 50}")
        for link in self.links:
            print(f"{link['index']:<8}{link['name']:<30}{link['mass']:<12.6f}")

    def print_joints(self) -> None:
        """Print joints in formatted table"""
        print(f"\n{'=' * 60}")
        print(f"Joints ({len(self.joints)} total)")
        print(f"{'=' * 60}")
        print(f"{'Index':<8}{'Name':<25}{'Parent':<10}{'Child':<10}{'Type':<10}")
        print(f"{'-' * 60}")
        for joint in self.joints:
            print(
                f"{joint['index']:<8}{joint['name']:<25}{joint['parent_link']:<10}{joint['child_link']:<10}{joint['type']:<10}")

    def print_tree(self) -> None:
        """Print tree structure visually"""
        print(f"\n{'=' * 50}")
        print(f"Kinematic Tree: {self.name}")
        print(f"{'=' * 50}")

        def print_node(idx: int, indent: int = 0):
            link_name = self.links[idx]['name']
            prefix = "  " * indent + ("└─ " if indent > 0 else "")
            print(f"{prefix}[{idx}] {link_name}")
            for child in self.children_indices[idx]:
                print_node(child, indent + 1)

        print_node(self.root_idx)
