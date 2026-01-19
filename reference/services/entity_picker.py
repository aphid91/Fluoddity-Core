import numpy as np
import moderngl


class EntityPicker:
    """Handles entity selection from screen coordinates."""

    def __init__(self, entity_buffer: moderngl.Buffer, entity_stride: int):
        """Initialize EntityPicker.

        Args:
            entity_buffer: GPU buffer containing entity data
            entity_stride: Number of floats per entity (e.g., 12 for pos:2 + vel:2 + size:1 + padding:3 + color:4)
        """
        self.entity_buffer = entity_buffer
        self.entity_stride = entity_stride

    def update_buffer(self, entity_buffer: moderngl.Buffer):
        """Update the entity buffer reference.

        Call this when the buffer is reallocated (e.g., world size change).

        Args:
            entity_buffer: New GPU buffer containing entity data
        """
        self.entity_buffer = entity_buffer

    def find_nearest_entity(self, tex_coords: tuple[float, float]) -> tuple[int, tuple[float, float], float]:
        """Find the entity closest to given texture coordinates.

        Args:
            tex_coords: (x, y) in texture space where (0,0) is top-left

        Returns:
            Tuple of (entity_index, (pos_x, pos_y), cohort_normalized)
            - entity_index: Index of the nearest entity
            - (pos_x, pos_y): World-space position of the entity (in [-1, 1] range)
            - cohort_normalized: Normalized cohort value (0-1) for parameter sweep calculations
        """
        ent_cache = np.frombuffer(self.entity_buffer.read(), dtype=np.float32)

        # Extract positions (every Nth float starting at 0 and 1)
        xs = ent_cache[0::self.entity_stride].copy()
        ys = ent_cache[1::self.entity_stride].copy()

        # Convert from [-1,1] to [0,1] texture space
        xs_tex = xs / 2.0 + 0.5
        ys_tex = ys / 2.0 + 0.5

        # Compute squared distances
        dx = xs_tex - tex_coords[0]
        dy = ys_tex - tex_coords[1]
        distances_sq = dx * dx + dy * dy

        nearest_idx = int(distances_sq.argmin())

        # Extract position and cohort for the nearest entity
        # Entity structure: pos(2) + vel(2) + size(1) + cohort(1) + padding(2) + color(4)
        pos_x = float(xs[nearest_idx])  # Already in world space [-1, 1]
        pos_y = float(ys[nearest_idx])
        cohort_normalized = float(ent_cache[nearest_idx * self.entity_stride + 5])  # Index 5 is cohort field

        return (nearest_idx, (pos_x, pos_y), cohort_normalized)
