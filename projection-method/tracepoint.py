import numpy as np

class TracePoint:
    # t signifies milliseconds since start of path trace.
    # pos is a (x, y, z) tuple
    def __init__(self, pos, t):
        self.pos = pos
        self.t = t

    def transform(self, rotation_matrix):
        # Applies a rotation matrix which transforms the point.
        transformed = np.dot(rotation_matrix, np.array(self.pos))
        self.pos = tuple(transformed)

    def __str__(self):
        return str(pos) + " " + str(t)

class TracePath:
    """Consists of a list of tracepoints."""

    def __init__(self, path=None, checkpoint_indices=None):
        self.path = path
        self.checkpoint_indices = checkpoint_indices
        if not path:
            self.path = []
        if not checkpoint_indices:
            self.checkpoint_indices = set()

    # NOTE: assumes there are no "skipped" frames, and that path is length 2 at least, and t is in ms
    def fps(self):
        return 1 / ((self.path[1].t - self.path[0].t) / 1000)

    def add(self, tracepoint, checkpoint=False):
        if checkpoint:
            self.checkpoint_indices.add(len(self.path))

        self.path.append(tracepoint)

    def transform(self, rotation_matrix):
        for point in self.path:
            point.transform(rotation_matrix)

    def normalize_preserving_aspect(self, big_sequence, small_sequences, lower_bound, upper_bound):
        big_min, big_max = big_sequence.min(), big_sequence.max()
        big_bound_spread = upper_bound - lower_bound
        new_big_sequence = np.interp(big_sequence, (big_min, big_max), (lower_bound, upper_bound))

        new_small_sequences = []
        for small_sequence in small_sequences:
            small_min, small_max = small_sequence.min(), small_sequence.max()

            aspect_ratio = (small_max - small_min) / (big_max - big_min) # Less than 1
            small_bound_spread = aspect_ratio * big_bound_spread
            small_lower_bound = lower_bound + ((big_bound_spread - small_bound_spread) / 2)
            small_upper_bound = upper_bound - ((big_bound_spread - small_bound_spread) / 2)

            new_small_sequence = np.interp(small_sequence, (small_min, small_max), (small_lower_bound, small_upper_bound))
            new_small_sequences.append(new_small_sequence)

        return (new_big_sequence, new_small_sequences)

    def normalize(self, low_bound=0, high_bound=1, preserve_aspect_ratio=True):
        xs = np.array([p.pos[0] for p in self.path])
        ys = np.array([p.pos[1] for p in self.path])
        zs = np.array([p.pos[2] for p in self.path])


        if preserve_aspect_ratio:
            x_diff = xs.max() - xs.min()
            y_diff = ys.max() - ys.min()
            z_diff = zs.max() - zs.min()
            max_difference = max(x_diff, y_diff, z_diff)
            if max_difference == x_diff:
                xs, others = self.normalize_preserving_aspect(xs, [ys, zs], low_bound, high_bound)
                ys = others[0]
                zs = others[1]
            elif max_difference == y_diff:
                ys, others = self.normalize_preserving_aspect(ys, [xs, zs], low_bound, high_bound)
                xs = others[0]
                zs = others[1]
            elif max_difference == z_diff:
                zs, others = self.normalize_preserving_aspect(zs, [xs, ys], low_bound, high_bound)
                xs = others[0]
                ys = others[1]


            normalized_positions = list(zip(
                xs, ys, zs
            ))
            
        else:
            normalized_positions = list(zip(
                np.interp(xs, (xs.min(), xs.max()), (low_bound, high_bound)),
                np.interp(ys, (ys.min(), ys.max()), (low_bound, high_bound)),
                np.interp(zs, (zs.min(), zs.max()), (low_bound, high_bound))
            ))

        for point, pos in zip(self.path, normalized_positions):
            point.pos = pos

    def time_sequence(self, coordinate):
        return [(p.t, p.pos[coordinate]) for p in self.path]

    def sequence(self, coordinate):
        return [p.pos[coordinate] for p in self.path]