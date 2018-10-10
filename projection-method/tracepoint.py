import numpy as np

class TracePoint:
    # t signifies milliseconds since start of path trace.
    # beginning signifies if this is the start of a "sequence" - i.e. to handle disconnected points
    # pos is a (x, y, z) tuple
    def __init__(self, pos, t, beginning=True):
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

    def __init__(self):
        self.path = []

    def add(self, tracepoint):
        self.path.append(tracepoint)

    def transform(self, rotation_matrix):
        transformed = []
        for point in self.path:
            point.transform(rotation_matrix)

    def normalizeAspect(self, bigs, smalls, big_lower, big_higher, aspectRatio):
        new_bigs = np.interp(bigs, (bigs.min(), bigs.max()), (big_lower, big_higher))

        small_spread = aspectRatio * (big_higher - big_lower)
        small_lower = (((big_higher - big_lower) - small_spread) / 2) + big_lower
        small_higher = big_higher - (((big_higher - big_lower) - small_spread) / 2)
        new_smalls = np.interp(smalls, (smalls.min(), smalls.max()), (small_lower, small_higher))

        return (new_bigs, new_smalls)

    def normalize(self, low_bound=0, high_bound=1, preserve_aspect_ratio=True):
        """ TODO this ignores z when preserving aspect ratio """
        xs = np.array([p.pos[0] for p in self.path])
        ys = np.array([p.pos[1] for p in self.path])
        zs = np.array([p.pos[2] for p in self.path])

        if preserve_aspect_ratio:
            xDiff = xs.max() - xs.min()
            yDiff = ys.max() - ys.min()
            if xDiff > yDiff:
                # x the bigger boi
                xs, ys = self.normalizeAspect(xs, ys, low_bound, high_bound, yDiff / xDiff)
            else:
                # y the bigger boi
                ys, xs = self.normalizeAspect(ys, xs, low_bound, high_bound, xDiff / yDiff)

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