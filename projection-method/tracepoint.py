import numpy as np

class TracePoint:
    # t signifies milliseconds since start of path trace.
    # beginning signifies if this is the start of a "sequence" - i.e. to handle disconnected points
    def __init__(self, pos, t=0, beginning=True):
        self.pos = pos
        self.t = t

    def transform(self, rotation_matrix):
        # Applies a rotation matrix which transforms the point.
        transformed = np.dot(rotation_matrix, np.array(self.pos))
        self.pos = tuple(transformed)

    def __str__(self):
        return str(pos) + " " + str(t)
