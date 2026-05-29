import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_landmark_measurement(graph, initial_estimate, result):
    # Get optimized pose X(4) and landmark L(2)
    pose4 = result.atPose2(X(4))
    landmark2 = result.atPoint2(L(2))

    # Compute bearing and range from X(4) to L(2)
    bearing = pose4.bearing(landmark2)
    distance = pose4.range(landmark2)

    # Add bearing-range measurement factor
    graph.add(
        gtsam.BearingRangeFactor2D(
            X(4),
            L(2),
            bearing,
            distance,
            MEASUREMENT_NOISE
        )
    )

    return graph