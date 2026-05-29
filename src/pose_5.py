import copy
import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)


def add_pose(graph, initial_estimate, pose_5):
    pose_4 = initial_estimate.atPose2(X(4))

    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )

    return graph, initial_estimate


def add_landmark_measurement(graph, result, pose_5, landmark):
    landmark_point = result.atPoint2(L(landmark))

    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )

    return graph


def optimize(graph, initial_estimate):
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()

    return result


def minimize_marginals(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    best_chosen_landmark_marginal = float("inf")
    best_return_marginals = None

    for pose_name, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            test_graph = copy.deepcopy(graph)
            test_estimate = copy.deepcopy(initial_estimate)

            test_graph, test_estimate = add_pose(test_graph, test_estimate, pose_5)

            result = optimize(test_graph, test_estimate)

            test_graph = add_landmark_measurement(test_graph, result, pose_5, landmark)

            result = optimize(test_graph, test_estimate)

            marginals = gtsam.Marginals(test_graph, result)

            chosen_landmark_marginal = marginals.marginalCovariance(L(landmark)).sum()

            total_landmark_marginals = (
                marginals.marginalCovariance(L(1)).sum()
                + marginals.marginalCovariance(L(2)).sum()
            )

            if chosen_landmark_marginal < best_chosen_landmark_marginal:
                best_chosen_landmark_marginal = chosen_landmark_marginal
                best_pose = pose_name
                best_landmark = landmark
                best_return_marginals = total_landmark_marginals

    return best_pose, best_landmark, best_return_marginals


def minimize_errors(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    best_error = float("inf")

    list_of_errors = []

    for pose_name, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            test_graph = copy.deepcopy(graph)
            test_estimate = copy.deepcopy(initial_estimate)

            test_graph, test_estimate = add_pose(test_graph, test_estimate, pose_5)

            result = optimize(test_graph, test_estimate)

            test_graph = add_landmark_measurement(test_graph, result, pose_5, landmark)

            result = optimize(test_graph, test_estimate)

            error = test_graph.error(result)
            list_of_errors.append(error)

            if error < best_error:
                best_error = error
                best_pose = pose_name
                best_landmark = landmark

    sum_of_errors = sum(list_of_errors)

    return best_pose, best_landmark, sum_of_errors