import numpy as np
import casadi as ca
from typing import Dict, Union, List, Tuple, Any

def compute_approximating_circle_radius(ego_length, ego_width) -> Tuple[Union[float, Any], Any]:
    """
    The dimension of the agent: length = 1.35, width = 1, height = 0.5
    Computes parameters of the circle approximation of the ego_vehicle

    :param ego_length: Length of ego vehicle
    :param ego_width: Width of ego vehicle
    :return: radius of circle approximation, circle center point distance
    """
    assert ego_length >= 0 and ego_width >= 0, 'Invalid vehicle dimensions = {}'.format([ego_length, ego_width])

    if np.isclose(ego_length, 0.0) and np.isclose(ego_width, 0.0):
        return 0.0, 0.0

    # Divide rectangle into 3 smaller rectangles
    square_length = ego_length / 3

    # Calculate minimum radius
    diagonal_square = np.sqrt((square_length / 2) ** 2 + (ego_width / 2) ** 2)

    # Round up value
    if diagonal_square > round(diagonal_square, 1):
        approx_radius = round(diagonal_square, 1) + 0.1
    else:
        approx_radius = round(diagonal_square, 1)

    return approx_radius, round(square_length * 2, 1)


def compute_centers_of_approximation_circles(x_position, y_position, v_length, v_width, orientation):
    """
    Compute three center coordinates of approximated circles of the vehicle.
    :param x_position: x position of the vehicle`s center
    :param y_position: y position of the vehicle`s center
    :param v_length: Length of the vehicle
    :param v_width: Width of the vehicle
    :param orientation: Orientation of the vehicle
    :return: center coordinates of middle circle,  center coordinates of front wheel circle,  center coordinates of rear wheel circle
    Hints: store the coordinates in a list, because casadi is not incompatible with numpy.
    """
    disc_radius, disc_distance = compute_approximating_circle_radius(v_length, v_width)

    # the distance between the first and last circle is computed as
    distance_centers = disc_distance / 2

    agent_center = [x_position, y_position]

    # compute the center position of middle circle
    center = [x_position + 0.175 * ca.cos(orientation), y_position + 0.175 * ca.sin(orientation)]

    # compute the center position of first circle (front)
    #center_fw = [x_position + (distance_centers / 2) * ca.cos(orientation), y_position + (distance_centers / 2) * ca.sin(orientation)]
    center_fw = [center[0] + (distance_centers / 2) * ca.cos(orientation),
                 center[1] + (distance_centers / 2) * ca.sin(orientation)]

    # compute the center position of second circle (rear)
    #center_rw = [x_position - (distance_centers / 2) * ca.cos(orientation), y_position - (distance_centers / 2) * ca.sin(orientation)]
    center_rw = [center[0] - (distance_centers / 2) * ca.cos(orientation),
                 center[1] - (distance_centers / 2) * ca.sin(orientation)]
    return center, center_fw, center_rw


def find_closest_distance_with_road_boundary(road_boundary, point):
    """
    Find closest distance with road boundary regarding point using casadi Framework
    :params point: a list [x, y]
    :params road_boundary: ndarray (n, 2)
    :return: closest distance in SX
    """
    road_boundary_list = road_boundary.tolist()
    num_points_boundary = len(road_boundary_list)
    distance = ca.SX.sym('distance', num_points_boundary)
    for i in range(num_points_boundary):
        distance[i] = ca.sqrt((point[0] - road_boundary_list[i][0]) ** 2 + (point[1] - road_boundary_list[i][1]) ** 2)
    return ca.mmin(distance)


def update_state(states, controls):
    x_k, y_k, theta_k = states[0], states[1], states[2]
    u_k, v_k = controls[0], controls[1]

    x_k1 = x_k + u_k * np.cos(theta_k + v_k)
    y_k1 = y_k + u_k * np.sin(theta_k + v_k)
    theta_k1 = theta_k + v_k

    return ca.vertcat(x_k1, y_k1, theta_k1)


def divide_boundary_into_points(vertices, num_points):
    """
    Divide the given boundary into a specified number of points.

    :param vertices: np.array, Array of vertices of the boundary.
    :param num_points: int, The total number of points to be generated.
    :return: np.array, Array containing all generated points.
    """
    # Initialize an array to store all points.
    all_points = np.zeros((num_points, 2))

    # Calculate the number of points to be allocated on each edge.
    total_edges = len(vertices) - 1
    points_per_edge = np.full(total_edges, (num_points - len(vertices)) // total_edges)
    for i in range((num_points - len(vertices)) % total_edges):
        points_per_edge[i] += 1

    # Allocate points to each edge.
    current_index = 0
    for i in range(total_edges):
        start, end = vertices[i], vertices[i + 1]
        # Linearly interpolate along each edge to generate points.
        edge_points = np.linspace(start, end, points_per_edge[i] + 2, endpoint=False)[1:]
        all_points[current_index:current_index + len(edge_points)] = edge_points
        current_index += len(edge_points)

    # The last point needs special handling to ensure that the total number of points reaches num_points.
    if current_index < num_points:
        all_points[current_index] = vertices[-1]

    return all_points

vertices_left = np.array([[-2, 2], [6, 2], [6, 6], [-2, 6], [-2,18], [10, 18]]) #Definition of left lane
vertices_right = np.array([[-2, -2], [10, -2], [10, 10], [2, 10], [2, 14], [10, 14]]) #Definition of right lane

class Configuration:
    def __init__(self):
        self.left_road_boundary = divide_boundary_into_points(vertices_left, 30)
        self.right_road_boundary = divide_boundary_into_points(vertices_right, 30)
        self.l = 1.35
        self.w = 1

