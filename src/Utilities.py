import math
from shapely.geometry.polygon import Polygon
import numpy as np


def wrap_angle(angle):
    two_pi = 2 * math.pi
    number_of_wraps = abs(angle) // two_pi
    if angle >= 0:
        wrap_angle = angle - (number_of_wraps * two_pi)
    else:
        wrap_angle = two_pi + angle + (number_of_wraps * two_pi)
    return wrap_angle


def distance_point_point(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def distance_point_segment(obstacle, point):
    v = obstacle[0]
    w = obstacle[1]
    # Return minimum distance between line segment vw and point p
    v, w, p = np.array(v), np.array(w), np.array(point)
    l2 = np.sum((w - v) ** 2)  # i.e. |w-v|^2 -  avoid a sqrt
    if l2 == 0.0:
        return np.linalg.norm(p - v)  # v == w case
    # Consider the line extending the segment, parameterized as v + t (w - v).
    # We find projection of point p onto the line.
    # It falls where t = [(p-v) . (w-v)] / |w-v|^2
    # We clamp t from [0,1] to handle points outside the segment vw.
    t = max(0, min(1, np.dot(p - v, w - v) / l2))
    projection = v + t * (w - v)  # Projection falls on the segment
    return np.linalg.norm(p - projection)


def polygon_to_segments(polygon):
    # given a polygon, returns a list of all the segments that form its boundary
    lines = []
    points = list(polygon.exterior.coords)
    points.remove(points[len(points) - 1])
    for i in range(len(points)):
        p1 = points[i]
        if i != len(points) - 1:
            p2 = points[i + 1]
        else:
            p2 = points[0]
        line = (p1, p2)
        lines.append(line)
    return lines


def polygons_from_vertexes_list(vertexes_list):
    polygons = []
    for vertex in vertexes_list:
        polygons.append(Polygon(vertex))
    return polygons


def polygons_to_segments(polygons):
    segments = []
    if polygons:
        for i in range(len(polygons)):
            for segment in polygon_to_segments(polygons[i]):
                segments.append(segment)
        return segments
    else:
        return segments
