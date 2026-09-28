from data import C1, C2, C3, C4, C5, C6, C7, C8, ALL_POINTS, INFINITY_POINTS
from interactive_plane import InteractivePlane

ip = InteractivePlane(
    curves=[C1, C2, C3, C4, C5, C6, C7, C8],
    intersection_points=ALL_POINTS,
    infinity_points=INFINITY_POINTS,
)
ip.show()