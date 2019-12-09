# -*- coding: utf-8 -*-
"""
Created on  : Tue Jul 11 11:35:53 2017
Author      : Guus Rongen
Project     : PR3594.10 HB Havens
Description : Geoemtry classes and functions

"""
import geopandas as gpd
import numpy as np
from scipy.interpolate import interp1d
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.ops import polygonize as _polygonizeprep


def is_left(pt, linecoords):
    """
    Determine if coordinate if left from line

    Parameters
    ----------
    pt : tuple
        coordinate for which the side is checked
    linecoords : list of tuples
        coordinates of the line
    """

    input_0d = False
    if isinstance(pt, tuple) or np.ndim(pt) == 1:
        pt = np.array(pt)[np.newaxis, :]
        input_0d = True

    A, B = linecoords
    # Determine if the point is located left or right of the line
    sign = ((B[0] - A[0]) * (pt[:, 1] - A[1])) - ((B[1] - A[1]) * (pt[:, 0] - A[0]))

    # Make outputarray as False
    left = np.zeros(len(pt), dtype=bool)
    # Set left points to true
    left[np.where(sign > 0)] = True

    if input_0d:
        left = left[0]

    return left

def average_angle(angles, factors=None, degrees=False):
    """
    Calculate average angles, optionally with a weight factor

    Parameters
    ----------
    angles : NxM-array
        angles to be averaged. First axis different angles, second axis different angles
    factors : NxM-array
        factors to add to the averaging. Optional. If given, dimensions should be equal to angles

    Returns
    -------
    average_angles : size-M array
    """
    if factors is not None and np.isnan(factors.ravel()).any():
        raise ValueError('NaN values in factors')
    if np.isnan(angles.ravel()).any():
        raise ValueError('NaN values in angles')

    # Optionally convert degrees to radians
    if degrees:
        angles = np.deg2rad(angles)

    # Check shape of input arguments
    if factors is not None:
        if not angles.shape == factors.shape:
            raise ValueError('input arguments (angles, factors) should have equal shapes')

    # convert to x and y component
    sinhoek = np.sin(np.atleast_1d(angles))
    coshoek = np.cos(np.atleast_1d(angles))

    if factors is not None:
        sinhoek *= np.atleast_1d(factors)
        coshoek *= np.atleast_1d(factors)

    # calculate angle
    angle = np.arctan2(sinhoek.sum(axis=-1), coshoek.sum(axis=-1)) % (2 * np.pi)

    # Convert angles back
    if degrees:
        angle = np.rad2deg(angle)

    return angle.squeeze()

def extend_point_to_linestring(pt, direction, extend, as_LineString=False):
    """
    Function to extend point to linestring in a certain direction (nautical)

    Parameters
    ----------
    pt : tuple or shapely.geometry.Point
        starting point of line
    direction : real
        direction (nautical) in which the line is extended
    extend : real or tuple with reals
        extend of the requested line
    return_coords : boolean, default False
        to return the coordinates instead of a LineString

    """
    if isinstance(pt, Point):
        pt = pt.coords[0]

    # Convert to tuple if necessary
    if isinstance(extend, float) or isinstance(extend, int):
        extend = (0, extend)

    # Convert angle to radians
    angle_radians = np.deg2rad(nau2car(direction))

    # Calculate fraction x and y
    dx = np.atleast_1d(np.cos(angle_radians))
    dy = np.atleast_1d(np.sin(angle_radians))

    lines = np.zeros((len(dx), 2, 2))
    for i in range(len(dx)):
        # Extend to line
        lines[i, :, :] = [
            [extend[0] * dx[i] + pt[0], extend[0] * dy[i] + pt[1]],
            [extend[1] * dx[i] + pt[0], extend[1] * dy[i] + pt[1]]
        ]

    # Convert to right output format
    if as_LineString:
        lines = [LineString(line) for line in lines]

    if isinstance(direction, (np.int, np.float, float, int)):
        lines = lines[0]

    return lines

def get_orientation(line, dist, eps=1e-6):
    """
    Get carthesian orientation of line at distance
    """
    if isinstance(dist, Point):
        dist = line.project(dist)
        
    pt1 = line.interpolate(dist - eps)
    pt2 = line.interpolate(dist + eps)
    return np.arctan2(pt2.y - pt1.y, pt2.x - pt1.x)

def calculate_angle(a, b, c):
    """
    Function to calculate the angle between three points

    Parameters
    ----------
    a : tuple or numpy.array
        coordinate 1
    b : tuple or numpy.array
        coordinate 2 (at which the angle is calculated)
    c : tuple or numpy.array
        coordinate 3

    Return
    ------
    Angle between coordinates in radians

    """

    if isinstance(a, tuple):
        a = np.array(a)
    if isinstance(b, tuple):
        b = np.array(b)
    if isinstance(c, tuple):
        c = np.array(c)

    ba = np.hypot(*(a-b).T)
    ac = np.hypot(*(c-a).T)
    bc = np.hypot(*(b-c).T)

    frac = (ba**2 + bc**2 - ac ** 2) / (2 * ba * bc)
    if isinstance(frac, (float, np.float)):
        frac = np.array([frac])

    ind = np.isclose(frac, 1.0)
    
    angle = np.zeros(frac.size)
    angle[ind] = 0.0
    angle[~ind] = np.arccos(frac[~ind])
    
    return angle.squeeze()


def calculate_XY(origin, location, wavedirection, breakwater=None, shading=None):
    """
    Function to calculate horizontal and vertical distance to location in
    coordinate system parallel to wave direction.

    X and Y are calculation with goniometric function, see the code.
    Depending on the situation type (type I or type II), which is determined
    from the input argument breakwater (present means type I), the X needs
    to be inverted if the breakwater is shading the location. The shading is
    checked by calculation the wave-breakwater-angle and the wave-location-angle
    If the second is smaller, the breakwater is shading the location.

    Parameters
    ----------
    origin : tuple or Nx2-array
        (array with) coordinate(s) of the origin
    location : tuple or Nx2-array
        (array with) coordinate(s) of the output locations
    wavedirection : array with floats
        wave direction (nautical)
    breakwater : pandas row or tuple
        breakwater

    """
    if isinstance(origin, tuple):
        origin = np.array(origin)[np.newaxis, :]
    if isinstance(location, tuple):
        location = np.array(location)[np.newaxis, :]

    # To calculate X and Y, the angle between the wave direction and
    # hrd location is needed:
    # calculate the absolute distance between origin (harbor entrance) and location
    dist = np.hypot(*(origin - location).T)
    # calculate a point in the direction of the wave, by extending the origin in the direction of the wave
    wavedirectionpoint = np.zeros((max(len(origin), len(wavedirection)), 2))
    wavedirectionpoint[:, 0] = origin[:, 0] + np.cos(np.radians(nau2car(wavedirection)))
    wavedirectionpoint[:, 1] = origin[:, 1] + np.sin(np.radians(nau2car(wavedirection)))
    # Calculate angle at the origin
    angle = calculate_angle(location, origin, wavedirectionpoint)
    # From this and the distance the X and Y are calculated
    X = np.sin(angle) * dist
    Y = np.cos(angle) * dist
    
    if shading is not None:
        X[shading] = X[shading] * -1.
    
    # Next, we need to find if the breakwater shades the location at all
    # to do so, we calculate the angle between the wave direction and the
    # breakwater, and the angle between the location and the breakwater. If the
    # locationangle is smaller than the breakwater angle, the breakwater is
    # shading the location, and X is negative.

    # After this also the angle between the breakwater and the location is
    # needed, to find if a location is shaded by the breakwater
    elif breakwater is not None:
        # Add an extra point 1 meter on the breakwater from the head
        head = breakwater.breakwaterhead
        # The next line is needed since we do not know the direction of the breakwater geometry
        dist = 1 if breakwater.geometry.project(head) < 0.01 else breakwater.geometry.length-1.0
        # Calculate angle between breakwater and wave direction
        breakwaterpoint = breakwater.geometry.interpolate(dist)
        wavedirangle = calculate_angle(breakwaterpoint.coords[0], head.coords[0], wavedirectionpoint)
        # Calculate angle between breakwater and location
        locationangle = calculate_angle(breakwaterpoint.coords[0], head.coords[0], location)
        # Make X negative if the location is shaded by the breakwater (wavedirangle > locationangle)
        X[np.where(wavedirangle > locationangle)] = X[np.where(wavedirangle > locationangle)] * -1.

    return X, Y

def polygonize_intersecting_lines(lines, round_decimals=None):

    # First alle lines must be splitted on intersections
    splitted_lines = []
    # Loop trough lines
    for line in lines:
        if not isinstance(line, LineString):
            raise TypeError('Expected LineString, got {}'.format(type(line)))

        # Select all other lines
        others = lines[:]
        others.remove(line)

        # Add all intersections to a list
        isects = []
        for other in others:
            if line.crosses(other):
                isect = line.intersection(other)
                if isinstance(isect, Point):
                    isect = [isect]
                else:
                    print(isect)
                isects += isect

        # If there are no intersections, add the line as a whole
        if not isects:
            splitted_lines.append(np.vstack(line.coords[:]))
            continue

        # Calculate distance of intersections along line
        dist = sorted([line.project(pt) for pt in isects])

        # Split line by distance
        splitted_lines += split_line_by_distance(line, dist)

    gpd.GeoDataFrame(
        np.arange(len(splitted_lines)),
        columns=['id'],
        geometry=[LineString(line) for line in splitted_lines],
        crs='epsg:28992'
    ).to_file('lines.shp')

    if round_decimals != None:
        splitted_lines = [line.round(round_decimals) for line in splitted_lines]


    polygons = [p for p in _polygonize(splitted_lines)]

    return polygons

def split_line_by_distance(line, splitdist):
    """
    Function to split a line in different line segments by distance.

    Parameters
    ----------
    line : shapely.geometry.LineString
        LineString of the line to be splitted
    splitdist : list
        List of distances where the line should be split
    """

    # Calculate cumulative distance of coordinates on line
    dists = np.r_[0.0, np.cumsum(np.hypot(*np.diff(np.vstack(line.coords), axis=0).T))]

    # Add startpoint and endpoint if not present
    if not splitdist[0] == 0.0:
        splitdist = np.r_[0.0, splitdist]
    if not splitdist[-1] == line.length:
        splitdist = np.r_[splitdist, line.length]

    lines = []
    # For each line
    for i in range(len(splitdist)-1):
        # Determine indices of points to add
        indices = (dists >= splitdist[i]) & (dists <= splitdist[i+1])
        coords = [line.coords[i] for i in range(len(line.coords)) if indices[i]]

        # Add first coordinate if not already present
        if splitdist[i] not in dists[indices]:
            coords = [line.interpolate(splitdist[i]).coords[0]] + coords
        # Add last coordinate if not already present
        if splitdist[i+1] not in dists[indices]:
            coords = coords + [line.interpolate(splitdist[i+1]).coords[0]]

        # Add to list with lines
        lines.append(LineString(coords))

    return lines

def find_nearest_intersection(L1, L2, P):
    """
    Find the nearest intersection of two lines, to a point.

    L1 : shapely.geometry.LineString
        line 1
    L2 : shapely.geometry.LineString
        line 2
    P : shapely.geometry.Point
        point

    """

    # Get single intersection
    intersections = L1.intersection(L2)
    if isinstance(intersections, Point):
        return intersections

    # Check if lines intersect
    if not L1.intersects(L2):
        return None

    # Convert tuple to point if necessary
    if isinstance(P, tuple):
        P = Point(P)

    # If multiple intersections
    argmin = np.argmin([isect.distance(P) for isect in intersections])
    return intersections[argmin]



def snap_flooddefence_lines(lines, max_snap_dist=np.inf):
    """
    Function to snap the end points of non touching lines to each other,
    in order to get one or more closing lines

    Parameters
    ----------
    lines : numpy.ndarray
        Nlines x Npoints x 2
    maxdist : float
        maximum distance over which snapping will be performed.

    """
    # Make copy of lines to modify
    newlines = np.copy(lines)

    # Collect start and end points of lines
    headpoints = np.vstack([line[[0, -1], :] for line in lines])

    def other_index(i):
        """Function to find other index of line end in headpoints"""
        return i + (-1 if i % 2 == 1 else 1)

    def head_index(i):
        """Function to find index of line head based on headpoint index"""
        return (i % 2) * -1

    # Determine the point that is furthest away from any other line    
    maxdist = 0
    startindex = 0
    for i, point in enumerate(headpoints):
        dists = np.hypot(*(headpoints - point).T)
        # Set distance to itself and other point of line to zero
        dists[i] = 0
        dists[other_index(i)]

        # If the maximum distance is larger, update startindex
        if dists.max() > maxdist:
            startindex = dists.argmax()

    # Adjust all points
    headpoints[startindex, :] = np.nan
    # Aangepast Svasek 05/10/18 - Printen van de startindex is overbodig
#    print(startindex)

    for _ in range(np.size(lines, 0) - 1):
        # Determine other point of line
        endindex = other_index(startindex)

        # Find next startpoint
        dists = np.hypot(*(headpoints - headpoints[endindex, :]).T)
        dists[other_index(startindex)] = np.nan
        startindex = np.nanargmin(dists)

        # Calculate point in middle and adjust
        if np.nanmin(dists) < max_snap_dist:
            midpoint = headpoints[[endindex, startindex], :].mean(axis=0)
            newlines[endindex // 2][head_index(endindex)] = midpoint
            newlines[startindex // 2][head_index(startindex)] = midpoint

        # Set endpoint and startpoint to nan
        headpoints[[endindex, startindex], :] = np.nan

    return newlines

def multiple_polygons(geometry):
    """
    Function to convert MultiPolygon to list, or Polygon to list with
    the Polygon.

    Parameters
    ----------
    geometry : Polygon or MultiPolygon
    """

    if isinstance(geometry, Polygon):
        return [geometry]

    elif isinstance(geometry, MultiPolygon):
        return [poly for poly in geometry]
    
    else:
        raise TypeError('Wrong input argument type: {}\n Give Polygon or MultiPolygon.'.format(type(geometry)))

def nau2car(nautical):
    """
    Function to convert natuical angle to carthesian angle

    Parameters
    ----------
    nautical : float
        angle from north clockwise, in degrees

    Returns
    -------
    carthesian : float
        angle in carthesian coordinate system degrees
    """
    # Concert from nautical to degrees
    cartesian = (270 - nautical) % 360

    return cartesian

def car2nau(cartesian):
    """
    Function to convert cathesian angle to nautical angle

    Parameters
    ----------
    carthesian : float
        angle in carthesian coordinate system degrees

    Returns
    -------
    nautical : float
        angle from north clockwise, in degrees
    """
    # Concert from cathesian to degrees
    nautical = nau2car(cartesian)

    return nautical

def interp_angles(t, tp, angles, extrapolate=False):

    """
    Interpolate angles

    Parameters
    ----------
    t : ndarray
        Time steps at which the angles are requestes
    tp : ndarray
        Time steps at which the angles are available
    angles : ndarray
        Angles
    """
    
    # convert to x and y component
    yp = np.sin(np.radians(angles))
    xp = np.cos(np.radians(angles))

    # Interpolate x and y
    if extrapolate:
        x = interp1d(tp, xp, fill_value='extrapolate')(t)
        y = interp1d(tp, yp, fill_value='extrapolate')(t)
    else:
        x = np.interp(t, tp, xp)
        y = np.interp(t, tp, yp)
    
    # calculate angle and make sure they are positive
    angles = (np.degrees(np.arctan2(y, x)) + 360) % 360
    
    return angles

def perp_dist_to_line(pt, line):
    """
    https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points
    """
    x0, y0 = pt
    x1, y1 = line[0]
    x2, y2 = line[1]
    return abs((x2-x1)*(y1-y0) - (x1-x0)*(y2-y1)) / ((x2-x1)**2 + (y2-y1)**2)**0.5

def intersection(line1, line2):
    """
    Calculate intersection two line segments.
    
    Parameters
    ----------
    line1 : list
        sequence of two coordinates of the first line
    line2 : list
        sequence of two coordinates of the second line
    """

    p1, p2 = line1[0], line1[1]
    L1_A = (p1[1] - p2[1])
    L1_B = (p2[0] - p1[0])
    L1_C = -(p1[0] * p2[1] - p2[0] * p1[1])
    
    xmin1 = min(p1[0], p2[0])
    xmax1 = max(p1[0], p2[0])
    ymin1 = min(p1[1], p2[1])
    ymax1 = max(p1[1], p2[1])

    p1, p2 = line2[0], line2[1]
    L2_A = (p1[1] - p2[1])
    L2_B = (p2[0] - p1[0])
    L2_C = -(p1[0] * p2[1] - p2[0] * p1[1])

    xmin2 = min(p1[0], p2[0])
    xmax2 = max(p1[0], p2[0])
    ymin2 = min(p1[1], p2[1])
    ymax2 = max(p1[1], p2[1])

    D  = L1_A * L2_B - L1_B * L2_A
    if D != 0:
        x = (L1_C * L2_B - L1_B * L2_C) / D
        y = (L1_A * L2_C - L1_C * L2_A) / D

        if (xmin1 <= x <= xmax1) & (xmin2 <= x <= xmax2) & (ymin1 <= y <= ymax1) & (ymin2 <= y <= ymax2):
            return x, y

def intersection_lines(line1, line2):
    """
    Calculate intersection two line segments.
   
    Parameters
    ----------
    line1 : list
        sequence of two coordinates of the first line
    line2 : list
        sequence of two coordinates of the second line
    """

    p1, p2 = line1[0], line1[1]
    L1_A = (p1[1] - p2[1])
    L1_B = (p2[0] - p1[0])
    L1_C = -(p1[0] * p2[1] - p2[0] * p1[1])
   
    p1, p2 = line2[0], line2[1]
    L2_A = (p1[1] - p2[1])
    L2_B = (p2[0] - p1[0])
    L2_C = -(p1[0] * p2[1] - p2[0] * p1[1])

    D  = L1_A * L2_B - L1_B * L2_A
    if D != 0:
        x = (L1_C * L2_B - L1_B * L2_C) / D
        y = (L1_A * L2_C - L1_C * L2_A) / D
        
        return x, y


def as_linestring_list(linestring):
    """Return a list of linestrings, retrieved from
    the given geometry object,
    
    Parameters
    ----------
    linestring : LineString or MultiLineString or list
        object with linestrings
    
    Returns
    -------
    list
        list with (only) LineStrings
    
    Raises
    ------
    TypeError
        if geometries in object are not LineString of MultiLineString
    """
    if isinstance(linestring, LineString):
        return [linestring]
    elif isinstance(linestring, MultiLineString):
        return [l for l in linestring]
    elif isinstance(linestring, list):
        lst = []
        for item in linestring:
            lst.extend(as_linestring_list(item))
        return lst
    else:
        raise TypeError(f'Expected LineString or MultiLineString. Got "{type(linestring)}"')

def as_polygon_list(polygon):
    """Return a list of polygons, retrieved from
    the given geometry object,
    
    Parameters
    ----------
    polygon : Polygon or MultiPolygon or list
        object with linestrings
    
    Returns
    -------
    list
        list with (only) Polygons
    
    Raises
    ------
    TypeError
        if geometries in object are not Polygon of MultiPolygon
    """
    if isinstance(polygon, Polygon):
        return [polygon]
    elif isinstance(polygon, MultiPolygon):
        return [p for p in polygon]
    elif isinstance(polygon, list):
        lst = []
        for item in polygon:
            lst.extend(as_polygon_list(item))
        return lst
    else:
        raise TypeError(f'Expected Polygon or MultiPolygon. Got "{type(polygon)}"')

def rotate_coordinates(origin, theta, xcrds, ycrds):
    """
    Rotate coordinates around origin (x0, y0) with a certain angle (radians)
    """
    x0, y0 = origin
    xcrds_rot = x0 + (xcrds - x0) * np.cos(theta) + (ycrds - y0) * np.sin(theta)
    ycrds_rot = y0 - (xcrds - x0) * np.sin(theta) + (ycrds - y0) * np.cos(theta)
    return xcrds_rot, ycrds_rot