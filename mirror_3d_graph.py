import sys
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import RegularGridInterpolator, interp1d
import functools
import os
import matplotlib.path as mpath
from matplotlib import cm

# Basic class for a point of light. In this code I frequently change between LightPoint and Arrays given context.
class LightPoint:
    def __init__(self, x: float, y: float, z: float, intensity=0):
        self.x = x
        self.y = y
        self.z = z
        self.intensity = intensity

    def __add__(self, other):
        return LightPoint(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return LightPoint(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return LightPoint(self.x * scalar, self.y * scalar, self.z * scalar)
        raise TypeError(f"Cannot multiply LightPoint by {type(scalar)}")

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def magnitude(self):
        return np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def __repr__(self):
        return f"LightPoint(x={self.x}, y={self.y}, z={self.z}, intensity={self.intensity})"


# Reflects Point across plane.
def reflect_across_plane(point, A, B, C, D):
    w = A * point.x + B * point.y + C * point.z
    q = D - w
    t = q / (A ** 2 + B ** 2 + C ** 2)
    return point + LightPoint(A, B, C) * (t * 2)


# Detects if point in Above, On, or Below plane.
def above_or_below_plane(point, A, B, C, D):
    result = A * point.x + B * point.y + C * point.z + D
    if result > 0:
        return "Above"
    elif result == 0:
        return "On"
    else:
        return "Below"


# Rotate mirror points around the scene.
def rotate_points(mirror_p1, mirror_p2, mirror_p3, mirror_p4, angle_x, angle_y, angle_z):
    angle_x = math.radians(angle_x)
    angle_y = math.radians(angle_y)
    angle_z = math.radians(angle_z)

    rotation_x = [
        [1, 0, 0],
        [0, math.cos(angle_x), -math.sin(angle_x)],
        [0, math.sin(angle_x), math.cos(angle_x)]
    ]

    rotation_y = [
        [math.cos(angle_y), 0, math.sin(angle_y)],
        [0, 1, 0],
        [-math.sin(angle_y), 0, math.cos(angle_y)]
    ]

    rotation_z = [
        [math.cos(angle_z), -math.sin(angle_z), 0],
        [math.sin(angle_z), math.cos(angle_z), 0],
        [0, 0, 1]
    ]

    def rotate_point(point, matrix):
        x = point.x * matrix[0][0] + point.y * matrix[0][1] + point.z * matrix[0][2]
        y = point.x * matrix[1][0] + point.y * matrix[1][1] + point.z * matrix[1][2]
        z = point.x * matrix[2][0] + point.y * matrix[2][1] + point.z * matrix[2][2]
        return LightPoint(x, y, z)

    def apply_all(point):
        return rotate_point(rotate_point(rotate_point(point, rotation_z), rotation_y), rotation_x)

    return apply_all(mirror_p1), apply_all(mirror_p2), apply_all(mirror_p3), apply_all(mirror_p4)


# Gets plane equation from mirror points.
def get_plane_equation(p1, p2, p3):
    v1 = np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])
    v2 = np.array([p3.x - p1.x, p3.y - p1.y, p3.z - p1.z])
    normal_vector = np.cross(v1, v2)
    A, B, C = normal_vector
    D = A * p1.x + B * p1.y + C * p1.z
    return A, B, C, D


def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm else v


# Is a 3d point between four other points (the mirror).
def is_point_in_bounds(mp1, mp2, mp3, mp4, point):
    v1 = np.array([mp2.x - mp1.x, mp2.y - mp1.y, mp2.z - mp1.z])
    v2 = np.array([mp3.x - mp1.x, mp3.y - mp1.y, mp3.z - mp1.z])
    normal = normalize(np.cross(v1, v2))
    u_axis = normalize(v1)
    v_axis = normalize(np.cross(normal, u_axis))

    def project(p):
        vec = np.array([p.x - mp1.x, p.y - mp1.y, p.z - mp1.z])
        return np.dot(vec, u_axis), np.dot(vec, v_axis)

    polygon_uv = [project(p) for p in [mp1, mp2, mp4, mp3]]
    point_uv = project(point)

    return mpath.Path(polygon_uv).contains_point(point_uv)


# Gets intersection between light ray and mirror plane.
def get_intersection_lightray(ray_origin, ray_direction, A, B, C, D):
    denominator = A * ray_direction.x + B * ray_direction.y + C * ray_direction.z
    if abs(denominator) < 1e-6:
        return None
    numerator = D - (A * ray_origin.x + B * ray_origin.y + C * ray_origin.z)
    t = numerator / denominator
    return ray_origin + ray_direction * t


# Light intensity calculation function
def light_calc(receiver, source, source_strength, mp1, mp2, mp3, mp4, A, B, C, D):

    # Calculates the angle between light ray and normal vector of light source plane.
    # angle_deg is defined if one needs to print it (debugging purposes)
    v1 = receiver - source
    v1 = np.array([v1.x, v1.y, v1.z])
    v2 = np.array([0, 0, -1])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # in radians
    angle_deg = np.degrees(angle_rad)
    theta = angle_rad
    receiver.intensity = 0

    # ========================= Direct path =========================
    same_side = (
        above_or_below_plane(receiver, A, B, C, D)
        == above_or_below_plane(source, A, B, C, D)
    )
    receiver_plane_pos = above_or_below_plane(receiver, A, B, C, D)

    if same_side and receiver_plane_pos != "On":
        print("Direct Path on Same Side")
        dist = (receiver - source).magnitude() / 100
        if dist == 0:
            dist = 0.0001
        intensity = (source_strength / (4 * math.pi * dist ** 2)) * math.cos(theta) ** 2
        print(intensity)
        receiver.intensity += intensity

    else:
        direct_intersection = get_intersection_lightray(
            source, receiver - source, A, B, C, D
        )

        if direct_intersection is None or not is_point_in_bounds(
            mp1, mp2, mp3, mp4, direct_intersection
        ):
            print("Direct Path NOT on Same Side")
            dist = (receiver - source).magnitude() / 100
            if dist == 0:
                dist = 0.0001
            intensity = (source_strength / (4 * math.pi * dist ** 2)) * math.cos(theta) ** 2
            print(intensity)
            receiver.intensity += intensity

    # ======================== Indirect path ========================
    if same_side and receiver_plane_pos != "On":
        reflected_point = reflect_across_plane(receiver, A, B, C, D)
        indirect_intersection = get_intersection_lightray(
            source, reflected_point - source, A, B, C, D
        )

        if indirect_intersection is not None and is_point_in_bounds(
            mp1, mp2, mp3, mp4, indirect_intersection
        ):
            dist = (reflected_point - source).magnitude() / 100
            v1 = reflected_point - source
            v1 = np.array([v1.x, v1.y, v1.z])
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)
            theta = angle_rad
            if dist == 0:
                dist = 0.0001
            receiver.intensity += (source_strength / (4 * math.pi * dist ** 2)) * math.cos(theta) ** 2

    return receiver.intensity


# Scene setup function with light_source, strength, mirror_points, mirror_angles & translation
# Source strength = 10 by default for graphing purposes.
# If you want to change it, you'll need to change the range of values displayed.
# Haven't extensively tested changing the translation values. I worked under an assumption that it is fixed.
def setup_scene(source_pos=(50, 50, 100), source_str=10, mirror_angles=(0, 0, 0), translation=(50, 100, 50)):
    source = LightPoint(*source_pos)
    source_strength = source_str

    # Create mirror points
    mirror_p1 = LightPoint(10, 0, 10)
    mirror_p2 = LightPoint(-10, 0, 10)
    mirror_p3 = LightPoint(10, 0, -10)
    mirror_p4 = LightPoint(-10, 0, -10)

    # Apply rotation to mirror
    mirror_x_angle, mirror_y_angle, mirror_z_angle = mirror_angles
    mirror_p1, mirror_p2, mirror_p3, mirror_p4 = rotate_points(
        mirror_p1, mirror_p2, mirror_p3, mirror_p4, 
        mirror_x_angle, mirror_y_angle, mirror_z_angle
    )

    # Apply translation to mirror (it starts centered at 0,0,0)
    tx, ty, tz = translation
    translationvector = LightPoint(tx, ty, tz)
    mp1 = mirror_p1 + translationvector
    mp2 = mirror_p2 + translationvector
    mp3 = mirror_p3 + translationvector
    mp4 = mirror_p4 + translationvector

    # Calculate plane equation
    A, B, C, D = get_plane_equation(mp1, mp2, mp3)

    return source, source_strength, mp1, mp2, mp3, mp4, A, B, C, D


# Visualization functions
def plot_3d_scene(source, mp1, mp2, mp3, mp4, intensity_map=None, x_vals=None, y_vals=None):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot mirror as a polygon
    mirror_vertices = [[(p.x, p.y, p.z) for p in [mp1, mp2, mp4, mp3]]]
    mirror_poly = Poly3DCollection(mirror_vertices, color='silver', alpha=0.5)
    ax.add_collection3d(mirror_poly)

    # Plot source
    ax.scatter(source.x, source.y, source.z, color='yellow', label='Source', s=100)

    # Plot mirror corners
    mirror_points = [mp1, mp2, mp3, mp4]
    for i, mp in enumerate(mirror_points, 1):
        ax.scatter(mp.x, mp.y, mp.z, color='blue')
        ax.text(mp.x, mp.y, mp.z, f'MP{i}', color='black')

    # If intensity map is provided, plot it as a surface
    if intensity_map is not None and x_vals is not None and y_vals is not None:
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.zeros_like(X)  # Surface is at z = 0

        # Normalize colors, Not normalizing values.
        norm = mcolors.Normalize(vmin=0.55, vmax=1.35)
        colors = cm.inferno(norm(intensity_map))

        # Plot intensity as colored surface at z = 0
        ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1, alpha=0.9, shade=False, zorder=1)

        # Add colorbar
        mappable = cm.ScalarMappable(norm=norm, cmap='inferno')
        mappable.set_array([])
        fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label='Intensity')

    # Set axis limits and labels
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 120)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Scene with Light Source and Mirror')
    ax.legend()

    return fig, ax


# Makes the intensity map at z = 0 (i.e. the floor).
def generate_intensity_map(grid_size, source, source_strength, mp1, mp2, mp3, mp4, A, B, C, D):
    x_vals = np.linspace(0, 100, grid_size)
    y_vals = np.linspace(0, 100, grid_size)
    intensity_map = np.zeros((grid_size, grid_size))

    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            point = LightPoint(x, y, 0)
            intensity = light_calc(point, source, source_strength, mp1, mp2, mp3, mp4, A, B, C, D)
            intensity_map[j, i] = intensity  # j, i because rows = y, cols = x

    return intensity_map, x_vals, y_vals

# Plots the intensity map at z = 0 (i.e. the floor).
def plot_intensity_map(intensity_map, x_vals, y_vals):
    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.imshow(
        intensity_map,
        extent=(0, 100, 0, 100),
        origin='lower',
        cmap='inferno',
        vmin=0.55,
        vmax=1.35
    )
    fig.colorbar(c, ax=ax, label='Intensity')
    ax.set_title('Light Intensity on Bottom Surface (z=0)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    return fig, ax

# Generate Line Path
def generate_line_path(start, end, step_size=0.2):
    start = np.array(start)
    end = np.array(end)
    direction = end - start
    length = np.linalg.norm(direction)
    unit_direction = direction / length

    num_steps = int(length / step_size) + 1
    distances = np.linspace(0, length, num_steps)

    path = [start + unit_direction * d for d in distances]
    return np.array(path)

# Generate Arc Path
def generate_arc_path(start, end, arc_height=50, step_size=0.2):
    start = np.array(start)
    end = np.array(end)

    chord = end - start
    chord_length = np.linalg.norm(chord)
    midpoint = (start + end) / 2

    normal = np.cross(chord, [0, 0, 1])
    normal = normal / np.linalg.norm(normal)
    arc_midpoint = midpoint + normal * arc_height

    v1 = start - arc_midpoint
    v2 = end - arc_midpoint

    radius = np.linalg.norm(v1)
    angle_total = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
    arc_length = radius * angle_total

    num_steps = int(arc_length / step_size) + 1
    angles = np.linspace(0, angle_total, num_steps)

    axis = np.cross(v1, v2)
    axis = axis / np.linalg.norm(axis)

    def rotate_vector(vec, axis, angle):
        return (vec * np.cos(angle) +
                np.cross(axis, vec) * np.sin(angle) +
                axis * np.dot(axis, vec) * (1 - np.cos(angle)))

    path = [arc_midpoint + rotate_vector(v1, axis, theta) for theta in angles]
    return np.array(path)


def combine_paths(*paths, decimal_precision=5):
    combined = np.vstack(paths)  # Stack all paths together
    # Round to specified decimal places to avoid float precision issues
    rounded = np.round(combined, decimals=decimal_precision)
    # Use np.unique to remove duplicates (based on rows)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    # Reorder in original order
    unique_ordered = combined[np.sort(unique_indices)]
    return unique_ordered


def resample_path_and_compute_intensities(path, step_size, light_calc, source, source_strength, mp1, mp2, mp3, mp4, A, B, C, D):
    # Compute segment distances
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    total_length = np.sum(segment_lengths)

    # Compute cumulative arc length
    cumulative_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)

    # Target distances along the path
    target_distances = np.arange(0, total_length + step_size, step_size)

    # Resample the path at these distances
    resampled_points = []
    intensities = []

    for d in target_distances:
        # Find the segment that contains this distance
        idx = np.searchsorted(cumulative_lengths, d) - 1
        idx = min(max(idx, 0), len(path) - 2)

        # Local interpolation
        t = (d - cumulative_lengths[idx]) / segment_lengths[idx]
        point = (1 - t) * path[idx] + t * path[idx + 1]

        resampled_points.append(point)

        receiver = LightPoint(*point)
        intensity = light_calc(receiver, source, source_strength, mp1, mp2, mp3, mp4, A, B, C, D)
        intensities.append(intensity)

    return np.array(resampled_points), np.array(intensities)


# Path reconstruction functions
def silent_light_calc_wrapper(light_calc_func):
    @functools.wraps(light_calc_func)
    def wrapper(*args, **kwargs):
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            result = light_calc_func(*args, **kwargs)
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout
        return result
    return wrapper


# This is the main purpose of this file. It reconstructs the path given limited information.
def reconstruct_path(intensities, sampling_rate, start_point, light_calc_func, source, source_strength, 
                        mp1, mp2, mp3, mp4, A, B, C, D, beam_width=5, top_k=2):
    silent_calc = silent_light_calc_wrapper(light_calc_func)

    class BeamEntry:
        def __init__(self, path, current_point, prev_direction, total_error):
            self.path = path
            self.current_point = current_point
            self.prev_direction = prev_direction
            self.total_error = total_error

    # Initialize beam with the start point
    start_array = np.array([start_point.x, start_point.y, start_point.z])
    beam = [BeamEntry([start_array], start_point, None, 0.0)]

    num_angles = 120
    search_radii = [sampling_rate, sampling_rate * 0.9, sampling_rate * 1.1]

    for i in range(1, len(intensities)):
        target_intensity = intensities[i]
        new_beam = []

        for entry in beam:
            current_point = entry.current_point
            prev_direction = entry.prev_direction

            def intensity_diff(coords):
                x, y = coords
                x = np.clip(x, 0, 100)
                y = np.clip(y, 0, 100)
                receiver = LightPoint(x, y, 0)
                calculated_intensity = silent_calc(receiver, source, source_strength, mp1, mp2, mp3, mp4, A, B, C, D)

                dist = np.linalg.norm([x - current_point.x, y - current_point.y])
                distance_penalty = (dist - sampling_rate) ** 2

                direction_penalty = 0
                if prev_direction is not None:
                    new_direction = np.array([x - current_point.x, y - current_point.y])
                    if np.linalg.norm(new_direction) > 0:
                        new_direction = new_direction / np.linalg.norm(new_direction)
                        direction_change = 1 - np.dot(prev_direction, new_direction)
                        direction_penalty = direction_change

                intensity_error = abs(calculated_intensity - target_intensity)
                return 3.0 * intensity_error + distance_penalty + direction_penalty, intensity_error

            candidates = []
            for radius in search_radii:
                for angle in np.linspace(0, 2 * np.pi, num_angles, endpoint=False):
                    x = current_point.x + radius * np.cos(angle)
                    y = current_point.y + radius * np.sin(angle)

                    if 0 <= x <= 100 and 0 <= y <= 100:
                        score, intensity_err = intensity_diff((x, y))
                        candidates.append((score, intensity_err, x, y))

            # Sort and take top beam_width candidates
            candidates.sort(key=lambda c: c[0])
            for score, intensity_err, x, y in candidates[:beam_width]:
                new_point = LightPoint(x, y, 0)
                direction = np.array([x - current_point.x, y - current_point.y])
                normed_direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else prev_direction

                new_path = entry.path + [np.array([x, y, 0])]
                new_error = entry.total_error + intensity_err

                new_beam.append(BeamEntry(new_path, new_point, normed_direction, new_error))

        # Sort new beam and keep top beam_width entries
        new_beam.sort(key=lambda e: e.total_error)
        beam = new_beam[:beam_width]

        if i % 10 == 0 or i == len(intensities) - 1:
            print(f"Processed {i}/{len(intensities) - 1} steps | Beam size: {len(beam)}")

    # Return the top_k best paths sorted by total error
    beam.sort(key=lambda e: e.total_error)
    best_entries = beam[:top_k]
    return [np.array(entry.path) for entry in best_entries]


# Analysis functions
def resample_path(path, num_points):
    t = np.zeros(len(path))
    for i in range(1, len(path)):
        t[i] = t[i-1] + np.linalg.norm(path[i] - path[i-1])
    if t[-1] > 0:
        t = t / t[-1]

    x_interp = interp1d(t, [p[0] for p in path])
    y_interp = interp1d(t, [p[1] for p in path])
    z_interp = interp1d(t, [p[2] for p in path])

    t_new = np.linspace(0, 1, num_points)
    resampled = np.zeros((num_points, 3))
    resampled[:, 0] = x_interp(t_new)
    resampled[:, 1] = y_interp(t_new)
    resampled[:, 2] = z_interp(t_new)
    return resampled


# Calculates how good/bad the reconstructed path is compared to the original path.
def analyze_path(original_path, reconstructed_path, source, source_strength, mp1, mp2, mp3, mp4, A, B, C, D):
    # Calculate spatial errors
    comparison_length = max(len(original_path), len(reconstructed_path))
    if len(original_path) != len(reconstructed_path):
        original_resampled = resample_path(original_path, comparison_length)
        reconstructed_resampled = resample_path(reconstructed_path, comparison_length)
        distances = np.sqrt(np.sum((original_resampled - reconstructed_resampled)**2, axis=1))
    else:
        distances = np.sqrt(np.sum((original_path - reconstructed_path)**2, axis=1))

    mean_error = np.mean(distances)
    max_error = np.max(distances)

    # Calculate intensities along reconstructed path
    reconstructed_intensities = []
    for point in reconstructed_path:
        receiver = LightPoint(point[0], point[1], point[2])
        intensity = light_calc(receiver, source, source_strength, mp1, mp2, mp3, mp4, A, B, C, D)
        reconstructed_intensities.append(intensity)

    return {
        'mean_error': mean_error,
        'max_error': max_error,
        'reconstructed_intensities': reconstructed_intensities
    }


# Plots the path comparison between original path and reconstructed path.
def plot_path_comparison(original_path, reconstructed_path, idx=0, colors=['r', 'm']):
    plt.figure(figsize=(8, 6))

    # Plot original path
    plt.plot([p[0] for p in original_path], [p[1] for p in original_path], 'b-', linewidth=2, label='Original Path')
    plt.scatter([original_path[0][0]], [original_path[0][1]], color='green', s=100, label='Start Point')
    plt.scatter([original_path[-1][0]], [original_path[-1][1]], color='blue', s=100, label='End Point')

    # Plot reconstructed path
    plt.plot([p[0] for p in reconstructed_path], [p[1] for p in reconstructed_path], linestyle='--',
             color=colors[idx], linewidth=2, label=f'Reconstructed Path {idx+1}')
    plt.scatter([reconstructed_path[-1][0]], [reconstructed_path[-1][1]], color=colors[idx], s=100,
                label=f'Reconstructed End Point {idx+1}')

    # Add arrows for direction
    arrow_step = max(1, len(reconstructed_path) // 20)
    for i in range(0, len(reconstructed_path) - 1, arrow_step):
        plt.arrow(reconstructed_path[i][0], reconstructed_path[i][1],
                  reconstructed_path[i + 1][0] - reconstructed_path[i][0],
                  reconstructed_path[i + 1][1] - reconstructed_path[i][1],
                  head_width=1.0, head_length=1.5, fc=colors[idx], ec=colors[idx], alpha=0.6)

    plt.title(f"Path Comparison - Reconstructed Path {idx+1}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    return plt.gcf()

# Plots the intensity comparison between original path and reconstructed path.
def plot_intensity_comparison(intensities, reconstructed_intensities, idx=0):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot intensity comparison
    ax1.plot(range(len(intensities)), intensities, 'b-', label='Original Intensities')
    ax1.plot(range(len(reconstructed_intensities)), reconstructed_intensities, 'r--', label=f'Reconstructed Intensities {idx+1}')
    ax1.set_title(f"Intensity Comparison - path {idx+1}")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Intensity")
    ax1.grid(True)
    ax1.legend()

    # Plot intensity error if possible
    if len(intensities) == len(reconstructed_intensities):
        intensity_errors = np.abs(np.array(intensities) - np.array(reconstructed_intensities))
        ax2.plot(range(len(intensity_errors)), intensity_errors, 'g-', label='Intensity Error')
        ax2.set_title(f"Intensity Error - path {idx+1}")
        ax2.set_xlabel("Steps")
        ax2.set_ylabel("Absolute Error")
        ax2.grid(True)
        mean_intensity_error = np.mean(intensity_errors)
        max_intensity_error = np.max(intensity_errors)
        ax2.text(0.02, 0.95, f"Mean Error: {mean_intensity_error:.6f}\nMax Error: {max_intensity_error:.6f}",
                 transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, "Cannot calculate errors - different number of points",
                 ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()
    return fig


# Plots reconstructed paths on heat/intensity map.
def plot_paths_on_heatmap(intensity_map, x_vals, y_vals, original_path, reconstructed_paths, colors=['r', 'm']):
    plt.figure(figsize=(10, 8))
    plt.imshow(intensity_map, extent=(0, 100, 0, 100), origin='lower', cmap='inferno')
    plt.colorbar(label='Intensity')

    # Plot original path
    plt.plot([p[0] for p in original_path], [p[1] for p in original_path], 'w-', linewidth=2, label='Original Path')
    plt.scatter([original_path[0][0]], [original_path[0][1]], color='green', s=100, label='Start Point')
    plt.scatter([original_path[-1][0]], [original_path[-1][1]], color='blue', s=100, label='End Point')

    # Plot reconstructed paths
    for idx, reconstructed_path in enumerate(reconstructed_paths):
        plt.plot([p[0] for p in reconstructed_path], [p[1] for p in reconstructed_path],
                 linestyle='--', color=colors[idx], linewidth=2, label=f'Reconstructed Path {idx+1}')
        plt.scatter([reconstructed_path[-1][0]], [reconstructed_path[-1][1]], color=colors[idx], s=100, 
                    label=f'Reconstructed End Point {idx+1}')

    plt.title("Paths Overlaid on Intensity Map")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(False)
    plt.legend(loc='upper left')
    plt.tight_layout()

    return plt.gcf()


# Main function
def main():
    # Setup the scene
    source, source_strength, mp1, mp2, mp3, mp4, A, B, C, D = setup_scene()

    # Generate intensity map
    intensity_map, x_vals, y_vals = generate_intensity_map(101, source, source_strength, mp1, mp2, mp3, mp4, A, B, C, D)

    # Plot 3D scene
    plot_3d_scene(source, mp1, mp2, mp3, mp4, intensity_map, x_vals, y_vals)

    # Plot intensity map
    plot_intensity_map(intensity_map, x_vals, y_vals)

    # Define path points
    start = np.array([0, 0, 0])
    end = np.array([100, 100, 0])

    # Generate path segments
    path1 = generate_line_path(start, [10, 30, 0])
    path2 = generate_arc_path([10, 30, 0], [50, 70, 0], arc_height=10)
    path3 = generate_line_path([50, 70, 0], end)

    # Combine path segments
    original_path = combine_paths(path1, path2, path3)

    # Resample path and compute intensities
    original_path, intensities = resample_path_and_compute_intensities(
        original_path, 0.2,
        light_calc=light_calc,
        source=source,
        source_strength=source_strength,
        mp1=mp1, mp2=mp2, mp3=mp3, mp4=mp4,
        A=A, B=B, C=C, D=D
    )

    # Plot intensity vs steps
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(intensities)), intensities, color='green', marker='o')
    plt.title("Intensity vs Steps")
    plt.xlabel("Steps")
    plt.ylabel("Intensities")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Reconstruct path from intensities
    print("\n" + "="*80)
    print("FINAL SOLUTION - RECONSTRUCTING THE PATH FROM INTENSITIES")
    print("="*80)
    print("Using only:")
    print("1. The intensities array")
    print("2. The sampling rate (0.2)")
    print("3. The start point (0,0,0)")
    print("4. The light_calc function")
    print("="*80 + "\n")

    start_point = LightPoint(start[0], start[1], start[2], intensities[0])
    # k represents top k paths. I have set this to 1 as the top paths are near identical.
    reconstructed_paths = reconstruct_path(
        intensities, 0.2, start_point, light_calc, 
        source, source_strength, mp1, mp2, mp3, mp4, A, B, C, D,
        beam_width=5, top_k=1
    )

    # Define colors for visualization
    colors = ['g', 'r']

    # Analyze and visualize each reconstructed path
    for idx, reconstructed_path in enumerate(reconstructed_paths):
        # Plot path comparison
        plot_path_comparison(original_path, reconstructed_path, idx, colors)
        plt.show()

        # Analyze path
        print(f"\nAnalysis for Reconstructed path {idx + 1}:")
        analysis = analyze_path(original_path, reconstructed_path, source, source_strength, mp1, mp2, mp3, mp4, A, B, C, D)
        print(f"Mean Error: {analysis['mean_error']:.4f} units")
        print(f"Max Error: {analysis['max_error']:.4f} units")

        # Plot intensity comparison
        plot_intensity_comparison(intensities, analysis['reconstructed_intensities'], idx)
        plt.show()

        # Print start and end points
        print(f"Start point: ({reconstructed_path[0][0]:.2f}, {reconstructed_path[0][1]:.2f}, {reconstructed_path[0][2]:.2f})")
        print(f"End point: ({reconstructed_path[-1][0]:.2f}, {reconstructed_path[-1][1]:.2f}, {reconstructed_path[-1][2]:.2f})")

    # Plot all paths on heatmap
    plot_paths_on_heatmap(intensity_map, x_vals, y_vals, original_path, reconstructed_paths, colors)
    plt.show()


if __name__ == "__main__":
    main()
