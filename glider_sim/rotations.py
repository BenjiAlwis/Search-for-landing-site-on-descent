"""Rotation helpers. Quaternions are [w, x, y, z], rotating body -> NED."""
import numpy as np


def quat_from_euler(roll, pitch, yaw):
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def euler_from_quat(q):
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def rot_from_quat(q):
    """Rotation matrix R such that v_ned = R @ v_body."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rot_from_euler(roll, pitch, yaw):
    return rot_from_quat(quat_from_euler(roll, pitch, yaw))


def quat_derivative(q, omega_body):
    p, qq, r = omega_body
    w, x, y, z = q
    return 0.5 * np.array([
        -x * p - y * qq - z * r,
        w * p + y * r - z * qq,
        w * qq + z * p - x * r,
        w * r + x * qq - y * p,
    ])


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
