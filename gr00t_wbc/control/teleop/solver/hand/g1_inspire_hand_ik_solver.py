"""
Inspire Hand IK Solver for G1 Robot

This solver converts Pico hand tracking data to Inspire hand joint commands.
It supports two modes:
- finger_curl: Map individual finger curl angles to Inspire joints
- fingertip_distance: Use fingertip-to-thumb distances for grip detection
"""

import numpy as np

from gr00t_wbc.control.teleop.solver.solver import Solver


# OpenXR Hand Joint Indices (27 joints total)
class OpenXRHandJoint:
    """OpenXR hand tracking joint indices."""

    PALM = 0
    WRIST = 1
    # Thumb (4 joints)
    THUMB_METACARPAL = 2
    THUMB_PROXIMAL = 3
    THUMB_DISTAL = 4
    THUMB_TIP = 5
    # Index (5 joints)
    INDEX_METACARPAL = 6
    INDEX_PROXIMAL = 7
    INDEX_INTERMEDIATE = 8
    INDEX_DISTAL = 9
    INDEX_TIP = 10
    # Middle (5 joints)
    MIDDLE_METACARPAL = 11
    MIDDLE_PROXIMAL = 12
    MIDDLE_INTERMEDIATE = 13
    MIDDLE_DISTAL = 14
    MIDDLE_TIP = 15
    # Ring (5 joints)
    RING_METACARPAL = 16
    RING_PROXIMAL = 17
    RING_INTERMEDIATE = 18
    RING_DISTAL = 19
    RING_TIP = 20
    # Little/Pinky (5 joints)
    LITTLE_METACARPAL = 21
    LITTLE_PROXIMAL = 22
    LITTLE_INTERMEDIATE = 23
    LITTLE_DISTAL = 24
    LITTLE_TIP = 25


# Inspire Hand Joint Indices (6 DOFs, padded to 7 for compatibility)
class InspireHandJoint:
    """Inspire hand joint indices."""

    THUMB = 0  # Thumb rotation/flexion
    INDEX = 1  # Index finger curl
    MIDDLE = 2  # Middle finger curl
    RING = 3  # Ring finger curl
    LITTLE = 4  # Little/Pinky finger curl
    THUMB_FLEXION = 5  # Additional thumb DOF


class G1InspireHandIKSolver(Solver):
    """
    IK Solver for Inspire hands using Pico hand tracking data.

    Supports two modes:
    - finger_curl: Maps finger curl angles from hand tracking to joint commands
    - fingertip_distance: Uses fingertip-to-thumb distances for discrete grip poses

    Args:
        side: Which hand ("left" or "right")
        mode: IK computation mode ("finger_curl" or "fingertip_distance")
    """

    # Inspire hand joint limits (approximate, in radians)
    JOINT_MIN = 0.0
    JOINT_MAX = 1.0  # Normalized range, actual limits depend on hardware

    def __init__(self, side: str, mode: str = "fingertip_distance") -> None:
        super().__init__()
        self.side = "L" if side.lower() == "left" else "R"
        self.mode = mode

        # Distance threshold for fingertip-to-thumb grip detection
        self.dist_threshold = 0.05

        # Joint angle scaling factors for finger curl mode
        self.curl_scale = 1.0

    def register_robot(self, robot):
        """Register robot model (not used for Inspire hands)."""
        pass

    def __call__(self, finger_data) -> np.ndarray:
        """
        Compute Inspire hand joint commands from finger tracking data.

        Args:
            finger_data: Dictionary with "position" key containing (25, 4, 4) array
                        of fingertip transformation matrices

        Returns:
            np.ndarray: Joint commands for Inspire hand (7 elements, 6 active + 1 padding)
        """
        if self.mode == "finger_curl":
            return self._finger_curl_ik(finger_data)
        else:
            return self._fingertip_distance_ik(finger_data)

    def _finger_curl_ik(self, finger_data) -> np.ndarray:
        """
        Compute joint commands by mapping finger curl angles.

        Uses the relative positions of finger joints to estimate curl amount.
        """
        q_desired = np.zeros(7)  # 6 active DOFs + 1 padding

        fingertips = finger_data["position"]

        # Extract positions from transformation matrices
        positions = np.array([finger[:3, 3] for finger in fingertips])
        positions = np.reshape(positions, (-1, 3))

        # Get key finger positions for curl calculation
        # Using the 25-joint format where fingertips are at indices 4, 9, 14, 19, 24
        thumb_tip = positions[4, :]
        index_tip = positions[9, :]
        middle_tip = positions[14, :]
        ring_tip = positions[19, :]
        pinky_tip = positions[24, :]

        # Get metacarpal/base positions for curl reference (indices 0, 5, 10, 15, 20)
        thumb_base = positions[0, :]
        index_base = positions[5, :]
        middle_base = positions[10, :]
        ring_base = positions[15, :]
        pinky_base = positions[20, :]

        # Compute curl as normalized distance from base to tip
        # When fully extended, distance is maximum; when curled, distance is minimum
        def compute_curl(tip, base, max_length=0.1):
            """Compute curl amount (0=extended, 1=fully curled)."""
            dist = np.linalg.norm(tip - base)
            # Normalize and invert (closer = more curled)
            curl = 1.0 - np.clip(dist / max_length, 0.0, 1.0)
            return curl * self.curl_scale

        # Compute curl for each finger
        thumb_curl = compute_curl(thumb_tip, thumb_base, max_length=0.06)
        index_curl = compute_curl(index_tip, index_base, max_length=0.10)
        middle_curl = compute_curl(middle_tip, middle_base, max_length=0.11)
        ring_curl = compute_curl(ring_tip, ring_base, max_length=0.10)
        pinky_curl = compute_curl(pinky_tip, pinky_base, max_length=0.08)

        # Map to Inspire hand joints
        q_desired[InspireHandJoint.THUMB] = thumb_curl
        q_desired[InspireHandJoint.INDEX] = index_curl
        q_desired[InspireHandJoint.MIDDLE] = middle_curl
        q_desired[InspireHandJoint.RING] = ring_curl
        q_desired[InspireHandJoint.LITTLE] = pinky_curl
        q_desired[InspireHandJoint.THUMB_FLEXION] = thumb_curl * 0.5  # Secondary thumb DOF

        return q_desired

    def _fingertip_distance_ik(self, finger_data) -> np.ndarray:
        """
        Compute joint commands based on fingertip-to-thumb distances.

        Similar to the G1Gripper approach but outputs continuous values
        based on proximity rather than discrete poses.
        """
        q_desired = np.zeros(7)  # 6 active DOFs + 1 padding

        fingertips = finger_data["position"]

        # Extract positions from transformation matrices
        positions = np.array([finger[:3, 3] for finger in fingertips])
        positions = np.reshape(positions, (-1, 3))

        # Get fingertip positions (indices 4, 9, 14, 19, 24 in 25-joint format)
        thumb_pos = positions[4, :]
        index_pos = positions[9, :]
        middle_pos = positions[14, :]
        ring_pos = positions[19, :]
        pinky_pos = positions[24, :]

        # Compute distances from thumb to each finger
        index_dist = np.linalg.norm(thumb_pos - index_pos)
        middle_dist = np.linalg.norm(thumb_pos - middle_pos)
        ring_dist = np.linalg.norm(thumb_pos - ring_pos)
        pinky_dist = np.linalg.norm(thumb_pos - pinky_pos)

        # Convert distances to curl values (closer = more curled)
        # Use a soft threshold for smooth transitions
        def distance_to_curl(dist, threshold=0.05, max_dist=0.15):
            """Convert distance to curl value with smooth transition."""
            if dist < threshold:
                return 1.0  # Fully closed
            elif dist > max_dist:
                return 0.0  # Fully open
            else:
                # Linear interpolation between threshold and max_dist
                return 1.0 - (dist - threshold) / (max_dist - threshold)

        # Compute curl values for each finger
        index_curl = distance_to_curl(index_dist)
        middle_curl = distance_to_curl(middle_dist)
        ring_curl = distance_to_curl(ring_dist)
        pinky_curl = distance_to_curl(pinky_dist)

        # Thumb curl based on average proximity to other fingers
        avg_dist = (index_dist + middle_dist + ring_dist + pinky_dist) / 4.0
        thumb_curl = distance_to_curl(avg_dist, threshold=0.06, max_dist=0.12)

        # Map to Inspire hand joints
        q_desired[InspireHandJoint.THUMB] = thumb_curl
        q_desired[InspireHandJoint.INDEX] = index_curl
        q_desired[InspireHandJoint.MIDDLE] = middle_curl
        q_desired[InspireHandJoint.RING] = ring_curl
        q_desired[InspireHandJoint.LITTLE] = pinky_curl
        q_desired[InspireHandJoint.THUMB_FLEXION] = thumb_curl * 0.5

        return q_desired

    def _get_open_hand_pose(self) -> np.ndarray:
        """Return joint values for fully open hand."""
        return np.zeros(7)

    def _get_closed_hand_pose(self) -> np.ndarray:
        """Return joint values for fully closed hand (fist)."""
        q_desired = np.zeros(7)
        q_desired[InspireHandJoint.THUMB] = 1.0
        q_desired[InspireHandJoint.INDEX] = 1.0
        q_desired[InspireHandJoint.MIDDLE] = 1.0
        q_desired[InspireHandJoint.RING] = 1.0
        q_desired[InspireHandJoint.LITTLE] = 1.0
        q_desired[InspireHandJoint.THUMB_FLEXION] = 0.5
        return q_desired

    def _get_pinch_pose(self) -> np.ndarray:
        """Return joint values for pinch grip (thumb + index)."""
        q_desired = np.zeros(7)
        q_desired[InspireHandJoint.THUMB] = 0.8
        q_desired[InspireHandJoint.INDEX] = 0.8
        q_desired[InspireHandJoint.THUMB_FLEXION] = 0.4
        return q_desired
