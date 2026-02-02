from gr00t_wbc.control.teleop.solver.hand.g1_gripper_ik_solver import (
    G1GripperInverseKinematicsSolver,
)
from gr00t_wbc.control.teleop.solver.hand.g1_inspire_hand_ik_solver import (
    G1InspireHandIKSolver,
)


# initialize hand ik solvers for g1 robot (Dex3 gripper)
def instantiate_g1_hand_ik_solver():
    """Instantiate hand IK solvers for G1 robot with Dex3 gripper."""
    left_hand_ik_solver = G1GripperInverseKinematicsSolver(side="left")
    right_hand_ik_solver = G1GripperInverseKinematicsSolver(side="right")
    return left_hand_ik_solver, right_hand_ik_solver


def instantiate_g1_inspire_hand_ik_solver(mode: str = "fingertip_distance"):
    """
    Instantiate hand IK solvers for G1 robot with Inspire hands.

    Args:
        mode: IK computation mode for Inspire hands.
              - "fingertip_distance": Uses thumb-to-fingertip distances for grip detection
              - "finger_curl": Maps finger curl angles from hand tracking

    Returns:
        Tuple of (left_hand_ik_solver, right_hand_ik_solver)
    """
    left_hand_ik_solver = G1InspireHandIKSolver(side="left", mode=mode)
    right_hand_ik_solver = G1InspireHandIKSolver(side="right", mode=mode)
    return left_hand_ik_solver, right_hand_ik_solver
