from argparse import ArgumentParser
import json
from franky import Affine, Robot, CartesianMotion, Gripper


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", default="172.16.0.2", help="FCI IP of the robot")
    parser.add_argument("--plan_path", type=str, required=True)
    parser.add_argument("--use_waypoints", action="store_true")
    parser.add_argument("--safety_height", type=float, default=0.156)
    parser.add_argument("--grasp", type=bool, default=True,
                        help="Whether to execute the grasp action")
    args = parser.parse_args()

    # Load plan
    with open(args.plan_path) as f:
        plan = json.load(f)
    
    kp = plan["keypoints_3d"]
    wp = plan["waypoints_3d"]

    # Initialize robot
    robot = Robot(args.host)
    gripper = Gripper(args.host)
    robot.set_cartesian_impedance([10.0] * 6)
    robot.set_collision_behavior([100.0] * 7, [100.0] * 7, [100.0] * 7, [100.0] * 7,
                                  [100.0] * 6, [100.0] * 6, [100.0] * 6, [100.0] * 6)
    robot.relative_dynamics_factor = 0.03

    # Get current orientation
    quat = robot.current_pose.end_effector_pose.quaternion

    def safe_move(pos):
        """Move with z-height safety enforcement"""
        safe_pos = [pos[0], pos[1], max(pos[2], args.safety_height)]
        print(f"Moving to safe position: {safe_pos}")
        input("Press Enter to continue...")
        robot.move(CartesianMotion(Affine(safe_pos, quat)))

    # Move to grasp
    safe_move(kp["grasp"])
    if args.grasp:
        input("Press Enter to execute grasp...")
        gripper.grasp(0.003, speed=0.02, force=30.0, 
                      epsilon_inner=0.04, epsilon_outer=0.04)

    # Calculate delta
    delta = [kp["target"][i] - kp["function"][i] for i in range(3)]

    if args.use_waypoints:
        # Pre-contact waypoint
        pre = [wp["pre_contact"][0][i] + delta[i] for i in range(3)]
        safe_move(pre)
        
        # Post-contact waypoint
        post = [wp["post_contact"][0][i] + delta[i] for i in range(3)]
        safe_move(post)
    else:
        # Direct move with delta
        target = [kp["function"][i] + delta[i] for i in range(3)]
        safe_move(target)