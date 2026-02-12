from argparse import ArgumentParser
import json
from franky import Affine, Robot, CartesianMotion, Gripper


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", default="172.16.0.2", help="FCI IP of the robot")
    parser.add_argument("--plan_path", type=str, required=True)
    parser.add_argument("--use_waypoints", action="store_true")
    parser.add_argument("--safety_height", type=float, default=0.217)
    # parser.add_argument("--safety_height", type=float, default=0.264362)
    parser.add_argument("--grasp", type=bool, default=False,
                        help="Whether to execute the grasp action")
    parser.add_argument("--pick_place", type=bool, default=False,
                        help="Whether to execute pick and place action after grasp")
    parser.add_argument("--rope", type=bool, default=False,
                        help="Whether the object is a rope")
    args = parser.parse_args()

    # Load plan
    with open(args.plan_path) as f:
        plan = json.load(f)
    
    kp = plan["keypoints_3d"]
    wp = plan["waypoints_3d"]

    # Initialize robot
    robot = Robot(args.host)
    gripper = Gripper(args.host)
    gripper.move(0.001, speed=0.02)
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

    # if kp["grasp"] is None:
    #     safe_move(kp["target"])
    #     raise ValueError("No grasp keypoint provided in the plan.")

    # Move to grasp
    if args.rope:
        # safe_move([0.46, -0.19, 0.14])
        # safe_move([0.52, -0.14, 0.15])
        # safe_move([0.51, -0.2, 0.14])
        safe_move([0.48, -0.14, 0.15])
    else:
        # safe_move(kp["grasp"])
        pass
    if args.grasp:
        input("Press Enter to execute grasp...")
        gripper.grasp(0.004, speed=0.02, force=20.0, 
                      epsilon_inner=0.01, epsilon_outer=0.02)
        input("Press Enter to execute grasp...")
        gripper.move(0.1, speed=0.02)
    
    curr_pos = robot.current_pose.end_effector_pose.translation
    print(f"Current position after grasp: {curr_pos}")
    # Move up after grasp
    # above_pos = [float(curr_pos[0]), float(curr_pos[1]), float(curr_pos[2] + 0.2)]
    # safe_move(above_pos)
    # exit()
    
    if args.rope:
        safe_move(wp["post_contact"][0])
        gripper.move(0.1, speed=0.02)
        input("Press Enter to finish...")
        end_pos = robot.current_pose.end_effector_pose.translation
        end_pos = [end_pos[0], end_pos[1], 0.4]
        safe_move(end_pos)

    # Calculate delta
    if "function" not in kp or kp["function"] is None:
        delta = [0.0, 0.0, 0.0]
    else:
        delta = [kp["target"][i] - kp["function"][i] for i in range(3)]

    if args.use_waypoints:
        # Pre-contact waypoint
        pre = [wp["pre_contact"][0][i] + delta[i] for i in range(3)]
        safe_move(pre)
        
        # Post-contact waypoint
        post = [wp["post_contact"][0][i] + delta[i] for i in range(3)]
        safe_move(post)
    else:
        # # Direct move with delta
        # target = [kp["function"][i] + delta[i] for i in range(3)]
        # safe_move(target)

        target = [kp["grasp"][i] + delta[i] for i in range(3)]

        curr_pos = robot.current_pose.end_effector_pose.translation
        if args.pick_place:
            # Move above target
            above_target = [ float(curr_pos[0]), float(curr_pos[1]), 
                             float(curr_pos[2] + 0.2) ]
            safe_move(above_target)

            # Move above target
            above_target = [ target[0] + 0.0, target[1] + 0.0, 
                             float(curr_pos[2] + 0.2) ]
            safe_move(above_target)

            # Move down to target
            target = [target[0] + 0.0, target[1] + 0.0, 
                      target[2] + 0.05]
            safe_move(target)

            input("Press Enter to release object...")
            gripper.move(0.1, speed=0.02)

            # Move back to above target
            safe_move(above_target)