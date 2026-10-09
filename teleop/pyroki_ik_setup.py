import os
import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import pyroki as pk
import numpy as np

def setup_pyroki_ik(urdf_path, target_link_name="tool0"):
    import yourdfpy
    # Load robot from URDF
    urdf = yourdfpy.URDF.load(urdf_path)
    robot = pk.Robot.from_urdf(urdf)
    target_link_idx = robot.links.names.index(target_link_name)

    @jax.jit
    def solve_ik(target_pos, target_quat, start_q):
        # target_pos: (3,)
        # target_quat: (4,) wxyz
        # start_q: (6,)
        
        target_pose = jaxlie.SE3(jnp.concatenate([target_quat, target_pos]))
        
        joint_var = robot.joint_var_cls()
        factors = []
        
        @jaxls.Cost.factory(name="SE3PoseMatchJointCost")
        def match_joint_to_pose_cost(vals: jaxls.VarValues, joint_var: jaxls.Var[jnp.ndarray]):
            joint_cfg = vals[joint_var]
            Ts_joint_world = robot.forward_kinematics(joint_cfg)
            actual_pose = jaxlie.SE3(Ts_joint_world[..., target_link_idx, :])
            residual = (actual_pose.inverse() @ target_pose).log()
            return residual * 100.0

        @jaxls.Cost.factory(name="MinTravelCost")
        def min_travel_cost(vals: jaxls.VarValues, joint_var: jaxls.Var[jnp.ndarray]):
            return (vals[joint_var] - start_q) * 5.0

        factors.append(match_joint_to_pose_cost(joint_var))
        factors.append(min_travel_cost(joint_var))
        
        solution = jaxls.LeastSquaresProblem(factors, [joint_var]).analyze().solve(
            verbose=False,
            initial_vals=jaxls.VarValues.make((joint_var.with_value(start_q),)),
            termination=jaxls.TerminationConfig(max_iterations=15),
        )
        return solution[joint_var]

    @jax.jit
    def check_error(q_sol, target_pos):
        Ts = robot.forward_kinematics(q_sol)
        # Ts is array of shape (num_links, 7) where last 3 is translation
        actual_pos = Ts[target_link_idx, 4:7]
        return jnp.linalg.norm(actual_pos - target_pos)

    return robot, solve_ik, check_error, target_link_idx

def get_pyroki_fk(robot, target_link_idx):
    def fk(q):
        Ts = robot.forward_kinematics(jnp.array(q))
        pose = Ts[target_link_idx]
        quat = pose[:4]
        pos = pose[4:]
        return {
            "pos": np.array(pos).tolist(),
            "quat": np.array(quat).tolist() # w, x, y, z
        }
    return fk
