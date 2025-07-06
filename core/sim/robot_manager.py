import time
import numpy as np
import pybullet as p
import pybullet_data
import roboticstoolbox as rtb
import spatialmath as sm
from roboticstoolbox.tools.types import ArrayLike
from typing import Union

class Joint(object):
	"""Represents a robot joint with position control capabilities."""

	def __init__(self, robot_id, joint_id, limits):
		self.robot_id = robot_id
		self.joint_id = joint_id
		self.limits = limits

	def get_position(self):
		"""Get current joint position."""
		joint_state = p.getJointState(self.robot_id, self.joint_id)
		return joint_state[0]

	def set_position(self, position, max_force=100.):
		"""Set joint to target position."""
		p.setJointMotorControl2(
			self.robot_id,
			self.joint_id,
			controlMode=p.POSITION_CONTROL,
			targetPosition=position,
			force=max_force,
			positionGain=0.5,
			velocityGain=1.0
		)

class RobotController:
	"""Controls robot arm with mobile base capabilities."""

	def __init__(self, model_path, rtb_model, 
					scale=1, initial_base_pos=[0, 0, 0], 
					mode='stationary', use_fixed_base=True):
		self.robot_id = p.loadURDF(
			model_path, 
			initial_base_pos, 
			globalScaling=scale,
			useFixedBase=use_fixed_base
		)
		self.rtb_model = rtb_model
	
		self._time_step = p.getPhysicsEngineParameters()['fixedTimeStep']
		self._gripper_joint_id = 6
		self._left_finger_joint_id = 9
		self._right_finger_joint_id = 10
		self._end_effector_link_id = 11
		
		self.workspace_x_limits = [initial_base_pos[0], -initial_base_pos[0]]
		self.workspace_y_limits = [initial_base_pos[0], -initial_base_pos[0]]
		self.mode = mode
		
		# Internal state for holding objects
		self._grasped_object_id = None
		self._grasp_constraint_id = None
		
		self.load_model()

	def load_model(self):
		"""Load robot model and initialize joint objects."""
		joints = {}
		for i in range(p.getNumJoints(self.robot_id)):
			joint_info = p.getJointInfo(self.robot_id, i)
			joint_limits = {
				'lower': joint_info[8], 
				'upper': joint_info[9],
				'force': joint_info[10]
			}
			joints[i] = Joint(self.robot_id, i, joint_limits)
			# print(joint_info)
		
		self.rtb_model.qlim = np.array([[joints[i].limits['lower'], joints[i].limits['upper']] for i in range(7)]).T
		self.joints = joints
		self._left_finger = self.joints[self._left_finger_joint_id]
		self._right_finger = self.joints[self._right_finger_joint_id]

		self.reset_joints()

		self.wrist_neutral = self.joints[self._gripper_joint_id].get_position()   # ~ 0.785 rad

	def simulate_step(self, duration=0):
		"""Run simulation for specified duration."""
		for _ in range(int(duration / self._time_step)):
			p.stepSimulation()

	def reset_joints(self, initial_positions=[0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0, 0]):
		"""Reset all joints to initial positions."""
		for joint_id in range(len(initial_positions)):
			p.resetJointState(self.robot_id, joint_id, initial_positions[joint_id])

		self.init_ee_pos = self.get_ee_pos()

	def reset_ee(self):
		"""Reset end-effector to initial pose."""
		self.move_end_effector(self.init_ee_pos[0], self.init_ee_pos[1])

	def get_pos(self):
		"""Get current joint positions."""
		return [self.joints[i].get_position() for i in range(7)]

	def get_ee_pos(self):
		"""Get end-effector position and orientation."""
		ee_pos, ee_orn = p.getLinkState(self.robot_id, self._end_effector_link_id)[:2]
		return ee_pos, ee_orn

	def open_gripper(self, max_limit=True):
		"""Open gripper to specified limit."""
		if max_limit:
			self._left_finger.set_position(self._left_finger.limits['upper'])
			self._right_finger.set_position(self._right_finger.limits['upper'])
		else:
			self._left_finger.set_position(self._left_finger.limits['upper']/2)
			self._right_finger.set_position(self._right_finger.limits['upper']/2)
		
		self.simulate_step(0.2)

	def close_gripper(self):
		"""Close gripper completely."""
		self._left_finger.set_position(self._left_finger.limits['lower'])
		self._right_finger.set_position(self._right_finger.limits['lower'])

		self.simulate_step(0.2)

	def rotate_gripper_yaw(self, yaw_angle):
		"""Rotate gripper to target yaw angle smoothly."""
		# 1) get full EE rotation matrix
		q = self.get_ee_pos()[1]
		R = np.array(p.getMatrixFromQuaternion(q)).reshape(3,3)

		# 2) EE's local Z axis in world frame
		z_world = R[:,2]

		# 3) if that axis points mostly DOWN (z<0), flips sign of yaw→joint mapping
		sign = 1.0 if z_world[2] >= 0 else -1.0

		# 4) what's our current joint‐relative yaw?
		q_raw = self.joints[self._gripper_joint_id].get_position()
		rel0  = q_raw - self.wrist_neutral

		# 5) compute true global yaw now (from your helper or R→atan2), but
		#    we only need the delta in global yaw:
		_, _, glob_yaw = p.getEulerFromQuaternion(q)
		dglob = (yaw_angle - glob_yaw + np.pi) % (2*np.pi) - np.pi

		# 6) map that into joint‐space delta, accounting for the flip
		djoint = sign * dglob

		# 7) interpolate exactly as before
		N = max(int(abs(djoint) / 0.02), 1)
		for i in range(1, N+1):
			rel = rel0 + djoint * (i/N)                # new relative yaw
			q_cmd = rel + self.wrist_neutral          # back to joint angle
			q_cmd = np.clip(q_cmd,
							self.joints[self._gripper_joint_id].limits['lower'],
							self.joints[self._gripper_joint_id].limits['upper'])
			self.joints[self._gripper_joint_id].set_position(q_cmd)
			self.simulate_step(0.05)

	def inverse_kinematics(self, target_pose: sm.SE3, orientation: list, initial_guess: Union[ArrayLike, None]=None, max_iterations: int=100) -> np.ndarray:
		"""Calculate joint angles for target pose using inverse kinematics."""
		# if initial_guess is None:
		# 	initial_guess = np.zeros(7)
		
		# lower_limit_joints = [self.joints[i].limits['lower'] for i in range(7)]
		# upper_limit_joints = [self.joints[i].limits['upper'] for i in range(7)]

		# succes = False
		# for _ in range(max_iterations):
		# 	ik_solution = self.rtb_model.ikine_LM(target_pose, q0=initial_guess, joint_limits=True)
		# 	if ik_solution.success:
		# 		q = ik_solution.q
		# 		succes = True
		# 		break
		# 	else:
		# 		initial_guess = np.random.uniform(lower_limit_joints, upper_limit_joints)

		# if not succes:
		# 	raise ValueError('Could not find a valid IK solution.')
		
		# if np.any(q < lower_limit_joints) or np.any(q > upper_limit_joints):
		# 	print('IK solution out of joint limits.')
		# 	q = np.clip(q, lower_limit_joints, upper_limit_joints)

		q = p.calculateInverseKinematics(
			self.robot_id, 
			self._end_effector_link_id, 
			target_pose.t, 
			p.getQuaternionFromEuler(orientation),
			lowerLimits=[self.joints[i].limits['lower'] for i in range(7)],
			upperLimits=[self.joints[i].limits['upper'] for i in range(7)],
			maxNumIterations=max_iterations,
		)
		return q[:7]

	def generate_linear_path(self, start_pose: sm.SE3, end_pose: sm.SE3, num_waypoints: int, orientation: list) -> np.ndarray:
		"""Generate linear path between two poses."""
		trajectory = rtb.ctraj(start_pose, end_pose, num_waypoints)

		path = np.zeros((num_waypoints, 7))
		for i in range(num_waypoints):
			try:
				path[i] = self.inverse_kinematics(trajectory[i], orientation)
			except ValueError as e:
				print(f"IK failed at point {i+1}/{num_waypoints}: {e}")
				break  
	
		return path

	def find_closest_table_side(self, position):
		"""Find closest table side and return approach position."""
		x, y, z = position
		distances = {
			'0': x - self.workspace_x_limits[0],		# Top side
			'1': self.workspace_y_limits[1] - y,		# Right side
			'2': self.workspace_x_limits[1] - x,		# Bottom side
			'3': y - self.workspace_y_limits[0],		# Left side
		}
		
		thresh = 0.1

		# Find the side with the minimum distance
		closest_side = min(distances, key=distances.get)
		if closest_side == '0':
			target_position = [self.workspace_x_limits[0]-thresh, y, z]
		elif closest_side == '1':
			target_position = [x, self.workspace_y_limits[1]+thresh, z]
		elif closest_side == '2':
			target_position = [self.workspace_x_limits[1]+thresh, y, z]
		else:
			target_position = [x, self.workspace_y_limits[0]-thresh, z]
		
		return closest_side, target_position

	def move_to_position(self, target_position):
		"""Move end-effector to target position with appropriate orientation."""
		if self.mode == "mobile":
			pre_base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
			self.move_base_along_table(target_position, duration=10)
			base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)

			state = self.find_closest_table_side(base_pos)[0]
			if state == '0':
				target_orientation=[np.pi, 0, 0]
			elif state == '1':
				target_orientation=[np.pi, 0, -np.pi/2]
			elif state == '2':
				target_orientation=[np.pi, 0, np.pi]
			else:
				target_orientation=[np.pi, 0, np.pi/2]
		else:
			# grab current EE quaternion and extract yaw
			ee_quat = self.get_ee_pos()[1]
			_, _, current_yaw = p.getEulerFromQuaternion(ee_quat)
			# do a straight-line move keeping that same yaw
			target_orientation = [np.pi, 0, current_yaw]

		self.move_end_effector(target_position, target_orientation)

	def move_end_effector(self, target_position, target_orientation):
		"""Move end-effector through linear path to target pose."""
		# get current EE pose as SM.SE3
		position, orientation = self.get_ee_pos()
		orientation = p.getEulerFromQuaternion(orientation)
		current_pose = sm.SE3(position) * sm.SE3.RPY(orientation, order='xyz', unit='rad')

		# build the desired end‑pose
		target_pose = sm.SE3(target_position) * sm.SE3.RPY(target_orientation, order='xyz', unit='rad')

		# q = self.inverse_kinematics(target_pose)
		# self.reset(q)
		# self.set_joint_positions(q)

		# sweep through the straight‑line path
		for q in self.generate_linear_path(current_pose, target_pose, num_waypoints=20, orientation=target_orientation):
			# self.reset(q)
			self.set_joint_positions(q)
			self.simulate_step(0.25)

	def set_joint_positions(self, joint_angles):
		"""Set all joint positions simultaneously."""
		for i in range(7):
			self.joints[i].set_position(joint_angles[i])

	def move_base(self, new_base_pos, new_base_orn):
		"""Instantly move robot base to new position and orientation."""
		p.resetBasePositionAndOrientation(self.robot_id, new_base_pos, new_base_orn)

	def move_base_smoothly(self, target_base_pos, duration=1.0, steps=100):
		"""Smoothly move robot base to target position over duration."""
		# Get the current base position and orientation
		current_pos, current_orn = p.getBasePositionAndOrientation(self.robot_id)
		target_base_pos = [target_base_pos[0], target_base_pos[1], current_pos[2]]  # Keep the z-coordinate unchanged
		
		# Create a linear interpolation for the position
		pos_traj = np.linspace(current_pos, target_base_pos, steps)
		
		# Move along the trajectory
		for i in range(steps):
			t = i / (steps - 1)
			# Update the base position and orientation
			p.resetBasePositionAndOrientation(self.robot_id, pos_traj[i], current_orn)
			self.simulate_step(duration / steps)

	def rotate_base_smooth(self, target_base_orn, duration=1.0, steps=100):
		"""Smoothly rotate robot base to target orientation over duration."""
		# Get the current base position and orientation
		current_pos, current_orn = p.getBasePositionAndOrientation(self.robot_id)
		target_base_orn = p.getQuaternionFromEuler(target_base_orn)
		
		# Create a linear interpolation for the orientation
		orn_traj = np.zeros((steps, 4))
		for i in range(steps):
			t = i / (steps - 1)
			orn_traj[i] = p.getQuaternionSlerp(current_orn, target_base_orn, t)
		
		# Move along the trajectory
		for i in range(steps):
			# Update the base position and orientation
			p.resetBasePositionAndOrientation(self.robot_id, current_pos, orn_traj[i])
			self.simulate_step(duration / steps)

	def move_base_to_corner(self, previous_side, target_side, z_height, duration=1.0):
		"""Move robot base to corner between two table sides."""
		side_orientations = {
			'0': [0, 0, 0],            # Top side -> Facing down
			'1': [0, 0, -np.pi/2],     # Right side -> Facing left
			'2': [0, 0, np.pi],        # Bottom side -> Facing up
			'3': [0, 0, np.pi/2],      # Left side -> Facing right
		}

		thresh = 0.1

		corners = {
			('0', '1'): [self.workspace_x_limits[0]-thresh, self.workspace_y_limits[1]+thresh],
			('1', '0'): [self.workspace_x_limits[0]-thresh, self.workspace_y_limits[1]+thresh],
			('0', '3'): [self.workspace_x_limits[0]-thresh, self.workspace_y_limits[0]-thresh],
			('3', '0'): [self.workspace_x_limits[0]-thresh, self.workspace_y_limits[0]-thresh],
			('1', '2'): [self.workspace_x_limits[1]+thresh, self.workspace_y_limits[1]+thresh],
			('2', '1'): [self.workspace_x_limits[1]+thresh, self.workspace_y_limits[1]+thresh],
			('2', '3'): [self.workspace_x_limits[1]+thresh, self.workspace_y_limits[0]-thresh],
			('3', '2'): [self.workspace_x_limits[1]+thresh, self.workspace_y_limits[0]-thresh],
		}
		
		assert (previous_side, target_side) in corners

		corner_pos = corners[(previous_side, target_side)] + [z_height]
		self.move_base_smoothly(corner_pos, duration)
		self.rotate_base_smooth(side_orientations[target_side], duration)

	def move_base_along_table(self, target_position, duration=1.0):
		"""Move robot base along table perimeter to target position."""
		current_pos, _ = p.getBasePositionAndOrientation(self.robot_id)

		current_side, _ = self.find_closest_table_side(current_pos)
		target_side, mapped_target = self.find_closest_table_side(target_position)

		if current_side == target_side:
			self.move_base_smoothly(mapped_target, duration)
		elif abs(int(current_side) - int(target_side)) == 1 or abs(int(current_side) - int(target_side)) == 3:
			self.move_base_to_corner(current_side, target_side, mapped_target[2], duration)
			self.move_base_smoothly(mapped_target, duration)
		else:
			if current_side == '0':
				if self.workspace_y_limits[1] - target_position[1] + self.workspace_y_limits[1] - current_pos[1] < target_position[1] - self.workspace_y_limits[0] + current_pos[1] - self.workspace_y_limits[0]:
					# clockwise
					self.move_base_to_corner(current_side, '1', mapped_target[2], duration)
					self.move_base_to_corner('1', target_side, mapped_target[2], duration)
				else:
					# counter-clockwise
					self.move_base_to_corner(current_side, '3', mapped_target[2], duration)
					self.move_base_to_corner('3', target_side, mapped_target[2], duration)
			elif current_side == '1':
				if self.workspace_x_limits[1] - target_position[0] + self.workspace_x_limits[1] - current_pos[0] < target_position[0] - self.workspace_x_limits[0] + current_pos[0] - self.workspace_x_limits[0]:
					# clockwise
					self.move_base_to_corner(current_side, '2', mapped_target[2], duration)
					self.move_base_to_corner('2', target_side, mapped_target[2], duration)
				else:
					# counter-clockwise
					self.move_base_to_corner(current_side, '0', mapped_target[2], duration)
					self.move_base_to_corner('0', target_side, mapped_target[2], duration)
			elif current_side == '2':
				if self.workspace_y_limits[1] - target_position[1] + self.workspace_y_limits[1] - current_pos[1] > target_position[1] - self.workspace_y_limits[0] + current_pos[1] - self.workspace_y_limits[0]:
					# clockwise
					self.move_base_to_corner(current_side, '3', mapped_target[2], duration)
					self.move_base_to_corner('3', target_side, mapped_target[2], duration)
				else:
					# counter-clockwise
					self.move_base_to_corner(current_side, '1', mapped_target[2], duration)
					self.move_base_to_corner('1', target_side, mapped_target[2], duration)
			else:
				if self.workspace_x_limits[1] - target_position[0] + self.workspace_x_limits[1] - current_pos[0] > target_position[0] - self.workspace_x_limits[0] + current_pos[0] - self.workspace_x_limits[0]:
					# clockwise
					self.move_base_to_corner(current_side, '0', mapped_target[2], duration)
					self.move_base_to_corner('0', target_side, mapped_target[2], duration)
				else:
					# counter-clockwise
					self.move_base_to_corner(current_side, '2', mapped_target[2], duration)
					self.move_base_to_corner('2', target_side, mapped_target[2], duration)
			self.move_base_smoothly(mapped_target, duration)

	def pick_object(self, obj_id, target_yaw=None, approach_height=0.3, grasp_height=0.1):
		"""Pick up object by grasping and creating constraint."""
		# Check if already holding an object
		if self._grasped_object_id is not None:
			print(f"Robot is already holding object {self._grasped_object_id}")
			return False

		self.open_gripper()
		obj_pos, obj_orn = p.getBasePositionAndOrientation(obj_id)
		
		# Approach object from above
		self.move_to_position([obj_pos[0], obj_pos[1], obj_pos[2] + approach_height])

		# Rotate gripper to target yaw if specified
		if target_yaw is not None:
			self.rotate_gripper_yaw(target_yaw)

		# Move down to grasp height
		self.move_to_position([obj_pos[0], obj_pos[1], obj_pos[2] + grasp_height])

		# Create the fixed constraint with relative pose
		ee_pos, ee_orn = p.getLinkState(self.robot_id, self._end_effector_link_id)[:2]
		obj_pos, obj_orn = p.getBasePositionAndOrientation(obj_id)

		# Compute object pose in end-effector frame: T_obj_ee = inv(T_ee_w) * T_obj_w
		inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos, ee_orn)
		rel_pos, rel_orn  = p.multiplyTransforms(inv_ee_pos, inv_ee_orn, obj_pos, obj_orn)

		constraint_id = p.createConstraint(
			parentBodyUniqueId    = self.robot_id,
			parentLinkIndex       = self._end_effector_link_id,
			childBodyUniqueId     = obj_id,
			childLinkIndex        = -1,
			jointType             = p.JOINT_FIXED,
			jointAxis             = [0, 0, 0],
			parentFramePosition   = rel_pos,
			parentFrameOrientation= rel_orn,
			childFramePosition    = [0, 0, 0],
			childFrameOrientation = [0, 0, 0, 1]      # No extra rotation in object’s own frame
		)
		
		# Store internal state
		self._grasped_object_id = obj_id
		self._grasp_constraint_id = constraint_id
		
		self.close_gripper()
		self.move_to_position([obj_pos[0], obj_pos[1], obj_pos[2] + approach_height])
		
		return True

	def place_object(self, target_position, target_yaw=None, approach_height=0.3, place_height=0.1):
		"""Place held object at target position."""
		# Check if holding an object
		if self._grasped_object_id is None:
			print("Robot is not holding any object to place")
			return False
		
		# Approach target position from above
		self.move_to_position([target_position[0], target_position[1], target_position[2] + approach_height])

		# Rotate gripper to target yaw if specified
		if target_yaw is not None:
			self.rotate_gripper_yaw(target_yaw)
		
		# Move to place height with target yaw orientation for the EE
		self.move_to_position([target_position[0], target_position[1], target_position[2] + place_height])
		
		self.open_gripper()
		
		# Remove constraint and clear internal state
		if self._grasp_constraint_id is not None:
			p.removeConstraint(self._grasp_constraint_id)
			
		self._grasped_object_id = None
		self._grasp_constraint_id = None
		
		self.move_to_position([target_position[0], target_position[1], target_position[2] + approach_height])

		# Reset gripper to neutral position
		self.rotate_gripper_yaw(0)
		
		return True
