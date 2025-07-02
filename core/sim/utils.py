import os
import time
import random
import numpy as np
import pybullet as p
import pybullet_data
import trimesh
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import roboticstoolbox as rtb
import spatialmath as sm
from roboticstoolbox.tools.types import ArrayLike
from typing import Union

def decompose_obj(directory, obj_name):
    input_path = os.path.join(directory, f"{obj_name}.obj")
    output_path = os.path.join(directory, f"{obj_name}_vhacd.obj")
    log_path = os.path.join(directory, "vhacd_log.txt")

    p.vhacd(
        input_path,
        output_path,
        log_path,
        resolution=100000,
        depth=20,
        concavity=0.0025,
        planeDownsampling=4,
        convexhullDownsampling=4,
        alpha=0.04,
        beta=0.05,
        gamma=0.00125,
        minVolumePerCH=0.0001
    )

    print(f"[VHACD] Decomposition complete: {output_path}")

def create_urdf(directory, name, mass, concave=False):
    obj_path = os.path.join(directory, f"{name}.obj")
    scene = trimesh.load(obj_path, force='scene')
    parts = []

    if len(scene.geometry) == 1:
        parts.append(("link_1", f"{name}.obj"))
    else:
        for idx, (geom_name, geom) in enumerate(scene.geometry.items(), start=1):
            part_filename = f"{name}_{idx}.obj"
            part_path = os.path.join(directory, part_filename)

            if hasattr(geom.visual, 'material') and geom.visual.material is not None:
                geom.visual.material.name = geom_name
            else:
                geom.visual.material = trimesh.visual.material.SimpleMaterial(name=geom_name)

            obj_text = trimesh.exchange.obj.export_obj(geom, mtl_name='material.mtl')

            lines = obj_text.splitlines()
            for i, line in enumerate(lines):
                if line.startswith('mtllib'):
                    lines.insert(i + 1, f'o {name}_{idx}')
                    break
            obj_text_with_o = '\n'.join(lines)

            with open(part_path, 'w') as f:
                f.write(obj_text_with_o)

            parts.append((f"link_{idx}", part_filename))

    collision_file = f"{name}_vhacd.obj" if concave else f"{name}.obj"
    if concave:
        decompose_obj(directory, name)

    links_xml = []
    joints_xml = []
    for i, (link_name, mesh_file) in enumerate(parts):
        links_xml.append(f'''
  <link name="{link_name}">
    <inertial>
      <mass value="{mass / len(parts):.4f}"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
    <visual>
      <geometry><mesh filename="{mesh_file}"/></geometry>
    </visual>
    <collision>
      <geometry><mesh filename="{collision_file}"/></geometry>
    </collision>
  </link>''')

        if i > 0:
            joints_xml.append(f'''
  <joint name="fixed_{parts[i - 1][0]}_{link_name}" type="fixed">
    <parent link="{parts[i - 1][0]}"/>
    <child link="{link_name}"/>
  </joint>''')

    urdf_content = f'''<?xml version="1.0"?>
<robot name="{name}">
{''.join(links_xml)}
{''.join(joints_xml)}
</robot>'''

    urdf_path = os.path.join(directory, f"{name}.urdf")
    with open(urdf_path, 'w') as f:
        f.write(urdf_content)

    print(f"[URDF] Created with {len(parts)} link(s): {urdf_path}")

def create_obj(obj_name, mass, concave=False):
    base_path = os.path.join("sim/objects", obj_name)

    for subdir in os.listdir(base_path):
        folder_path = os.path.join(base_path, subdir)

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and not (filename.endswith('.glb') or filename.endswith('.blend')):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"[Cleanup] Error deleting {file_path}: {e}")

        glb_path = os.path.join(folder_path, f"{subdir}.glb")
        obj_path = os.path.join(folder_path, f"{subdir}.obj")

        scene = trimesh.load(glb_path, force='scene')
        scene.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        scene.export(obj_path)
        print(f"[Export] Converted to OBJ: {obj_path}")

        create_urdf(folder_path, subdir, mass, concave)

def load_object(name, pos, orn=[0, 0, 0], scale=1, bodyType=None, verbose=0):
    if bodyType is None:
        bodyType = random.randint(1, len(os.listdir(f"sim/objects/{name}")))
    urdf_path = f"sim/objects/{name}/{name}_{bodyType}/{name}_{bodyType}.urdf"
    orn = p.getQuaternionFromEuler(orn)
    bodyId = p.loadURDF(urdf_path, pos, orn, useFixedBase=False,
                        globalScaling=scale, flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
    p.changeVisualShape(bodyId, -1, rgbaColor=[1, 1, 1, 1])
    if verbose:
        print(f"Loaded object {name} at {pos}")
    return bodyId, bodyType

def load_table(gridSize, pos=[0, 0, 0], verbose=0):
    if gridSize == (100, 100):
        file_path = "sim/objects/table/table_1.1/table_1.1.urdf"
    elif gridSize == (100, 300):
        file_path = "sim/objects/table/table_1.3/table_1.3.urdf"
    else:
        raise ValueError("Invalid grid size. Supported sizes are (100, 100) and (100, 300).")
    bodyId = p.loadURDF(file_path, pos, useFixedBase=True)
    p.changeVisualShape(bodyId, -1, rgbaColor=[1, 1, 1, 1])
    if verbose > 0:
        print(f"Loaded table with grid size {gridSize}")
    return bodyId

def get_size(obj_id):
	# Get the bounding box of the object
	bounding_box = p.getAABB(obj_id)
	# Calculate the size of the bounding box
	size = np.array(bounding_box[1]) - np.array(bounding_box[0])
	return size

class PyBulletSim:
	def __init__(self, mode=p.DIRECT):
		try:
			p.disconnect()
		except:
			pass
		
		if mode == p.DIRECT:
			self.client = p.connect(p.DIRECT, options="--egl")
		else:
			self.client = p.connect(mode)
			
		p.setPhysicsEngineParameter(enableFileCaching=0)
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		p.resetSimulation()
		p.setGravity(0, 0, -9.8)
		p.setRealTimeSimulation(0)
		p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
		p.loadURDF("plane.urdf")

		self.time_step = p.getPhysicsEngineParameters()['fixedTimeStep']

	def step(self, duration=0):
		for _ in range(int(duration / self.time_step)):
			p.stepSimulation()

	def close(self):
		if p.getConnectionInfo(self.client)['connectionMethod'] == p.GUI:
			while p.isConnected():
				p.stepSimulation()
				time.sleep(self.time_step)
		try:
			p.disconnect()
		except:
			pass

class Camera:
	def __init__(self, target_pos, distance, yaw, pitch, roll, width=640, height=480,
					near=0.01, far=100, fov=60):
		self.target = target_pos
		self.distance = distance
		self.yaw, self.pitch, self.roll = yaw, pitch, roll
		self.width, self.height = width, height
		self.near, self.far = near, far
		self.fov = fov
		self.up_axis = 2

		p.resetDebugVisualizerCamera(
			cameraDistance=distance, 
			cameraTargetPosition=target_pos, 
			cameraYaw=yaw, cameraPitch=pitch
		)

		self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
			self.target, self.distance, self.yaw, self.pitch, self.roll, self.up_axis)
		self.proj_matrix = p.computeProjectionMatrixFOV(
			self.fov, self.width/self.height, self.near, self.far)

		self.V = np.array(self.view_matrix).reshape(4, 4, order='F')
		self.P = np.array(self.proj_matrix).reshape(4, 4, order='F')

	def capture_image(self):
		_, _, rgb, _, _ = p.getCameraImage(
			self.width, self.height,
			viewMatrix=self.view_matrix,
			projectionMatrix=self.proj_matrix,
			shadow=True,
			renderer=p.ER_BULLET_HARDWARE_OPENGL)
		rgba = np.reshape(rgb, (self.height, self.width, 4)).astype(np.uint8)
		return rgba[:, :, :3]

	def project_points(self, world_points):
		N = world_points.shape[0]
		clip_pts = self.P @ (self.V @ np.vstack([world_points.T, np.ones(N)]))
		ndc = clip_pts / clip_pts[3]
		u = (ndc[0]*0.5 + 0.5) * self.width
		v = (1 - (ndc[1]*0.5 + 0.5)) * self.height
		return u, v

	def compute_bounding_box(self, bodyName, bodyId, bodyType, numSamples=200):
		mesh_path = f"sim/objects/{bodyName}/{bodyName}_{bodyType}/{bodyName}_{bodyType}.obj"
		mesh = trimesh.load_mesh(mesh_path)
		points, _ = trimesh.sample.sample_surface(mesh, numSamples)
		pos, orn = p.getBasePositionAndOrientation(bodyId)
		R = trimesh.transformations.quaternion_matrix([orn[3], *orn[:3]])[:3, :3]
		world_pts = (R @ points.T).T + np.array(pos)
		u, v = self.project_points(world_pts)
		u_min, u_max = np.floor(u.min()).astype(int), np.ceil(u.max()).astype(int)
		v_min, v_max = np.floor(v.min()).astype(int), np.ceil(v.max()).astype(int)
		return (u_min, v_min, u_max, v_max)

	def show_img(self, image, title=''):
		fig, ax = plt.subplots(figsize=(self.width / 100, self.height / 100), dpi=100)
		ax.imshow(image)
		plt.title(title)
		plt.axis('off')
		plt.show()

	def draw_bounding_boxes(self, image, boxes=[], labels=[], color='yellow', title=''):
		fig, axs = plt.subplots(1, 2, figsize=(2 * self.width / 100, self.height / 100), dpi=100)

		# Subplot 1: Raw image
		axs[0].imshow(image)
		axs[0].set_title("Raw Image")
		axs[0].axis('off')

		# Subplot 2: Image with bounding boxes
		axs[1].imshow(image)
		axs[1].set_title("Image with Bounding Boxes")
		for i, (u_min, v_min, u_max, v_max) in enumerate(boxes):
			rect = patches.Rectangle(
				(u_min, v_min), u_max - u_min, v_max - v_min,
				linewidth=2, edgecolor=color, facecolor='none'
			)
			axs[1].add_patch(rect)
			if labels and i < len(labels):
				axs[1].text(u_min, v_min-10, labels[i], color=color,
							fontsize=8, backgroundcolor='black')
		axs[1].axis('off')
		
		plt.suptitle(title)
		plt.tight_layout()
		plt.show()

class Joint(object):
    def __init__(self, physics_client, model_id, joint_id, limits):
        self.physics_client = physics_client
        self.model_id = model_id
        self.jid = joint_id
        self.limits = limits

    def get_position(self):
        joint_state = self.physics_client.getJointState(self.model_id, self.jid)
        return joint_state[0]

    def set_position(self, position, max_force=100.):
        self.physics_client.setJointMotorControl2(
            self.model_id,
            self.jid,
            controlMode=self.physics_client.POSITION_CONTROL,
            targetPosition=position,
            force=max_force,
            positionGain=0.5,
            velocityGain=1.0
        )

class Robot():
	def __init__(
			self,
			physics_client, 
			model_path, rtb_model, 
			initial_base_pos=[0, 0, 0], 
			useFixedBase=True
		):
		self.physics_client = physics_client
		self.model_id = self.physics_client.loadURDF(
			model_path, 
			initial_base_pos, 
			useFixedBase=useFixedBase
		)
		self.rtb_model = rtb_model
	
		self._time_step = 1. / 240.
		self._left_finger_id = 9
		self._right_finger_id = 10

	def init(self, mode, x_range=None, y_range=None):
		self.mode = mode
		self.x_range = x_range
		self.y_range = y_range
		return self.load_model()

	def load_model(self):
		joints = {}
		for i in range(self.physics_client.getNumJoints(self.model_id)):
			joint_info = self.physics_client.getJointInfo(self.model_id, i)
			joint_limits = {
				'lower': joint_info[8], 
				'upper': joint_info[9],
				'force': joint_info[10]
			}
			joints[i] = Joint(self.physics_client, self.model_id, i, joint_limits)
			# print(joint_info)
		
		self.rtb_model.qlim = np.array([[joints[i].limits['lower'], joints[i].limits['upper']] for i in range(7)]).T
		self.joints = joints
		self._left_finger = self.joints[self._left_finger_id]
		self._right_finger = self.joints[self._right_finger_id]

		self.reset_joints()
		return self.model_id

	def reset_joints(self, initial_positions=[0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0, 0]):
		for jid in range(len(initial_positions)):
			self.physics_client.resetJointState(self.model_id, jid, initial_positions[jid])
		
		self.init_ee_pos = self.get_ee_pos()

	def reset_ee(self):
		self.move_ee(self.init_ee_pos[0], self.init_ee_pos[1])

	def run(self, duration):
		for _ in range(int(duration / self._time_step)):
			self.physics_client.stepSimulation()

	def get_pos(self):
		return [self.joints[i].get_position() for i in range(7)]

	def get_ee_pos(self):
		ee_pos, ee_orn = self.physics_client.getLinkState(self.model_id, 11)[:2]
		return ee_pos, ee_orn

	def open_gripper(self, max_limit=True):
		if max_limit:
			self._left_finger.set_position(self._left_finger.limits['upper'])
			self._right_finger.set_position(self._left_finger.limits['upper'])
		else:
			self._left_finger.set_position(self._left_finger.limits['upper']/2)
			self._right_finger.set_position(self._left_finger.limits['upper']/2)
		
		self.run(0.2)

	def close_gripper(self):
		self._left_finger.set_position(self._left_finger.limits['lower'])
		self._right_finger.set_position(self._left_finger.limits['lower'])

		self.run(0.2)

	def ikine(self, T: sm.SE3, orientation: list, q0: Union[ArrayLike, None]=None, itr: int=100) -> np.ndarray:
		"""
		Parameters
		----------
		T : sm.SE3
			The desired end-effector pose.
		q0 : ArrayLike, optional
			The initial guess for the joint angles. If None, the initial guess is set to the zero vector.
		itr : int, optional
			The number of iterations to attempt in case the initial IK solution is invalid.
		
		Returns
		-------
		q : np.ndarray
			The joint angles that achieve the desired end-effector pose.
		"""
		# if q0 is None:
		# 	q0 = np.zeros(7)
		
		# lower_limit_joints = [self.joints[i].limits['lower'] for i in range(7)]
		# upper_limit_joints = [self.joints[i].limits['upper'] for i in range(7)]

		# succes = False
		# for _ in range(itr):
		# 	ik_solution = self.rtb_model.ikine_LM(T, q0=q0, joint_limits=True)
		# 	if ik_solution.success:
		# 		q = ik_solution.q
		# 		succes = True
		# 		break
		# 	else:
		# 		q0 = np.random.uniform(lower_limit_joints, upper_limit_joints)

		# if not succes:
		# 	raise ValueError('Could not find a valid IK solution.')
		
		# if np.any(q < lower_limit_joints) or np.any(q > upper_limit_joints):
		# 	print('IK solution out of joint limits.')
		# 	q = np.clip(q, lower_limit_joints, upper_limit_joints)

		q = self.physics_client.calculateInverseKinematics(
			self.model_id, 
			11, 
			T.t, 
			self.physics_client.getQuaternionFromEuler(orientation),
			lowerLimits=[self.joints[i].limits['lower'] for i in range(7)],
			upperLimits=[self.joints[i].limits['upper'] for i in range(7)],
			maxNumIterations=itr,
		)
		return q[:7]

	def generate_linear_path(self, start_pos: sm.SE3, end_pos: sm.SE3, num_points: int, orientation: list) -> np.ndarray:
		"""
		Generates a linear path between two poses.

		Parameters
		----------
		start_pos : sm.SE3
			The start pose.
		end_pos : sm.SE3
			The end pose.
		num_points : int
			The number of points to generate along the path.

		Returns
		-------
		path : np.ndarray
			An array of shape (num_points, 7) containing the joint angles along the path.
		"""

		trajectory = rtb.ctraj(start_pos, end_pos, num_points)

		path = np.zeros((num_points, 7))
		for i in range(num_points):
			try:
				path[i] = self.ikine(trajectory[i], orientation)
			except ValueError as e:
				print(f"IK failed at point {i+1}/{num_points}: {e}")
				break  
	
		return path

	def map_to_sides(self, pos, x_range, y_range):
		x, y, z = pos
		distances = {
			'0': x - x_range[0],		# Top side
			'1': y_range[1] - y,		# Right side
			'2': x_range[1] - x,		# Bottom side
			'3': y - y_range[0],		# Left side
		}
		
		thresh = 0.1

		# Find the side with the minimum distance
		closest_side = min(distances, key=distances.get)
		if closest_side == '0':
			target_pos = [x_range[0]-thresh, y, z]
		elif closest_side == '1':
			target_pos = [x, y_range[1]+thresh, z]
		elif closest_side == '2':
			target_pos = [x_range[1]+thresh, y, z]
		else:
			target_pos = [x, y_range[0]-thresh, z]
		
		return closest_side, target_pos

	def go_to_pose(self, target_pos):
		if self.mode == "mobile":
			pre_base_pos, _ = self.physics_client.getBasePositionAndOrientation(self.model_id)
			self.move_base_along_table(target_pos, duration=10)
			base_pos, _ = self.physics_client.getBasePositionAndOrientation(self.model_id)

			# print(pre_base_pos, base_pos)
			# if pre_base_pos == base_pos:
			# 	print("In position.")
			# else:
			# 	print("Going to the position.")
			# 	self.init_ee_pos = self.get_ee_pos()

			state = self.map_to_sides(base_pos, self.x_range, self.y_range)[0]
			if state == '0':
				target_orn=[np.pi, 0, 0]
			elif state == '1':
				target_orn=[np.pi, 0, -np.pi/2]
			elif state == '2':
				target_orn=[np.pi, 0, np.pi]
			else:
				target_orn=[np.pi, 0, np.pi/2]
		else:
			target_orn=[np.pi, 0, 0]
		
		self.move_ee(target_pos, target_orn)
		
	def move_ee(self, target_pos, target_orn):
		position, orientation = self.get_ee_pos()
		orientation = self.physics_client.getEulerFromQuaternion(orientation)

		current_pos = sm.SE3(position)
		current_orn = sm.SE3.RPY(orientation, order='xyz', unit='deg')
		cur_pos = current_pos * current_orn

		translation = sm.SE3(target_pos)
		rotation = sm.SE3.RPY(target_orn, order='xyz', unit='rad')
		tar_pos = translation * rotation
		
		# q = self.ikine(tar_pos)
		# self.reset(q)
		# self.rotate_joints(q)

		for q in self.generate_linear_path(cur_pos, tar_pos, 20, target_orn):
			# self.reset(q)
			self.rotate_joints(q)
			self.run(1)

	def rotate_joints(self, q):
		for i in range(7):
			self.joints[i].set_position(q[i])
			# self.run(0.2)

	def move_base(self, new_base_pos, new_base_orn):
		"""
		Instantly moves the robot base to a new position and orientation.

		Parameters:
			new_base_pos (list or tuple of float): The new (x, y, z) position.
			new_base_orn (list or tuple of float): The new orientation as a quaternion (x, y, z, w).
		"""
		self.physics_client.resetBasePositionAndOrientation(self.model_id, new_base_pos, new_base_orn)

	def move_base_smooth(self, target_base_pos, duration=1.0, steps=100):
		"""
		Smoothly moves the robot base to a new position and orientation over a specified duration.

		Parameters:
			target_base_pos (list or tuple of float): The target (x, y, z) position.
			duration (float): Total duration of the movement in seconds.
			steps (int): Number of intermediate steps.
		"""
		# Get the current base position and orientation
		current_pos, current_orn = self.physics_client.getBasePositionAndOrientation(self.model_id)
		target_base_pos = [target_base_pos[0], target_base_pos[1], current_pos[2]]  # Keep the z-coordinate unchanged
		
		# Create a linear interpolation for the position
		pos_traj = np.linspace(current_pos, target_base_pos, steps)
		
		# Move along the trajectory
		for i in range(steps):
			t = i / (steps - 1)
			# Update the base position and orientation
			self.physics_client.resetBasePositionAndOrientation(self.model_id, pos_traj[i], current_orn)
			self.run(duration / steps)

	def rotate_base_smooth(self, target_base_orn, duration=1.0, steps=100):
		"""
		Smoothly rotates the robot base to a new orientation over a specified duration.

		Parameters:
			target_base_orn (list or tuple of float): The target orientation as a quaternion (x, y, z, w).
			duration (float): Total duration of the movement in seconds.
			steps (int): Number of intermediate steps.
		"""
		# Get the current base position and orientation
		current_pos, current_orn = self.physics_client.getBasePositionAndOrientation(self.model_id)
		target_base_orn = self.physics_client.getQuaternionFromEuler(target_base_orn)
		
		# Create a linear interpolation for the orientation
		orn_traj = np.zeros((steps, 4))
		for i in range(steps):
			t = i / (steps - 1)
			orn_traj[i] = self.physics_client.getQuaternionSlerp(current_orn, target_base_orn, t)
		
		# Move along the trajectory
		for i in range(steps):
			# Update the base position and orientation
			self.physics_client.resetBasePositionAndOrientation(self.model_id, current_pos, orn_traj[i])
			self.run(duration / steps)

	def move_base_to_corner(self, pre_s, new_s, z, duration=1.0):
		side_orientations = {
			'0': [0, 0, 0],            # Top side -> Facing down
			'1': [0, 0, -np.pi/2],     # Right side -> Facing left
			'2': [0, 0, np.pi],        # Bottom side -> Facing up
			'3': [0, 0, np.pi/2],      # Left side -> Facing right
		}

		thresh = 0.1

		corners = {
			('0', '1'): [self.x_range[0]-thresh, self.y_range[1]+thresh],
			('1', '0'): [self.x_range[0]-thresh, self.y_range[1]+thresh],
			('0', '3'): [self.x_range[0]-thresh, self.y_range[0]-thresh],
			('3', '0'): [self.x_range[0]-thresh, self.y_range[0]-thresh],
			('1', '2'): [self.x_range[1]+thresh, self.y_range[1]+thresh],
			('2', '1'): [self.x_range[1]+thresh, self.y_range[1]+thresh],
			('2', '3'): [self.x_range[1]+thresh, self.y_range[0]-thresh],
			('3', '2'): [self.x_range[1]+thresh, self.y_range[0]-thresh],
		}
		
		assert (pre_s, new_s) in corners

		corner_pos = corners[(pre_s, new_s)] + [z]
		self.move_base_smooth(corner_pos, duration)
		self.rotate_base_smooth(side_orientations[new_s], duration)
	
	def move_base_along_table(self, target_pos, duration=1.0):
		current_pos, _ = self.physics_client.getBasePositionAndOrientation(self.model_id)

		pre_s, _ = self.map_to_sides(current_pos, self.x_range, self.y_range)
		new_s, mapped_target = self.map_to_sides(target_pos, self.x_range, self.y_range)

		if pre_s == new_s:
			self.move_base_smooth(mapped_target, duration)
		elif abs(int(pre_s) - int(new_s)) == 1 or abs(int(pre_s) - int(new_s)) == 3:
			self.move_base_to_corner(pre_s, new_s, mapped_target[2], duration)
			self.move_base_smooth(mapped_target, duration)
		else:
			if pre_s == '0':
				if self.y_range[1] - target_pos[1] + self.y_range[1] - current_pos[1] < target_pos[1] - self.y_range[0] + current_pos[1] - self.y_range[0]:
					# clockwise
					self.move_base_to_corner(pre_s, '1', mapped_target[2], duration)
					self.move_base_to_corner('1', new_s, mapped_target[2], duration)
				else:
					# counter-clockwise
					self.move_base_to_corner(pre_s, '3', mapped_target[2], duration)
					self.move_base_to_corner('3', new_s, mapped_target[2], duration)
			elif pre_s == '1':
				if self.x_range[1] - target_pos[0] + self.x_range[1] - current_pos[0] < target_pos[0] - self.x_range[0] + current_pos[0] - self.x_range[0]:
					# clockwise
					self.move_base_to_corner(pre_s, '2', mapped_target[2], duration)
					self.move_base_to_corner('2', new_s, mapped_target[2], duration)
				else:
					# counter-clockwise
					self.move_base_to_corner(pre_s, '0', mapped_target[2], duration)
					self.move_base_to_corner('0', new_s, mapped_target[2], duration)
			elif pre_s == '2':
				if self.y_range[1] - target_pos[1] + self.y_range[1] - current_pos[1] > target_pos[1] - self.y_range[0] + current_pos[1] - self.y_range[0]:
					# clockwise
					self.move_base_to_corner(pre_s, '3', mapped_target[2], duration)
					self.move_base_to_corner('3', new_s, mapped_target[2], duration)
				else:
					# counter-clockwise
					self.move_base_to_corner(pre_s, '1', mapped_target[2], duration)
					self.move_base_to_corner('1', new_s, mapped_target[2], duration)
			else:
				if self.x_range[1] - target_pos[0] + self.x_range[1] - current_pos[0] > target_pos[0] - self.x_range[0] + current_pos[0] - self.x_range[0]:
					# clockwise
					self.move_base_to_corner(pre_s, '0', mapped_target[2], duration)
					self.move_base_to_corner('0', new_s, mapped_target[2], duration)
				else:
					# counter-clockwise
					self.move_base_to_corner(pre_s, '2', mapped_target[2], duration)
					self.move_base_to_corner('2', new_s, mapped_target[2], duration)
			self.move_base_smooth(mapped_target, duration)
