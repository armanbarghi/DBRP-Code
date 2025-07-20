import os
import random
import trimesh
import numpy as np
import time
import pybullet as p
import pybullet_data


def load_object_urdf(objects_dir, obj_name, pos, orn=[0, 0, 0], scale=1, body_type=None, use_fixed_base=False):
	"""
	Load an object from URDF file into the simulation.
	
	Uses temporary working directory change to ensure PyBullet resolves
	relative material and texture paths from the correct object directory,
	preventing resource conflicts between different object variants.
	"""
	if body_type is None:
		# Safer to filter for directories to avoid errors with files in the object folder
		object_dir = os.path.join(objects_dir, obj_name)
		body_type_dirs = [d for d in os.listdir(object_dir) if os.path.isdir(os.path.join(object_dir, d))]
		body_type = random.randint(1, len(body_type_dirs))

	# This is the specific directory for the object variant, e.g., .../objects/apple/apple_1
	object_body_path = os.path.join(objects_dir, obj_name, f"{obj_name}_{body_type}")
	urdf_filename = f"{obj_name}_{body_type}.urdf"
	urdf_path = os.path.join(object_body_path, urdf_filename)

	if not os.path.exists(urdf_path):
		raise FileNotFoundError(f"URDF file not found: {urdf_path}")

	orn = p.getQuaternionFromEuler(orn)

	# Change working directory to resolve relative material/texture paths correctly
	original_dir = os.getcwd()
	try:
		os.chdir(object_body_path)
		body_id = p.loadURDF(
			urdf_filename,
			pos,
			orn,
			useFixedBase=use_fixed_base,
			globalScaling=scale,
			flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL
		)
	finally:
		os.chdir(original_dir)

	p.changeVisualShape(body_id, -1, rgbaColor=[1, 1, 1, 1])
	return body_id, body_type

def load_table_urdf(objects_dir, grid_size, pos=[0, 0, 0]):
	"""
	Load a table from URDF file based on grid size.
	
	Uses temporary working directory change to ensure PyBullet resolves
	relative material and texture paths from the correct table directory.
	"""
	if grid_size == (100, 100):
		table_dir = os.path.join(objects_dir, "table", "table_1.1")
		urdf_filename = "table_1.1.urdf"
	if grid_size == (100, 200):
		table_dir = os.path.join(objects_dir, "table", "table_1.2")
		urdf_filename = "table_1.2.urdf"
	elif grid_size == (100, 300):
		table_dir = os.path.join(objects_dir, "table", "table_1.3")
		urdf_filename = "table_1.3.urdf"
	else:
		raise ValueError("Invalid grid size. Supported sizes are (100, 100) and (100, 300).")
	
	urdf_path = os.path.join(table_dir, urdf_filename)
	if not os.path.exists(urdf_path):
		raise FileNotFoundError(f"URDF file not found: {urdf_path}")

	# Change working directory to resolve relative material/texture paths correctly
	original_dir = os.getcwd()
	try:
		os.chdir(table_dir)
		body_id = p.loadURDF(urdf_filename, pos, useFixedBase=True)
	finally:
		os.chdir(original_dir)

	p.changeVisualShape(body_id, -1, rgbaColor=[1, 1, 1, 1])
	return body_id

def get_object_extents(body_id):
	"""
	Get the 3D bounding box dimensions (width, height, depth) of a loaded object.
	"""
	bounding_box = p.getAABB(body_id)
	return np.array(bounding_box[1]) - np.array(bounding_box[0])

def get_object_footprint_size(objects_dir, obj_name, body_type):
	"""
	Get the maximum XY-plane extent (footprint size) of an object from its mesh file.
	"""
	object_body_path = os.path.join(objects_dir, obj_name, f"{obj_name}_{body_type}")
	mesh_path = os.path.join(object_body_path, f"{obj_name}_{body_type}.obj")
	mesh = trimesh.load_mesh(mesh_path)
	extents = mesh.bounding_box.extents
	footprint_size = np.max(extents[:2])  # XY-plane maximum extent
	if obj_name in ['basket', 'box']:
		footprint_size = (footprint_size + np.linalg.norm(extents[:2])) / 2
	return footprint_size

def get_object_type_max_footprints(objects_dir):
	"""
	Compute the maximum XY-plane footprint size for each object type.
	"""
	p.connect(p.DIRECT)
	max_footprints = {}

	object_names = [d for d in os.listdir(objects_dir) if os.path.isdir(os.path.join(objects_dir, d))]

	for name in object_names:
		try:
			body_types = [
				int(folder.split("_")[-1])
				for folder in os.listdir(os.path.join(objects_dir, name))
				if folder.startswith(f"{name}_")
			]
		except Exception as e:
			print(f"Skipping {name}: {e}")
			continue

		footprint_sizes = []
		for body_type in sorted(body_types):
			try:
				footprint_size = get_object_footprint_size(objects_dir, name, body_type)
				footprint_sizes.append(footprint_size)
			except Exception as e:
				print(f"Failed to process {name} bodyType {body_type}: {e}")
		if footprint_sizes:
			max_footprints[name] = np.round(max(footprint_sizes), 2)

	p.disconnect()
	return max_footprints

def apply_random_tilt(pos, orn, max_shift=0.05, shift_end=False, tilt_angle=10):
	"""
	Randomly shifts x and y coordinates and tilts the orientation 
	opposite to the shift direction.
	"""
	x0, y0, z0 = pos
	roll0, pitch0, yaw0 = orn

	if shift_end:
		# Pick a random angle and place shift on the circle
		theta = np.random.uniform(0, 2 * np.pi)
		dx = max_shift * np.cos(theta)
		dy = max_shift * np.sin(theta)
	else:
		# Uniform random in the square [-max_shift, +max_shift]
		dx = np.random.uniform(-max_shift, max_shift)
		dy = np.random.uniform(-max_shift, max_shift)

	new_pos = np.array([x0 + dx, y0 + dy, z0])

	# Compute opposite tilt direction
	opposite_dir = np.array([-dx, -dy, 0])
	if np.linalg.norm(opposite_dir) > 1e-6:
		opposite_dir = opposite_dir / np.linalg.norm(opposite_dir)  # normalize

	# Apply tilt proportional to displacement
	tilt_roll = -opposite_dir[1] * np.deg2rad(tilt_angle)  # around x-axis
	tilt_pitch = opposite_dir[0] * np.deg2rad(tilt_angle)  # around y-axis

	# New orientation by adding tilt to initial
	new_orn = np.array([roll0 + tilt_roll, pitch0 + tilt_pitch, yaw0])

	return new_pos, new_orn

def adjust_object_pose_for_stacking(obj, objects, base_z):
	"""
	Adjust position and orientation for an object considering stacking hierarchy and base objects.
	"""
	stack_hierarchy = obj['stack_hierarchy']
	# Create a mapping from object_id to its index in the list for quick lookup
	object_id_to_index = {o['object_id']: i for i, o in enumerate(objects)}

	pos = [obj['pos'][0], obj['pos'][1], base_z + 0.05 + 0.085 * stack_hierarchy]
	orn = [0, 0, 0]

	if stack_hierarchy > 0:
		base_object_id = obj['base_id']
		base_object = objects[object_id_to_index[base_object_id]]

		# Inherit the final X, Y position from the base object
		if 'final_pos' in base_object:
			pos[0] = base_object['final_pos'][0]
			pos[1] = base_object['final_pos'][1]
		base_label = base_object['label']

		if obj['label'] in [0, 1, 2]:	# fork, spoon, knife
			orn[0] = np.pi / 2
			if base_label in [6, 7]:	# paper-cup, mug
				pos, orn = apply_random_tilt(pos, orn, max_shift=0.01, tilt_angle=10)
			elif base_label == 8:
				pos, orn = apply_random_tilt(pos, orn, max_shift=0.02, shift_end=True, tilt_angle=10)
			else:
				orn[2] = random.uniform(0, 2 * np.pi)
		else:
			if base_label == 11:	# pot
				pos, _ = apply_random_tilt(pos, orn, max_shift=0.02, tilt_angle=0)
			elif base_label in [9, 10]:	# basket, box
				pos, _ = apply_random_tilt(pos, orn, max_shift=0.01, tilt_angle=0)
			orn[2] = random.uniform(0, 2 * np.pi)
	else:
		orn[2] = random.uniform(0, 2 * np.pi)

	return pos, orn


class PyBulletSim:
	"""PyBullet physics simulation manager."""
	
	def __init__(self, mode=p.DIRECT):
		"""Initialize the PyBullet simulation."""
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
		p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
		p.loadURDF("plane.urdf")

		self.time_step = p.getPhysicsEngineParameters()['fixedTimeStep']

	def step(self, duration=0):
		"""Step the simulation for a given duration."""
		for _ in range(int(duration / self.time_step)):
			p.stepSimulation()

	def run(self):
		"""Keep the GUI simulation running indefinitely."""
		if p.getConnectionInfo(self.client)['connectionMethod'] == p.GUI:
			try:
				while p.isConnected():
					p.stepSimulation()
					time.sleep(self.time_step)
			except KeyboardInterrupt:
				print("Simulation stopped by user.")
		else:
			print("Warning: GUI simulation can only run in GUI mode.")

	def close(self):
		"""Close the simulation."""
		try:
			p.disconnect()
		except:
			pass
