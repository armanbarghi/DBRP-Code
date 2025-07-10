import os
import time
import random
import numpy as np
import pybullet as p
import pybullet_data
import trimesh

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

def create_urdf(directory, name, mass, use_concave_collision=False):
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

	collision_file = f"{name}_vhacd.obj" if use_concave_collision else f"{name}.obj"
	if use_concave_collision:
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

def convert_glb_to_urdf(obj_name, mass, use_concave_collision=False):
	base_path = os.path.join("core/sim/objects", obj_name)

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

		create_urdf(folder_path, subdir, mass, use_concave_collision)

def load_object_urdf(name, pos, orn=[0, 0, 0], scale=1, body_type=None, verbose=0, use_fixed_base=False):
	if body_type is None:
		body_type = random.randint(1, len(os.listdir(f"core/sim/objects/{name}")))
	urdf_path = f"core/sim/objects/{name}/{name}_{body_type}/{name}_{body_type}.urdf"
	orn = p.getQuaternionFromEuler(orn)
	body_id = p.loadURDF(urdf_path, pos, orn, useFixedBase=use_fixed_base,
						globalScaling=scale, flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
	p.changeVisualShape(body_id, -1, rgbaColor=[1, 1, 1, 1])
	if verbose:
		print(f"Loaded object {name} at {pos}")
	return body_id, body_type

def load_table_urdf(grid_size, pos=[0, 0, 0], verbose=0):
	if grid_size == (100, 100):
		file_path = "core/sim/objects/table/table_1.1/table_1.1.urdf"
	elif grid_size == (100, 300):
		file_path = "core/sim/objects/table/table_1.3/table_1.3.urdf"
	else:
		raise ValueError("Invalid grid size. Supported sizes are (100, 100) and (100, 300).")
	body_id = p.loadURDF(file_path, pos, useFixedBase=True)
	p.changeVisualShape(body_id, -1, rgbaColor=[1, 1, 1, 1])
	if verbose > 0:
		print(f"Loaded table with grid size {grid_size}")
	return body_id

def get_bounding_box_size(obj_id):
	# Get the bounding box of the object
	bounding_box = p.getAABB(obj_id)
	# Calculate the size of the bounding box
	size = np.array(bounding_box[1]) - np.array(bounding_box[0])
	return size

def random_tilt(pos, orn, max_shift=0.05, shift_end=False, tilt_angle=10):
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
