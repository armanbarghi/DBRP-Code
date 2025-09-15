import os
import json
import random
import pybullet as p
import pybullet_data
from PIL import Image
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict, Counter
from core.env.scene_manager import OBJECTS
from core.sim.physics_utils import load_object_urdf, adjust_object_pose_for_stacking


def select_rearrangement_dir(dataset_dir: str, scene_id: Optional[int]=None, viewpoint_id: Optional[int]=None) -> Tuple[str, int]:
	"""
	Selects a rearrangement directory either randomly or by a specific scene ID.

	This helper function centralizes the logic for finding and selecting a
	rearrangement directory from the dataset.
	"""
	if not os.path.exists(dataset_dir):
		raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist")

	selected_dir_name = None
	if scene_id is not None:
		# Load a specific scene by ID
		selected_dir_name = f"rearrangement_{scene_id:05d}"
		full_path = os.path.join(dataset_dir, selected_dir_name)
	else:
		# Find all valid rearrangement directories
		rearrangement_dirs = [
			item for item in os.listdir(dataset_dir)
			if os.path.isdir(os.path.join(dataset_dir, item)) and item.startswith('rearrangement_')
		]
		if not rearrangement_dirs:
			raise FileNotFoundError(f"No rearrangement directories found in '{dataset_dir}'")
		
		if viewpoint_id:
			vp_id = -1
			while vp_id != viewpoint_id:
				# Select a directory at random
				selected_dir_name = random.choice(rearrangement_dirs)
				scene_id = int(selected_dir_name.split('_')[-1])

				full_path = os.path.join(dataset_dir, selected_dir_name)
				initial_lbl = os.path.join(full_path, 'initial_labels.json')
				with open(initial_lbl, 'r') as f:
					labels = json.load(f)
				cam_info = labels['camera_info']
				vp_id = cam_info['viewpoint_id']
		else:
			# Select a directory at random
			selected_dir_name = random.choice(rearrangement_dirs)
			scene_id = int(selected_dir_name.split('_')[-1])
			full_path = os.path.join(dataset_dir, selected_dir_name)

	# Return the full path to the selected directory and the scene ID
	return full_path, scene_id

def visualize_rearrangement(dataset_dir: str, scene_id: Optional[int]=None, show_bbx=False, figsize=(9, 4)):
	"""
	Visualizes the initial and target scenes of a rearrangement.
	"""
	def _draw_scene(ax, image_path, label_path, title):
		"""Inner function to draw a single scene with optional bounding boxes."""
		image = Image.open(image_path)
		ax.imshow(image)

		with open(label_path, 'r') as f:
			labels = json.load(f)
		
		if isinstance(labels, list):
			object_labels = labels
			viewpoint_id = 8
		else:
			object_labels = labels['objects']
			cam_info = labels['camera_info']
			viewpoint_id = cam_info['viewpoint_id']
		title += f" (Viewpoint ID: {viewpoint_id})"

		if show_bbx:
			for obj in object_labels:
				bbox = obj['bbox']
				xmin, ymin, xmax, ymax = bbox
				width, height = xmax - xmin, ymax - ymin

				rect = patches.Rectangle(
					(xmin, ymin), width, height,
					linewidth=2, edgecolor='yellow', facecolor='none'
				)
				ax.add_patch(rect)
				ax.text(xmin, ymin - 10, f"{obj['obj_id']}_{obj['model_name']}", color='yellow', fontsize=8)

		ax.set_title(title)
		ax.axis('off')

	try:
		# Use the helper function to select the directory
		selected_dir_path, scene_id_val = select_rearrangement_dir(dataset_dir, scene_id)
	except FileNotFoundError as e:
		print(e)
		return

	initial_img = os.path.join(selected_dir_path, 'initial_image.png')
	target_img = os.path.join(selected_dir_path, 'target_image.png')
	initial_lbl = os.path.join(selected_dir_path, 'initial_labels.json')
	target_lbl = os.path.join(selected_dir_path, 'target_labels.json')

	if not all(os.path.exists(path) for path in [initial_img, target_img, initial_lbl, target_lbl]):
		print(f"One or more image/label files are missing in {selected_dir_path}.")
		return

	fig, axes = plt.subplots(1, 2, figsize=figsize)

	_draw_scene(axes[0], initial_img, initial_lbl, 'Initial Scene')
	_draw_scene(axes[1], target_img, target_lbl, 'Target Scene')

	fig.suptitle(f'Scene ID: {scene_id_val}')
	plt.tight_layout()
	plt.show()

def load_rearrangement_meta(dataset_dir: str, scene_id: Optional[int]=None) -> Optional[dict]:
	"""
	Loads a rearrangement meta file either by scene ID or randomly.
	"""
	try:
		# Use the helper function to select the directory
		selected_dir_path, _ = select_rearrangement_dir(dataset_dir, scene_id)
		meta_file_path = os.path.join(selected_dir_path, 'meta.json')
		if not os.path.exists(meta_file_path):
			print(f"meta.json not found in {os.path.basename(selected_dir_path)}")
			return None
	except FileNotFoundError as e:
		print(e)
		return None

	# Load and return the meta file
	with open(meta_file_path, 'r') as f:
		meta_data = json.load(f)

	return meta_data

def grid_to_world_coords(grid_pos, grid_size=(100, 100), units_per_meter=100):
	"""
	Maps a 2D grid coordinate to a 3D world coordinate on the XY plane,
	centered at (0,0).
	"""
	world_width = grid_size[0] / units_per_meter
	world_height = grid_size[1] / units_per_meter

	world_x_range = [-world_width / 2, world_width / 2]
	world_y_range = [-world_height / 2, world_height / 2]

	norm_x = grid_pos[0] / grid_size[0]
	norm_y = grid_pos[1] / grid_size[1]
	world_x = world_x_range[0] + norm_x * (world_x_range[1] - world_x_range[0])
	world_y = world_y_range[0] + norm_y * (world_y_range[1] - world_y_range[0])
	return world_x, world_y

def world_to_grid_coords(world_pos, grid_size=(100, 100), units_per_meter=100):
	"""
	Maps a 3D world coordinate on the XY plane to a 2D grid coordinate.
	"""
	world_width = grid_size[0] / units_per_meter
	world_height = grid_size[1] / units_per_meter

	world_x_range = [-world_width / 2, world_width / 2]
	world_y_range = [-world_height / 2, world_height / 2]

	world_x, world_y = world_pos[0], world_pos[1]

	# De-normalize the world coordinates to a [0, 1] range
	norm_x = (world_x - world_x_range[0]) / (world_x_range[1] - world_x_range[0])
	norm_y = (world_y - world_y_range[0]) / (world_y_range[1] - world_y_range[0])

	# Scale the normalized coordinates to the grid size
	grid_x = norm_x * grid_size[0]
	grid_y = norm_y * grid_size[1]

	# Clamp the values to be within the grid boundaries and convert to integer
	grid_x = max(0, min(grid_size[0] - 1, int(grid_x)))
	grid_y = max(0, min(grid_size[1] - 1, int(grid_y)))

	return grid_x, grid_y


def get_available_body_types(objects_dir, objects):
	available = {}
	for obj in objects:
		model_name = obj['model_name']
		if model_name not in available:
			available[model_name] = len(os.listdir(f"{objects_dir}/{model_name}"))
	return available

def choose_least_used_body_type(model_name, available_body_types, counter):
	count = counter[model_name]
	num_types = available_body_types[model_name]
	min_used = min(count.get(bt, 0) for bt in range(1, num_types + 1))
	candidates = [bt for bt in range(1, num_types + 1) if count.get(bt, 0) == min_used]
	chosen = random.choice(candidates)
	counter[model_name][chosen] += 1
	return chosen

def generate_scene_objects_from_meta(objects_dir: str, scene_meta, z, grid_size, target_mode=False):
	"""
	Generates scene objects from metadata, assigning new body types for visual diversity.
	This is used when creating a new scene arrangement for the first time.
	"""
	objects = []
	for i, obj in enumerate(scene_meta['objects']):
		label = int(obj['label'])
		model_name = OBJECTS[label]['name']
		if target_mode:
			grid_pos = obj['target_pos']
			base_id = obj['target_base_id']
		else:
			grid_pos = obj['initial_pos']
			base_id = obj['initial_base_id']

		objects.append({
			'object_id': i,
			'label': label,
			'model_name': model_name,
			'pos': grid_to_world_coords(grid_pos, grid_size),
			'base_id': base_id,
		})

	# Create a lookup map for objects by their ID
	objects_by_id = {obj['object_id']: obj for obj in objects}

	# Calculate stack hierarchy for each object
	for obj in objects:
		stack_hierarchy = 0
		j = obj['base_id']
		while j is not None:
			stack_hierarchy += 1
			# Guard against malformed hierarchies
			if j in objects_by_id:
				j = objects_by_id[j]['base_id']
			else:
				j = None # Break loop if base_id is not found
		obj['stack_hierarchy'] = stack_hierarchy

	# Sort objects by stack hierarchy to load them bottom-up
	objects.sort(key=lambda x: x['stack_hierarchy'])

	available_body_types = get_available_body_types(objects_dir, objects)
	body_type_counter = defaultdict(Counter)

	for obj in objects:
		model_name = obj['model_name']
		pos, orn = adjust_object_pose_for_stacking(obj, objects, z)

		# Store the final computed pose back into the object dictionary
		obj['final_pos'] = pos
		obj['final_orn'] = orn

		chosen_type = choose_least_used_body_type(model_name, available_body_types, body_type_counter)

		body_id, _ = load_object_urdf(objects_dir, model_name, pos, orn, body_type=chosen_type)
		obj['body_id'] = body_id
		obj['body_type'] = chosen_type

	# Re-sort by object_id to maintain original order
	objects.sort(key=lambda x: x['object_id'])

	return objects

def load_scene_objects_from_labels(objects_dir: str, rearrangement_path: str):
	"""
	Loads scene objects for an existing rearrangement problem from saved label files.
	This reconstructs a scene that has already been generated and saved.
	"""
	initial_label_path = os.path.join(rearrangement_path, 'initial_labels.json')
	target_label_path = os.path.join(rearrangement_path, 'target_labels.json')
	
	if not os.path.exists(initial_label_path):
		raise FileNotFoundError(f"Initial labels file not found: {initial_label_path}")
	
	if not os.path.exists(target_label_path):
		raise FileNotFoundError(f"Target labels file not found: {target_label_path}")
	
	# Load initial labels
	with open(initial_label_path, 'r') as f:
		initial_labels = json.load(f)
	
	# Load target labels
	with open(target_label_path, 'r') as f:
		target_labels = json.load(f)
	
	# Create dictionary indexed by obj_id for target labels
	if isinstance(target_labels, list):
		target_object_labels = target_labels
	else:
		target_object_labels = target_labels['objects']

	target_dict = {label['obj_id']: label for label in target_object_labels}

	objects = []
	# Process each object from initial labels
	if isinstance(initial_labels, list):
		initial_object_labels = initial_labels
	else:
		initial_object_labels = initial_labels['objects']

	for initial_obj in initial_object_labels:
		obj_id = initial_obj['obj_id']
		model_name = initial_obj['model_name']
		model_id = initial_obj['model_id']
		body_type = int(model_id.split('_')[-1])

		# Extract initial 6D pose
		initial_pose_6d = initial_obj['6D_pose']
		initial_pos = initial_pose_6d[:3]  # x, y, z position
		initial_orn = initial_pose_6d[3:]  # roll, pitch, yaw orientation
		
		# Load object with initial pose
		body_id, _ = load_object_urdf(objects_dir, model_name, initial_pos, initial_orn, body_type=body_type)
		
		# Get initial base_id
		initial_base_id = None
		if initial_obj['natural_parent_list']:
			initial_base_id = initial_obj['natural_parent_list'][0]
		
		# Get target information
		target_obj = target_dict.get(obj_id)
		if target_obj is None:
			raise ValueError(f"Object {obj_id} found in initial labels but not in target labels")
		
		# Extract target 6D pose
		target_pose_6d = target_obj['6D_pose']
		target_pos = target_pose_6d[:3]
		target_orn = target_pose_6d[3:]
		
		# Get target base_id
		target_base_id = None
		if target_obj['natural_parent_list']:
			target_base_id = target_obj['natural_parent_list'][0]
		
		# Create complete object dictionary
		objects.append({
			'object_id': obj_id,
			'model_name': model_name,
			'body_id': body_id,
			'body_type': body_type,
			'initial_pos': initial_pos,
			'initial_orn': initial_orn,
			'initial_base_id': initial_base_id,
			'initial_bbox': initial_obj['bbox'],
			'target_pos': target_pos,
			'target_orn': target_orn,
			'target_base_id': target_base_id,
			'target_bbox': target_obj['bbox']
		})

	# Sort by object_id to maintain consistent ordering
	objects.sort(key=lambda x: x['object_id'])
	
	print(f"Loaded {len(objects)} objects with complete initial and target information")
	
	return objects

def adjust_objects_for_scene(objects, scene_meta, z, grid_size, scene):
	"""
	Adjusts existing objects dictionary for target scene configuration.
	Only updates position, base_id, stack_hierarchy, final_pos, and final_orn.
	All other properties (object_id, label, model_name, body_id, body_type) remain the same.
	"""
	assert scene in ["target", "initial"], "Invalid scene type"

	# Create a copy of objects to avoid modifying the original
	t_objects = []
	for obj in objects:
		t_obj = obj.copy()
		t_objects.append(t_obj)
	
	# Update target-specific properties from metadata
	for obj in t_objects:
		object_id = obj['object_id']
		meta_obj = scene_meta['objects'][object_id]
		
		# Update position and base_id from metadata
		if scene == "target":
			grid_pos = meta_obj['target_pos']
			obj['base_id'] = meta_obj['target_base_id']
		else:
			grid_pos = meta_obj['initial_pos']
			obj['base_id'] = meta_obj['initial_base_id']

		obj['pos'] = grid_to_world_coords(grid_pos, grid_size)

	# Create a lookup map for objects by their ID
	objects_by_id = {obj['object_id']: obj for obj in t_objects}
	
	# Calculate stack hierarchy for each object
	for obj in t_objects:
		stack_hierarchy = 0
		j = obj['base_id']
		while j is not None:
			stack_hierarchy += 1
			# Guard against malformed hierarchies
			if j in objects_by_id:
				j = objects_by_id[j]['base_id']
			else:
				j = None  # Break loop if base_id is not found
		obj['stack_hierarchy'] = stack_hierarchy
	
	# Sort objects by stack hierarchy to process them bottom-up
	t_objects.sort(key=lambda x: x['stack_hierarchy'])
	
	# Recalculate final positions and orientations for stacking
	for obj in t_objects:
		pos, orn = adjust_object_pose_for_stacking(obj, t_objects, z)
		obj['final_pos'] = pos
		obj['final_orn'] = orn
		
	# Re-sort by object_id to maintain original order
	t_objects.sort(key=lambda x: x['object_id'])

	# Update PyBullet object positions and orientations
	for obj in t_objects:
		orn_quat = p.getQuaternionFromEuler(obj['final_orn'])
		p.resetBasePositionAndOrientation(obj['body_id'], obj['final_pos'], orn_quat)

	return t_objects
