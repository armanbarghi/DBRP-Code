import os
import sys
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import pybullet as p
import pybullet_data

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.env.scene_manager import SceneManager, OBJECTS
from core.env.scene_utils import create_scene_meta, save_scene_meta, scene_meta_to_x
from core.sim.physics_utils import (
	PyBulletSim, load_table_urdf, get_object_extents, get_object_type_max_footprints
)
from core.sim.camera_manager import CameraManager
from core.sim.rearrangement_loader import generate_scene_objects_from_meta, adjust_objects_for_scene


def remove_all_objects_except_table(table_body_id):
    """
    Remove all objects in the PyBullet scene except the table and ground plane.
    """
    # Get all body IDs in the simulation
    num_bodies = p.getNumBodies()
    
    # Collect body IDs to remove (excluding table and ground plane)
    bodies_to_remove = []
    for i in range(num_bodies):
        body_id = p.getBodyUniqueId(i)
        # Keep table (given ID) and ground plane (usually ID 0)
        if body_id != table_body_id and body_id != 0:
            bodies_to_remove.append(body_id)
    
    # Remove the collected bodies
    for body_id in bodies_to_remove:
        p.removeBody(body_id)

def validate_scene_visibility(objs_dir, objects, cam, min_vis_ratio=0.3, min_bbox_size=400):
	"""
	Validate if all objects in the scene meet visibility requirements.
	"""
	for obj in objects:
		bbox, vis_ratio = cam.compute_2d_bounding_box(
			objs_dir, obj['model_name'], obj['body_id'], obj['body_type'], num_samples=500
		)
		bbox_size = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) if bbox else 0
		
		if bbox is None or vis_ratio < min_vis_ratio or bbox_size < min_bbox_size:
			# print(f"Object {obj['object_id']} ({obj['model_name']}) failed visibility check: "
			# 	f"VR={vis_ratio:.2f}, BB={bbox_size}")
			return False
	return True

def generate_image_and_label(
		sim, table, objs_dir, folder_path, meta_data, grid_size,
		min_vis_ratio=0.3, min_bbox_size=400,
		initial_objects=None
	):
	"""
	Generate image and label for a scene.
	"""
	if initial_objects:
		image_path = os.path.join(folder_path, f'target_image.png')
		label_path = os.path.join(folder_path, f'target_labels.json')
	else:
		image_path = os.path.join(folder_path, f'initial_image.png')
		label_path = os.path.join(folder_path, f'initial_labels.json')

	z = get_object_extents(table)[2]

	if initial_objects:
		initial_labels_path = os.path.join(folder_path, 'initial_labels.json')
		with open(initial_labels_path, 'r') as f:
			initial_labels = json.load(f)
		
		# Extract camera info from initial labels
		camera_info = initial_labels.get('camera_info')
		if camera_info:
			cam = CameraManager(
				target_pos=camera_info['target_pos'],
				distance=camera_info['distance'],
				yaw=camera_info['yaw'],
				pitch=camera_info['pitch'],
				roll=camera_info['roll'],
				width=camera_info['image_size'][0],
				height=camera_info['image_size'][1],
				fov=camera_info['fov']
			)

		# Adjust objects for target scene
		objects = adjust_objects_for_scene(initial_objects, meta_data, z, grid_size, scene="target")
	else:
		# For initial scene, use random viewpoint
		cam = CameraManager()
		cam.set_viewpoint(target_pos=[0, 0, z])

		camera_info = {
			'viewpoint_id': cam.viewpoint_id,
			'image_size': [cam.width, cam.height],
			'target_pos': cam.target_pos,
			'distance': cam.distance,
			'yaw': cam.yaw,
			'pitch': cam.pitch,
			'roll': cam.roll,
			'fov': cam.fov
		}
		
		# Generate objects for the initial scene
		objects = generate_scene_objects_from_meta(
			objs_dir, meta_data, z, grid_size,
			target_mode=False,
		)

	sim.step(2)

	# Validate scene visibility
	if not validate_scene_visibility(objs_dir, objects, cam, min_vis_ratio, min_bbox_size):
		return False, []

	# If validation passes, create labels and save image
	json_data = {
		'camera_info': camera_info,
		'objects': []
	}

	# Create a map of object_id to its direct parent's object_id
	parent_map = {
		obj['object_id']: obj['base_id'] 
		for obj in objects if obj['base_id'] is not None
	}

	for obj in objects:
		bbox, _ = cam.compute_2d_bounding_box(
			objs_dir, obj['model_name'], obj['body_id'], obj['body_type'], num_samples=500
		)
		
		pos, orn = p.getBasePositionAndOrientation(obj['body_id'])
		euler = p.getEulerFromQuaternion(orn)

		# Find all ancestors for the parent_list
		parent_list = []
		current_id = obj['object_id']
		while current_id in parent_map:
			parent_id = parent_map[current_id]
			parent_list.append(int(parent_id))
			current_id = parent_id

		json_data['objects'].append({
			"model_name": obj['model_name'],
			"model_id": f"{obj['model_name']}_{obj['body_type']}",
			"obj_id": int(obj['object_id']),
			"6D_pose": [float(x) for x in (pos + euler)],
			"natural_parent_list": [int(obj['base_id'])] if obj['base_id'] is not None else [],
			"parent_list": parent_list,
			"bbox": [float(x) for x in bbox],
		})

	with open(label_path, 'w') as f:
		json.dump(json_data, f, indent=4)

	image = cam.capture_image()
	img = Image.fromarray(image)
	img.save(image_path)

	return True, objects

def create_data(objs_dir, dataset_dir, scene_id, env, min_vis_ratio=0.3, min_bbox_size=400, max_attempts=10):
	folder_path = os.path.join(dataset_dir, f'rearrangement_{scene_id:05d}')
	os.makedirs(folder_path, exist_ok=True)
	meta_path = os.path.join(folder_path, 'meta.json')

	if os.path.exists(meta_path):
		# Load existing meta data
		with open(meta_path, 'r') as f:
			meta_data = json.load(f)
		initial_x, target_x = scene_meta_to_x(meta_data)
		
		initial_img_path = os.path.join(folder_path, 'initial_image.png')
		target_img_path = os.path.join(folder_path, 'target_image.png')
		if os.path.exists(initial_img_path) and os.path.exists(target_img_path):
			return
	else:
		# Create new meta data
		env.reset(use_stack=True, use_sides=False)
		initial_x = env.initial_x.clone()
		target_x = env.target_x.clone()
		meta_data = create_scene_meta(initial_x, target_x, scene_id, env.grid_size)

	sim = PyBulletSim(p.GUI)
	table = load_table_urdf(objs_dir, env.grid_size)

	for _ in range(max_attempts):
		# Delete previous images and labels
		for file in os.listdir(folder_path):
			if file.endswith('.png') or (file.endswith('.json') and file != 'meta.json'):
				os.remove(os.path.join(folder_path, file))
	
		# Reset environment with current metadata
		env.reset(initial_x, target_x)
		
		# Remove all the objects in the scene except table
		remove_all_objects_except_table(table)

		# Generate initial scene
		initial_success, objects = generate_image_and_label(
			sim, table, objs_dir, folder_path, meta_data, env.grid_size,
			min_vis_ratio=min_vis_ratio, min_bbox_size=min_bbox_size
		)

		if not initial_success:
			env.reset(use_stack=True, use_sides=False)
			initial_x = env.initial_x.clone()
			target_x = env.target_x.clone()
			meta_data = create_scene_meta(initial_x, target_x, scene_id, env.grid_size)
			continue

		# Generate target scene
		target_success, _ = generate_image_and_label(
			sim, table, objs_dir, folder_path, meta_data, env.grid_size,
			min_vis_ratio=min_vis_ratio, min_bbox_size=min_bbox_size,
			initial_objects=objects
		)

		if target_success:
			# Save metadata only if scene pair was successful
			save_scene_meta(meta_data, meta_path, verbose=0)
			break
		else:
			env.reset(use_stack=True, use_sides=False)
			initial_x = env.initial_x.clone()
			target_x = env.target_x.clone()
			meta_data = create_scene_meta(initial_x, target_x, scene_id, env.grid_size)

	# Remove all the objects in the scene except table before closing
	remove_all_objects_except_table(table)

	sim.close()


if __name__ == "__main__":
	import argparse
	import logging

	logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
	logger = logging.getLogger(__name__)

	parser = argparse.ArgumentParser(description='Create simulation dataset')
	parser.add_argument('objects_dir', help='Path to objects directory')
	parser.add_argument('dataset_dir', help='Path to dataset output directory')
	parser.add_argument('--num_cases', type=int, default=500, help='Number of cases per object count')
	parser.add_argument('--n_values', type=int, nargs='+', default=[5, 6, 7, 8, 9], help='List of object counts')
	parser.add_argument('--grid_size', type=int, nargs=2, default=[100, 100], help='Grid size (width height)')

	args = parser.parse_args()

	OBJS_DIR = args.objects_dir
	DATASET_DIR = args.dataset_dir
	num_cases = args.num_cases
	n_values = args.n_values
	grid_size = tuple(args.grid_size)

	if not os.path.exists(OBJS_DIR):
		raise ValueError(f"Objects directory {OBJS_DIR} does not exist.")

	os.makedirs(DATASET_DIR, exist_ok=True)  # Create if doesn't exist

	try:
		logger.info("Computing object footprint sizes...")
		max_footprints = get_object_type_max_footprints(OBJS_DIR)
		
		logger.info(f"Found footprints for {len(max_footprints)} object types")
		
		objects_processed = 0
		for obj in OBJECTS.values():
			name = obj["name"]
			if name in max_footprints:
				size = np.round(max_footprints[name] * 100)
				if size % 2 == 0:
					size += 1
				obj["size"] = (int(size), int(size))
				logger.info(f"  {name}: {max_footprints[name]:.3f}m → {int(size)}x{int(size)} grid")
				objects_processed += 1

		object_sizes = {k: v["size"] for k, v in OBJECTS.items()}
		logger.info(f"Processed {objects_processed}/{len(OBJECTS)} objects")

		total_scenes = sum(num_cases for _ in n_values)
		logger.info(f"Creating {total_scenes} total scenes")

		scene_id = 1
		for num_objects in n_values:
			logger.info(f"Processing {num_cases} scenes with {num_objects} objects")
			env = SceneManager(num_objects=num_objects, grid_size=grid_size, verbose=0)
			env.object_sizes = object_sizes
			for case_idx in tqdm(range(num_cases), desc=f"Objects: {num_objects}"):
				create_data(OBJS_DIR, DATASET_DIR, scene_id=scene_id, env=env)
				scene_id += 1
	except Exception as e:
		logger.error(f"Failed to create dataset: {e}")
		sys.exit(1)
