import os
import shutil
import trimesh
import numpy as np
import pybullet as p
import pybullet_data


def _clean_dir(dir_path, keep_extensions):
	"""
	Helper function to clean a single directory.
	"""
	for item_name in os.listdir(dir_path):
		item_path = os.path.join(dir_path, item_name)
		
		# Remove 'assets' directory if it exists
		if item_name == "assets" and os.path.isdir(item_path):
			try:
				shutil.rmtree(item_path)
			except Exception as e:
				print(f"Error removing directory {item_path}: {e}")
			continue
		
		# Remove files that don't have the allowed extensions
		if os.path.isfile(item_path):
			should_keep = any(item_name.endswith(ext) for ext in keep_extensions)
			if not should_keep:
				try:
					os.remove(item_path)
				except Exception as e:
					print(f"Error deleting {item_path}: {e}")

def clean_object_dirs(objects_dir, keep_extensions=None):
	"""
	Clean unwanted files from all object directories.
	"""
	if keep_extensions is None:
		keep_extensions = ['.glb', '.blend']
	
	for object_name in os.listdir(objects_dir):
		object_dir = os.path.join(objects_dir, object_name)
		
		if not os.path.isdir(object_dir):
			continue
			
		for body_type in os.listdir(object_dir):
			body_path = os.path.join(object_dir, body_type)
			
			if not os.path.isdir(body_path):
				continue
				
			_clean_dir(body_path, keep_extensions)

def decompose_mesh_vhacd(body_path, obj_name):
	"""
	Decompose mesh using V-HACD algorithm for better collision detection.
	"""
	input_path = os.path.join(body_path, f"{obj_name}.obj")
	output_path = os.path.join(body_path, f"{obj_name}_vhacd.obj")
	log_path = os.path.join(body_path, "vhacd_log.txt")

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

def create_urdf_file(body_path, obj_name, mass, use_concave_collision=False):
	"""
	Create URDF file from OBJ mesh files.
	"""
	obj_path = os.path.join(body_path, f"{obj_name}.obj")
	scene = trimesh.load(obj_path, force='scene')
	parts = []

	if len(scene.geometry) == 1:
		parts.append(("link_1", f"{obj_name}.obj"))
	else:
		for idx, (geom_name, geom) in enumerate(scene.geometry.items(), start=1):
			part_filename = f"{obj_name}_{idx}.obj"
			part_path = os.path.join(body_path, part_filename)

			if hasattr(geom.visual, 'material') and geom.visual.material is not None:
				geom.visual.material.name = geom_name
			else:
				geom.visual.material = trimesh.visual.material.SimpleMaterial(name=geom_name)

			obj_text = trimesh.exchange.obj.export_obj(geom, mtl_name='material.mtl')

			lines = obj_text.splitlines()
			for i, line in enumerate(lines):
				if line.startswith('mtllib'):
					lines.insert(i + 1, f'o {obj_name}_{idx}')
					break
			obj_text_with_o = '\n'.join(lines)

			with open(part_path, 'w') as f:
				f.write(obj_text_with_o)

			parts.append((f"link_{idx}", part_filename))

	collision_file = f"{obj_name}_vhacd.obj" if use_concave_collision else f"{obj_name}.obj"
	if use_concave_collision:
		decompose_mesh_vhacd(body_path, obj_name)

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
<robot name="{obj_name}">
{''.join(links_xml)}
{''.join(joints_xml)}
</robot>'''

	urdf_path = os.path.join(body_path, f"{obj_name}.urdf")
	with open(urdf_path, 'w') as f:
		f.write(urdf_content)

def convert_glb_to_urdf(objects_dir, obj_name, mass, use_concave_collision=False):
	"""
	Convert GLB files to URDF format for physics simulation.
	"""
	object_dir = os.path.join(objects_dir, obj_name)

	for body_type in os.listdir(object_dir):
		body_path = os.path.join(object_dir, body_type)
		
		if not os.path.isdir(body_path):
			continue

		# Clean the directory before processing, keeping only GLB and blend files
		_clean_dir(body_path, keep_extensions=['.glb', '.blend'])

		glb_path = os.path.join(body_path, f"{body_type}.glb")
		obj_path = os.path.join(body_path, f"{body_type}.obj")

		if not os.path.exists(glb_path):
			print(f"Warning: GLB file not found: {glb_path}")
			continue

		# Convert GLB to OBJ
		scene = trimesh.load(glb_path, force='scene')
		scene.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
		scene.export(obj_path)

		# Create URDF
		create_urdf_file(body_path, body_type, mass, use_concave_collision)


if __name__ == "__main__":
	import sys
	OBJS_DIR = sys.argv[1]
	if not os.path.exists(OBJS_DIR):
		print(f"Error: Objects directory does not exist: {OBJS_DIR}")
		sys.exit(1)

	object_specs = [
		('table', 5, False),
		('knife', 0.05, False),
		('spoon', 0.05, False),
		('fork', 0.05, False),
		('apple', 0.6, False),
		('banana', 0.7, False),
		('pear', 0.6, False),
		('mug', 0.7, True),
		('paper-cup', 0.5, True),
		('bowl', 0.7, True),
		('box', 0.8, True),
		('basket', 0.8, True),
		('pot', 1, True)
	]

	print(f"Starting GLB to URDF conversion for {len(object_specs)} object types...")
	print(f"Objects directory: {OBJS_DIR}")
	
	print("\n1. Cleaning object directories...")
	clean_object_dirs(OBJS_DIR)
	print("✓ Directory cleanup complete")

	print("\n2. Converting GLB files to URDF format...")
	success_count = 0
	for i, (name, mass, concave) in enumerate(object_specs):
		collision_type = "concave" if concave else "convex"
		print(f"   [{i+1:2d}/{len(object_specs)}] Processing {name:<12} (mass: {mass:>4}, collision: {collision_type})")
		try:
			convert_glb_to_urdf(OBJS_DIR, name, mass, concave)
			success_count += 1
		except Exception as e:
			print(f"      ❌ Failed to convert {name}: {e}")
			continue
	
	print(f"\n✓ Conversions completed: {success_count}/{len(object_specs)} successful")
