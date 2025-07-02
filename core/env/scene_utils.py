import os
import json
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from core.env.scene_manager import Indices, SceneManager, get_object_below, cal_density

def scene_meta_to_x(scene_meta: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Convert JSON scene representation to tensor format"""
	def base_id_to_relation(base_id, num_objects):
		relation = [0] * num_objects
		if base_id is not None:
			relation[base_id] = 1
		return relation

	num_objects = scene_meta['num_objects']

	initial_x = torch.tensor([
		[
			obj['label'], 
			*obj['size'], 
			*obj['initial_pos'], 
			*base_id_to_relation(obj.get('initial_base_id'), num_objects)
		]
		for obj in scene_meta['objects']
	], dtype=torch.long)

	target_x = torch.tensor([
		[
			obj['label'], 
			*obj['size'], 
			*obj['target_pos'], 
			*base_id_to_relation(obj.get('target_base_id'), num_objects)
		]
		for obj in scene_meta['objects']
	], dtype=torch.long)

	return initial_x, target_x

def load_scene_metas(num_objects: int, grid_size: Tuple[int, int], phi: float, 
					scenes_dir: str = 'abstract_scenes/scenes') -> List[Dict[str, Any]]:
	"""Load JSON scenes from directory"""
	scenes = []
	dir_path = f'{scenes_dir}/phi_{phi}/g{grid_size[0]}.{grid_size[1]}/n{num_objects}'

	if not os.path.exists(dir_path):
		return scenes

	for filename in os.listdir(dir_path):
		if not filename.endswith('.json'):
			continue
		
		with open(os.path.join(dir_path, filename), 'r') as f:
			scene = json.load(f)
		
		scenes.append(scene)
	
	# Sort the scenes by id
	scenes.sort(key=lambda x: x['scene_id'])
	
	return scenes

def get_next_scene_id(dataset_dir: str, prefix: str = 'scene') -> int:
	"""Get the next available scene ID"""
	if not os.path.exists(dataset_dir):
		return 0

	files = os.listdir(dataset_dir)
	scene_ids = []
	
	for filename in files:
		if filename.startswith(f'{prefix}_') and filename.endswith('.json'):
			try:
				# Extract number from filename like 'scene_0001.json'
				scene_id = int(filename.split('_')[1].split('.')[0])
				scene_ids.append(scene_id)
			except (ValueError, IndexError):
				continue
	
	return max(scene_ids) + 1 if scene_ids else 0

def create_scene_meta(initial_x: torch.Tensor, target_x: torch.Tensor, 
						scene_id: int, grid_size: Tuple[int, int]) -> Dict[str, Any]:
	"""Create scene metadata dictionary"""
	num_objects = initial_x.shape[0]
	
	objs = []
	for obj in range(num_objects):
		objs.append({
			'object_id': obj,
			'label': initial_x[obj, Indices.LABEL].item(),
			'size': initial_x[obj, Indices.SIZE].tolist(),
			'initial_pos': initial_x[obj, Indices.COORD].tolist(),
			'initial_base_id': get_object_below(initial_x, obj),
			'target_pos': target_x[obj, Indices.COORD].tolist(),
			'target_base_id': get_object_below(target_x, obj), 
		})

	return {
		'scene_id': scene_id,
		'phi': cal_density(initial_x, grid_size),
		'num_objects': num_objects,
		'grid_size': grid_size,
		'objects': objs
	}

def save_scene_meta(scene_meta: Dict[str, Any], output_path: str, verbose: bool = False) -> None:
	"""Save scene metadata to JSON file"""
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	
	with open(output_path, 'w') as f:
		json.dump(scene_meta, f, indent=4)
	
	if verbose:
		print(f'Saved {output_path}')

def load_scenes(num_objects: int, grid_size: Tuple[int, int], phi: float) -> List[Dict[str, Any]]:
	"""Load scenes and convert to tensor format"""
	scenes = []
	scene_metas = load_scene_metas(num_objects, grid_size, phi)
	for scene_meta in scene_metas:
		initial_x, target_x = scene_meta_to_x(scene_meta)
		scenes.append({
			'initial_scene': initial_x,
			'target_scene': target_x
		})
	return scenes


class SceneCreator:
	"""Organize scene creation based on config dictionary."""
	
	def __init__(self, config: Dict[str, Any]):
		self.config = config
	
	def _count_existing_scenes(self, num_objects: int) -> int:
		"""Count existing scenes for given configuration."""
		dir_path = (f'abstract_scenes/scenes/phi_{self.config["phi"]}/'
				f'g{self.config["grid_size"][0]}.{self.config["grid_size"][1]}/'
				f'n{num_objects}')
		
		if not os.path.exists(dir_path):
			return 0
		
		# Count scene_*.json files
		scene_files = [f for f in os.listdir(dir_path) if f.startswith('scene_') and f.endswith('.json')]
		return len(scene_files)
		
	def create_scenes(self) -> Dict[int, List[float]]:
		"""Create and save scenes for all n_values in config."""
		for num_objects in self.config['n_values']:
			existing_count = self._count_existing_scenes(num_objects)
			needed_count = self.config['num_cases'] - existing_count
			
			if self.config.get('verbose', 1) > 0:
				print(f'--n: {num_objects}-- (existing: {existing_count}, need: {max(0, needed_count)})')
			
			if needed_count <= 0:
				if self.config.get('verbose', 1) > 0:
					print(f'  Already have {existing_count} scenes (target: {self.config["num_cases"]})')
				continue
				
			# Create environment
			env = SceneManager(
				num_objects=num_objects, 
				grid_size=self.config['grid_size'], 
				phi=self.config['phi']
			)
			
			# Generate only needed scenes
			scenes = []
			for _ in range(needed_count):
				env.reset(
					use_stack=self.config.get('use_stack', False),
					use_sides=self.config.get('use_sides', False)
				)
				scenes.append({
					'initial_scene': env.initial_x.clone(),
					'target_scene': env.target_x.clone()
				})
			
			# Save new scenes
			self._save_scenes(scenes, num_objects)

	def _save_scenes(self, scenes: List[Dict[str, torch.Tensor]], num_objects: int):
		"""Save scenes to disk."""
		if not scenes:
			return
			
		dir_path = (f'abstract_scenes/scenes/phi_{self.config["phi"]}/'
				f'g{self.config["grid_size"][0]}.{self.config["grid_size"][1]}/'
				f'n{num_objects}')
		
		for scene in scenes:
			scene_id = get_next_scene_id(dir_path)
			scene_meta = create_scene_meta(
				scene['initial_scene'], scene['target_scene'], 
				scene_id, self.config['grid_size']
			)
			output_path = os.path.join(dir_path, f'scene_{scene_id:04d}.json')
			save_scene_meta(scene_meta, output_path, self.config.get('verbose', 1) > 1)
