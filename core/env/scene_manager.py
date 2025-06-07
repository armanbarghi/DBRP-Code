import math
import torch
import random
import numpy as np
import networkx as nx
from typing import Union, List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib.colors import ListedColormap, BoundaryNorm

OBJECTS = {
	0: {"name": "fork", "color": "yellow", "category": "cat1", "size": (19, 19)},
	1: {"name": "spoon", "color": "yellow", "category": "cat1", "size": (19, 19)},
	2: {"name": "knife", "color": "yellow", "category": "cat1", "size": (19, 19)},
	3: {"name": "apple", "color": "orange", "category": "cat2", "size": (9, 9)},
	4: {"name": "pear", "color": "orange", "category": "cat2", "size": (9, 9)},
	5: {"name": "banana", "color": "orange", "category": "cat2", "size": (17, 17)},
	6: {"name": "paper-cup", "color": "green", "category": "cat3", "size": (9, 9)},
	7: {"name": "mug", "color": "green", "category": "cat3", "size": (13, 13)},
	8: {"name": "bowl", "color": "green", "category": "cat3", "size": (13, 13)},
	9: {"name": "basket", "color": "blue", "category": "cat4", "size": (21, 21)},
	10: {"name": "box", "color": "blue", "category": "cat4", "size": (21, 21)},
	11: {"name": "pan", "color": "blue", "category": "cat4", "size": (27, 27)}
}

CATEGORY_STABILITY = {
	'cat1': ['cat3', 'cat4'],
	'cat2': ['cat4'],
	'cat3': ['cat4'],
	'cat4': []  # not stable on anything
}

class Indices:
	LABEL = slice(0, 1)
	SIZE = slice(1, 3)
	COORD = slice(3, 5)
	RELATION = slice(5, None)

## Utility functions for object relations
def get_object_below(x: torch.Tensor, obj: int) -> Optional[int]:
	"""
	Returns the index of the object directly below 'obj' in the stack, or None if it's a base object.
	x[:,RELATION] is one-hot rows: row[i,j]==1 means i sits on j.
	"""
	i = (x[obj, Indices.RELATION] == 1).nonzero(as_tuple=True)[0]
	if len(i):
		return int(i.item())
	return None

def get_object_above(x: torch.Tensor, obj: int) -> Optional[int]:
	"""
	Returns the index of the object directly above 'obj' in the stack, or None if nothing is stacked on it.
	x[:,RELATION] is one-hot rows: row[i,j]==1 means i sits on j.
	"""
	i = (x[:, Indices.RELATION.start + obj] == 1).nonzero(as_tuple=True)[0]
	if len(i):
		return int(i.item())
	return None

def is_stacked(x: torch.Tensor, obj: int) -> bool:
	"""
	Checks if an object is currently stacked on top of another object.
	Returns True if 'obj' has an object below it, False otherwise.
	"""
	return get_object_below(x, obj) is not None

def get_object_base(x: torch.Tensor, obj: int) -> int:
	"""
	Recursively finds the base object of a stack, i.e., the lowest object in the stack.
	Returns the index of the base object.
	"""
	obj_below = get_object_below(x, obj)
	while obj_below is not None:
		obj = obj_below
		obj_below = get_object_below(x, obj)
	return obj

def build_parent_of(x: torch.Tensor) -> torch.Tensor:
	"""
	For each object k, returns the object it sits on (parent), or -1 if none.
	x[:,RELATION] is one-hot rows: row[i,j]==1 means i sits on j.
	In this mapping, `parent_of[i] = j` means object `i` sits on object `j`.
	"""
	N = x.size(0)

	rel  = x[:, Indices.RELATION]            # [N,N]
	rows, cols = rel.nonzero(as_tuple=True)  # rows[k] is the child, cols[k] is the parent

	parent_of = torch.full((N,), -1, dtype=torch.long)
	parent_of[rows] = cols
	return parent_of

def build_child_of(x: torch.Tensor) -> torch.Tensor:
	"""
	For each object k, returns the object that sits on it (child), or -1 if none.
	x[:,RELATION] is one-hot rows: row[i,j]==1 means i sits on j.
	In this mapping, `child_of[j] = i` means object `i` sits on object `j`.
	"""
	N = x.size(0)

	rel  = x[:, Indices.RELATION]            # [N,N]
	rows, cols = rel.nonzero(as_tuple=True)  # cols[k] is the parent, rows[k] is the child

	child_of = torch.full((N,), -1, dtype=torch.long)
	child_of[cols] = rows
	return child_of

## Scene utility functions
def get_patch_slice(coor: List[int], size: List[int], grid_size: Tuple[int, int]):
	"""
	Calculates the slices for a rectangular patch given its center, size, and grid dimensions.
	Args:
		coor: (x, y) center coordinates of the object.
		size: (height, width) of the object.
		grid_size: (H, W) dimensions of the grid.
	Returns:
		A tuple of slices (x_slice, y_slice) that can be used to index a 2D tensor.
	"""
	x, y = coor
	hh, hw = size[0] // 2, size[1] // 2
	x0 = max(x - hh, 0)
	x1 = min(x + hh + 1, grid_size[0])
	y0 = max(y - hw, 0)
	y1 = min(y + hw + 1, grid_size[1])
	return slice(x0, x1), slice(y0, y1)

def generate_random_coordinates(
		grid_size: Tuple[int, int], 
		sizes: torch.Tensor,
		side_assignments: Optional[List[str]]=None, 
		max_attempts: int=100
	) -> Optional[List[torch.Tensor]]:
	"""
	Generate random coordinates for objects ensuring no collisions.

	Args:
		grid_size: Dimensions (height, width) of the grid.
		sizes: Tensor of (height, width) sizes for each object.
		max_attempts: Maximum number of attempts to place all objects.
		side_assignments: List of 'left' or 'right' for each object, influencing placement area.

	Returns:
		List of coordinates for the placed objects if successful, otherwise None.
	"""

	for _ in range(max_attempts):  # Keep trying until all objects are successfully placed
		table = torch.zeros(grid_size, dtype=torch.long)
		H, W = grid_size
		success = True
		coords = []

		for i, size in enumerate(sizes):
			placed = False
			hh, hw = size[0] // 2, size[1] // 2
			for _ in range(max_attempts):
				x_min, x_max = hh, (H-1) - hh

				# Determine bounds based on side assignment
				if side_assignments is not None:
					if side_assignments[i] == 'left':
						# put the object in the left side of the table
						left_bound = ((W-1) // 3) - hw
						y_min, y_max = hw, left_bound
					else:
						# put the object in the right side of the table
						right_bound = ((W-1) * 2) // 3 + hw
						y_min, y_max = right_bound, (W-1) - hw
				else:
					y_min, y_max = hw, (W-1) - hw

				# Generate random coordinates for the object's center
				coor = torch.tensor([
					random.randint(x_min, x_max),
					random.randint(y_min, y_max)
				], dtype=torch.long)

				# Check if the placement is valid
				x_slice, y_slice = get_patch_slice(coor.tolist(), size.tolist(), grid_size)
				if table[x_slice, y_slice].sum() == 0:
					table[x_slice, y_slice] = i + 1
					coords.append(coor)
					placed = True
					break

			if not placed:
				success = False
				break  # Restart placement process

		if success:
			return coords  # Successfully placed all objects

	return None  # Failed to place all objects after max attempts

def create_scene(
		num_objects: int, 
		grid_size: Tuple[int, int], 
		object_sizes: Dict[int, Tuple[int, int]], 
		labels: Optional[List[int]]=None, 
		stack_prob: float=0.6, 
		max_attempts: int=50,
		use_sides: bool=False,
		prev_sides: Optional[List[str]]=None,
		switch_prob: float=0.5,
	) -> torch.LongTensor:
	"""
	Creates a scene configuration (initial or target) with specified number of objects,
	grid size, and object properties, including potential stacking relationships.

	Args:
		num_objects: The total number of objects in the scene.
		grid_size: Dimensions (height, width) of the grid.
		object_sizes: A dictionary mapping object ID to its (height, width) size.
		labels: Optional list of integer labels for each object. If None, random labels are assigned.
		stack_prob: Probability of an object attempting to stack on another.
		max_attempts: Maximum attempts to generate a valid scene.
		use_sides: If True, objects might be assigned to 'left' or 'right' sides of the grid.
		prev_sides: If use_sides is True and prev_sides is provided, objects might switch sides.
		switch_prob: Probability of switching sides if prev_sides is used.

	Returns:
		A torch.LongTensor representing the scene state: [N, D] where D = 5 + N.
		Each row contains: [label, width, height, x_coord, y_coord, one-hot_relation_vector].
	"""
	assert num_objects >= 2

	if max_attempts == 0:
		raise ValueError('Failed to create a scene with the given parameters')

	label_flag = True
	if labels is None:
		label_flag = False
		labels = [random.randint(0, len(object_sizes)-1) for _ in range(num_objects)]
		if np.sum([object_sizes[labels[id]][0]*object_sizes[labels[id]][1] for id in range(num_objects)]) > grid_size[0]*grid_size[1]:
			print('Error: The combined object sizes exceed grid capacity')
			# Recursive call with decremented max_attempts for retrying
			return create_scene(num_objects, grid_size, object_sizes, None, stack_prob, max_attempts-1)
	elif len(labels) != num_objects:
		raise ValueError('Number of labels must be equal to the number of objs')
	else:
		if np.sum([object_sizes[labels[id]][0]*object_sizes[labels[id]][1] for id in range(num_objects)]) > grid_size[0]*grid_size[1]:
			raise ValueError('Man! The objects are too big!')

	# Determine the full dimension D of the object tensor
	# D = 1 (label) + 2 (size) + 2 (coord) + N (relation one-hot) = N + 5
	x_arr = torch.zeros((num_objects, num_objects+4), dtype=torch.long)
	x_arr = torch.cat([torch.tensor(labels, dtype=torch.long).reshape(-1, 1), x_arr], dim=1).to(torch.long)

	# Random label for each obj
	objs = list(range(num_objects))
	for obj in objs:
		x_arr[obj][Indices.LABEL] = labels[obj]
		x_arr[obj][Indices.SIZE] = torch.tensor(object_sizes[labels[obj]], dtype=torch.long)

	random.shuffle(objs)

	# Randomly create edges between objs based on stacking probability
	for obj in objs:
		if random.random() < stack_prob:
			obj1 = obj
			obj2 = obj1
			while obj2 == obj1:
				obj2 = random.randint(0, num_objects-1)

			# check if obj1 can stack on obj2
			label1 = int(x_arr[obj1, Indices.LABEL].item())
			label2 = int(x_arr[obj2, Indices.LABEL].item())
			cat1 = OBJECTS[label1]['category']
			cat2 = OBJECTS[label2]['category']
			if not cat2 in CATEGORY_STABILITY[cat1]:
				continue

			# Avoid stacking multiple objects on top of the same base
			if torch.sum(x_arr[:, Indices.RELATION.start + obj2]) >= 1:
				continue

			x_arr[obj1, Indices.RELATION.start + obj2] = 1

	base_objs = [i for i in range(num_objects) if not is_stacked(x_arr, i)]

	# For target scene: maybe switch side
	side_assignments = None
	if use_sides:
		if prev_sides is None:
			side_assignments_all = [random.choice(['left', 'right']) for _ in range(num_objects)]
		else:
			side_assignments_all = []
			for i in range(num_objects):
				if random.random() < switch_prob:
					side = 'right' if prev_sides[i] == 'left' else 'left'
					if i in base_objs:
						print(f'{i} switched to {side}')
				else:
					side = prev_sides[i]
				side_assignments_all.append(side)
		side_assignments = [side_assignments_all[i] for i in base_objs]

	# Allocate random positions to base objs
	base_sizes = x_arr[base_objs, Indices.SIZE]
	coords = generate_random_coordinates(grid_size, base_sizes, side_assignments=side_assignments, max_attempts=300)
	if coords is None:
		print('Trying again!')
		if label_flag:
			return create_scene(
				num_objects=num_objects,
				grid_size=grid_size, 
				object_sizes=object_sizes,
				labels=labels,
				stack_prob=stack_prob,
				max_attempts=max_attempts-1,
				use_sides=use_sides,
				prev_sides=prev_sides,
				switch_prob=switch_prob,
			)
		else:
			return create_scene(
				num_objects=num_objects,
				grid_size=grid_size, 
				object_sizes=object_sizes,
				labels=None,
				stack_prob=stack_prob,
				max_attempts=max_attempts-1,
				use_sides=use_sides,
				prev_sides=prev_sides,
				switch_prob=switch_prob,
			)
	
	unallocated_objs = list(range(num_objects))
	for i in base_objs:
		x_arr[i][Indices.COORD] = coords.pop(0).clone()
		unallocated_objs.remove(i)

	while len(unallocated_objs) > 0:
		i = unallocated_objs.pop(0)
		if not is_stacked(x_arr, i):
			raise ValueError('Unknown error in creating scene')
		target_obj = get_object_below(x_arr, i)
		if target_obj in unallocated_objs:
			unallocated_objs.append(i)
			continue
		x_arr[i][Indices.COORD] = x_arr[target_obj, Indices.COORD].clone()

	return x_arr

def plot_scene(
		scene_x: torch.Tensor, 
		grid_size: Tuple[int, int], 
		ax=None, 
		fig_size: float=2.5, 
		title: Optional[str]=None, 
		markers: torch.Tensor=torch.tensor([], dtype=torch.long)
	):
	"""
	Plots the current state of the scene_x on a 2D grid.

	Args:
		scene_x: The scene_x tensor [N, D] containing object information.
		grid_size: Dimensions (height, width) of the grid.
		ax: Matplotlib axes object to plot on. If None, a new figure and axes are created.
		fig_size: Base size for the figure.
		title: Optional title for the plot.
		markers: Optional tensor [M, 2] of (x, y) coordinates to highlight in red.
	"""
	if ax is None:
		fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size * (grid_size[1] / grid_size[0])))

	N = scene_x.shape[0]
	color_by_label = {k + 1: v['color'] for k, v in OBJECTS.items()}
	color_by_label[0] = 'white'  # Background
	color_by_label[max(color_by_label.keys()) + 1] = 'red'  # markers

	all_sizes = scene_x[:, Indices.SIZE]
	all_labels = sorted(color_by_label.keys())
	color_list = [color_by_label[label] for label in all_labels]

	# Initialize display table
	display_table = torch.zeros(grid_size, dtype=torch.long)
	
	# Render objects in dependency order (base objects first)
	unrendered_objs = list(range(N))
	while len(unrendered_objs) > 0:
		i = unrendered_objs.pop(0)
		label_i = int(scene_x[i, Indices.LABEL].item())
		coor_i = scene_x[i, Indices.COORD].tolist()
		h_i, w_i = scene_x[i, Indices.SIZE].tolist()

		if is_stacked(scene_x, i):
			child = get_object_below(scene_x, i)
			if child in unrendered_objs:
				unrendered_objs.append(i)
				continue

			# Apply size reduction rules for stacked objects
			child_label = int(scene_x[child, Indices.LABEL].item())
			if OBJECTS[label_i]['category'] == 'cat1' and OBJECTS[child_label]['category'] == 'cat3':
				h_i, w_i = h_i // 4, w_i // 4
			elif torch.all(all_sizes == all_sizes[0]):
				h_i, w_i = h_i // 2, w_i // 2

		# Draw object footprint
		display_table[get_patch_slice(coor_i, [h_i, w_i], grid_size)] = label_i + 1

	# Add constraint highlights
	if markers.numel() > 0:
		display_table[markers[:,0], markers[:,1]] = list(color_by_label.keys())[-1]

	# Create and display the plot
	cmap = ListedColormap(color_list)
	bounds = np.arange(len(color_list) + 1)
	norm = BoundaryNorm(bounds, cmap.N)
	ax.imshow(display_table.numpy(), cmap=cmap, norm=norm, origin='upper', interpolation='none')
	ax.set_xticks([])
	ax.set_yticks([])

	# Add object ID labels at top-left corner of each object
	for i in range(N):
		label_i = int(scene_x[i, Indices.LABEL].item())
		x_i, y_i = scene_x[i, Indices.COORD].tolist()
		h_i, w_i = scene_x[i, Indices.SIZE].tolist()

		# Apply same size reduction for text positioning
		if is_stacked(scene_x, i):
			child = get_object_below(scene_x, i)
			child_label = int(scene_x[child, Indices.LABEL].item())
			if OBJECTS[label_i]['category'] == 'cat1' and OBJECTS[child_label]['category'] == 'cat3':
				h_i, w_i = h_i // 4, w_i // 4
			elif torch.all(all_sizes == all_sizes[0]):
				h_i, w_i = h_i // 2, w_i // 2

		# Place text at top-left corner of the object's footprint
		ax.text(y_i - w_i // 2, x_i - h_i // 2, str(i),
				ha='center', va='center', color='black', fontweight='bold')

	if title is not None:
		ax.set_title(title)

def cal_density(x: torch.Tensor, grid_size: Tuple[int, int]) -> float:
	"""
	Calculates the density of objects in the scene.
	Args:
		x: The scene tensor [N, D] containing object information.
		grid_size: Dimensions (height, width) of the grid.
	Returns:
		The total area occupied by objects divided by the total grid area.
	"""
	phi = 0
	for i in range(x.shape[0]):
		size_i = x[i, Indices.SIZE].tolist()
		phi += (size_i[0] * size_i[1])

	return phi / (grid_size[0] * grid_size[1])

## State representation functions
def positional_encode(one_hot: torch.Tensor) -> torch.Tensor:
	"""
	Vectorized positional encoding: For each one-hot row, return the index of 1.
	If the row is all-zero (no one-hot match), return -1.
	Input shape: [N, D_relation]
	Output shape: [N, 1]
	"""
	# Find index of max value (argmax), but it's invalid if row is all zeros
	pos = one_hot.argmax(dim=1)                  # [N]
	is_zero = one_hot.sum(dim=1) == 0            # [N] boolean mask
	pos[is_zero] = -1                            # replace with -1
	return pos.view(-1, 1)                       # [N, 1]

def state_to_hashable(state: Dict[str, torch.Tensor]) -> tuple:
	"""
	Convert a structured state dict into a flat, hashable Python tuple.
	Args:
		state: A dictionary containing 'current' scene tensor and 'manipulator' position.
	Returns:
		A tuple of flattened tensor values, suitable for hashing.
	"""
	curr = state['current']

	# Vectorized positional encoding
	curr_rel = positional_encode(curr[:, Indices.RELATION])

	# Concatenate relevant columns from current and target
	# Each shape: [N, ?], final shape: [N, total_features]
	new_state = torch.cat([
		curr[:, Indices.LABEL],       # [N,1]
		curr[:, Indices.SIZE],        # [N,2]
		curr[:, Indices.COORD],       # [N,2]
		curr_rel,                     # [N,1]
	], dim=1)                         # → [N, 6]

	# Flatten to 1D
	flat_state = new_state.flatten()            # shape [N * 6]

	# Convert manipulator to 1D tensor
	manip = state['manipulator'].flatten()      # shape [2]

	# Concatenate and convert once to list
	all_vals = torch.cat([manip, flat_state])   # shape [2 + N*6]
	return tuple(all_vals.tolist())             # final tuple

def copy_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
	"""
	Creates a deep copy of the environment state dictionary.
	Args:
		state: The state dictionary to copy.
	Returns:
		A new state dictionary with cloned tensors.
	"""
	return {
		'current': state['current'].clone(), 
		'table': state['table'].clone(),
		# 'target': state['target'].clone(),
		'manipulator': state['manipulator'].clone(),
	}

## Env utility functions
def show_valid_score_map(
	env: 'SceneManager',
	obj: int,
	selected_positions: torch.Tensor = None
	):
	"""
	Displays three plots:
	1. A boolean mask of valid centers, with selected positions highlighted.
	2. A heatmap of inverted scores across all centers.
	3. A heatmap of inverted scores at only the valid centers.

	Args:
		env: The SceneManager environment instance.
		obj: The ID of the object.
		selected_positions (torch.Tensor, optional): A tensor of shape [N, 2]
			containing the selected (y, x) coordinates to highlight.
	"""
	all_centers = torch.ones(env.grid_size, dtype=torch.float32).nonzero(as_tuple=False)

	# Compute raw scores (lower is better) and invert them (higher is better)
	raw_scores = env.score(all_centers, obj)
	max_score = raw_scores.max()
	inverted_scores = max_score - raw_scores
	score_map = torch.zeros(env.grid_size, dtype=torch.float32)
	score_map[all_centers[:, 0], all_centers[:, 1]] = inverted_scores

	# Get valid mask and create the valid score map
	valid_mask = env.valid_center_mask(obj)
	valid_score_map = torch.zeros(env.grid_size, dtype=torch.float32)
	valid_score_map[valid_mask] = score_map[valid_mask]

	# Determine the shared min and max for consistent color scaling across plots.
	# We use the full score_map as the reference for the color range.
	vmin = score_map.min().item()
	vmax = score_map.max().item()

	# --- Plotting ---
	fig, axs = plt.subplots(1, 3, figsize=(8, 3))

	# Plot 1: Valid Map with highlighted selected positions
	axs[0].imshow(valid_mask.cpu().numpy(), interpolation='none')
	axs[0].set_title('Valid Map')

	# Overlay the selected positions if they are provided
	if selected_positions is not None and selected_positions.numel() > 0:
		# Convert to numpy and handle potential GPU tensor
		pos_np = selected_positions.cpu().numpy()
		# Use scatter to plot points. Note the coordinate order:
		# scatter(x, y) maps to scatter(column, row)
		axs[0].scatter(
			pos_np[:, 1],  # x-coordinates (columns)
			pos_np[:, 0],  # y-coordinates (rows)
			c='crimson',   # A bright color to stand out
			marker='x',    # A distinct marker
			s=50,          # Marker size
			linewidth=1.5
		)
		axs[0].set_title('Valid Map & Selections')

	# Plot 2: Score Map
	axs[1].imshow(
		score_map.cpu().numpy(),
		interpolation='none',
		vmin=vmin,  # Apply shared vmin
		vmax=vmax   # Apply shared vmax
	)
	axs[1].set_title('Full Score Map')

	# Plot 3: Valid Score Map
	im2 = axs[2].imshow(
		valid_score_map.cpu().numpy(),
		interpolation='none',
		vmin=vmin,  # Apply shared vmin
		vmax=vmax   # Apply shared vmax
	)
	axs[2].set_title('Valid Score Map')
	fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

	for ax in axs:
		ax.axis('off')

	fig.suptitle(f"Object {obj} Placement Analysis", fontsize=16)
	plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout for suptitle
	plt.show()

def draw_dependency_graph(env: 'SceneManager', fig_size: Tuple[float, float]=(2.5, 2.5)):
	"""
	Draws a dependency graph based on the current and target scene states.
	Edges indicate objects that must be moved out of the way for another object to reach its target.

	Args:
		env: The SceneManager environment instance.
		fig_size: Size of the matplotlib figure.
	"""
	def node_poses(num_nodes: int):
		"""Generates circular positions for graph nodes."""
		if num_nodes < 1:
			raise ValueError("Number of nodes must be at least 1")
		if num_nodes == 1:
			return {0: [0,0]} # Center for a single node

		radius = 1
		angle_step = 2 * math.pi / num_nodes
		positions = []

		for i in range(num_nodes):
			angle = i * angle_step
			x = radius * math.cos(angle)
			y = radius * math.sin(angle)
			positions.append([x, y])

		return positions

	dependency_graph = nx.DiGraph()
	for k in range(env.N):
		dependency_graph.add_node(k)
		i = get_object_below(env.target_x, k)
		if i is None:
			j = get_object_below(env.current_x, k)
			CK = env.current_x[k, Indices.COORD]
			TK = env.target_x[k, Indices.COORD]
			if torch.equal(CK, TK):
				while j is not None:
					if not dependency_graph.has_edge(k, j):
						dependency_graph.add_edge(k, j)
					j = get_object_below(env.current_x, j)
			else:
				for j in env.find_blocking_objects(k):
					j = get_object_base(env.current_x, j)
					if not dependency_graph.has_edge(k, j):
						dependency_graph.add_edge(k, j)
		else:
			j = get_object_above(env.current_x, i)
			if j is not None and j != k:
				dependency_graph.add_edge(k, j)

	fig, ax = plt.subplots(1, 1, figsize=fig_size)
	nx.draw(dependency_graph, node_poses(env.N), with_labels=True, node_size=400, ax=ax, node_color='skyblue')
	plt.title('Dependency Graph')
	plt.show()

## SceneManager class
class SceneManager:
	"""
	Manages the state and dynamics of a tabletop environment for object manipulation.
	Supports 'move' and 'stack' actions, handles collisions, stability rules,
	and calculates costs associated with actions.
	"""
	def __init__(
			self, mode: str, 
			num_objects: int, grid_size: Tuple[int, int], 
			static_stack: bool=False, terminal_cost: bool=True, 
			phi: Union[str, float]='mix', verbose: int=1
		):
		self.mode = mode
		self.N = num_objects
		self.grid_size = grid_size
		self.static_stack = static_stack # If True, cannot move/stack objects with others on top.
		self.terminal_cost = terminal_cost # If True, adds cost for manipulator return at terminal state.
		self.verbose = verbose
		self.object_sizes = self._get_obj_sizes(phi)

		H, W = self.grid_size
		if self.mode == 'mobile':
			self.manipulator_init_pos = torch.tensor(
				[0, (W - 1) // 2], dtype=torch.long
			)
		elif self.mode == 'stationary':
			self.manipulator_init_pos = torch.tensor(
				[(H - 1) // 2, (W - 1) // 2], dtype=torch.long
			)
		else:
			raise ValueError(f'Invalid mode: {mode}. Must be "mobile" or "stationary".')
		
		self.normalization_factor = 1 / min(self.grid_size)
		self.manipulator = self.manipulator_init_pos.clone() # Current manipulator position
		self.pp_cost = 0.2 			# Per-pickup cost for actions
		self._P = 2*(H + W) - 4 	# Number of indexes in the mobile mode

		# These will be initialized in reset()
		self.current_x: torch.Tensor
		self.target_x: torch.Tensor
		self.initial_x: torch.Tensor
		self.current_table: torch.Tensor
		self.target_table: torch.Tensor
		self.stability_mask: torch.Tensor
		self._i: torch.Tensor # For _build_table
		self._j: torch.Tensor # For _build_table

	def init(self):
		"""
		Initializes internal environment data structures, such as object category mappings,
		stability masks, boundary maps, and initial occupancy grids.
		This is typically called once after the scenes (current_x, target_x) are set.
		"""
		# Map each object → its category index
		categories  = list(CATEGORY_STABILITY.keys())
		cat_to_idx  = {c:i for i,c in enumerate(categories)}
		labels      = self.current_x[:, Indices.LABEL].squeeze().tolist()
		obj_categories = torch.tensor(
			[cat_to_idx[OBJECTS[l]['category']] for l in labels],
			dtype=torch.long,
		)

		# Build C×C category stability matrix
		C = len(categories)
		cat_stab = torch.zeros(C, C, dtype=torch.bool)
		for cat, ok_list in CATEGORY_STABILITY.items():
			i = cat_to_idx[cat]
			for ok_cat in ok_list:
				cat_stab[i, cat_to_idx[ok_cat]] = True

		# Derive N×N object‐to‐object stability mask
		#    stable_mask[i,j] == True iff object i can stably sit on object j
		self.stability_mask = cat_stab[
			obj_categories.view(-1, 1),  # shape [N,1]
			obj_categories.view(1, -1)   # shape [1,N]
		]  # result: [N, N] boolean

		# Build meshgrid indices for _build_table
		H, W = self.grid_size
		self._i = torch.arange(H).view(1,H,1)
		self._j = torch.arange(W).view(1,1,W)

		# Build the initial tables
		self.current_table = self._build_table(self.current_x)
		self.target_table = self._build_table(self.target_x)

		# Initialize manipulator position
		self.manipulator = self.manipulator_init_pos.clone()

	def create_scene(self, labels: Optional[List[int]]=None, use_stack: bool=True, use_sides: bool=False) -> torch.Tensor:
		"""
		Creates a new random scene configuration.

		Args:
			labels: Optional list of integer labels for each object. If None, random labels are assigned.
			use_stack: If True, allows objects to be stacked. If False, only flat placements.
			use_sides: If True, objects might be assigned to 'left' or 'right' sides of the grid.

		Returns:
			A torch.LongTensor representing the newly created scene state.
		"""
		stack_prob = 0.9 if use_stack else 0.0
		return create_scene(
			self.N, self.grid_size, self.object_sizes, labels, 
			stack_prob=stack_prob, 
			use_sides=use_sides, switch_prob=0.5,  # switch_prob is used when prev_sides is provided
		)

	def _get_obj_sizes(self, phi: Union[str, float]) -> Dict[int, Tuple[int, int]]:
		"""
		Determines object sizes based on the desired density (phi).
		If phi is 'mix', uses default object sizes. Otherwise, tries to find
		a uniform object size that results in a scene density close to phi.

		Args:
			phi: 'mix' for default sizes, or a float representing the target density.

		Returns:
			A dictionary mapping object ID to its (height, width) size.
		"""
		object_sizes = {k: v["size"] for k, v in OBJECTS.items()}

		if phi == 'mix':
			if self.verbose:
				print('Using default object sizes')
			return object_sizes

		# Binary search for the largest uniform square size that allows N objects
		# to theoretically fit into the grid.
		left, right = 1, min(self.grid_size)
		max_uniform_dim = 0	# Stores the largest possible odd dimension
		while left <= right:
			mid = (left + right) // 2
			count = (self.grid_size[0] // mid) * (self.grid_size[1] // mid)
			if count >= self.N:
				max_uniform_dim = mid + 1  # it's a valid size, try bigger
				left = mid + 1
			else:
				right = mid - 1
		
		best_phi = 0
		best_sizes = {k: (1, 1) for k in object_sizes}

		for i in range(1, max_uniform_dim, 2):
			current_sizes = {k: (i, i) for k in object_sizes}

			scene = create_scene(self.N, self.grid_size, current_sizes, stack_prob=0.0)
			new_phi = cal_density(scene, self.grid_size)

			if abs(new_phi - phi) > abs(best_phi - phi): # type: ignore
				break

			best_phi = new_phi
			best_sizes = current_sizes

		if self.verbose:
			first_key = next(iter(best_sizes))
			print(f'phi: {best_phi:.3f} | uniform size: {best_sizes[first_key]}')

		return best_sizes

	def reset(self, 
			initial_scene: Optional[torch.Tensor]=None, 
			target_scene: Optional[torch.Tensor]=None, 
			use_stack: bool=True, 
			use_sides: bool=False
		):
		"""
		Resets the environment to an initial state and sets a target state.

		Args:
			initial_scene: Optional pre-defined initial scene tensor. If None, a random scene is generated.
			target_scene: Optional pre-defined target scene tensor. If None, a random target scene is generated.
			use_stack: If True, allows stacking in generated scenes.
			use_sides: If True, allows side-based placement for generated scenes.
		"""
		if initial_scene is None:
			self.current_x = self.create_scene(use_stack=use_stack, use_sides=use_sides)
		else:
			self.current_x = initial_scene.clone()
			for i in range(self.N):
				label_i = self.current_x[i, Indices.LABEL].item()
				self.object_sizes[label_i] = tuple(self.current_x[i, Indices.SIZE].tolist())

		labels = list(self.current_x[:, Indices.LABEL].reshape(-1).numpy())
		if target_scene is None:
			self.target_x = self.create_scene(labels, use_stack=use_stack, use_sides=use_sides)
			while torch.equal(self.current_x, self.target_x):
				self.target_x = self.create_scene(labels, use_stack=use_stack, use_sides=use_sides)
		else:
			self.target_x = target_scene.clone()
			target_labels = list(self.target_x[:, Indices.LABEL].reshape(-1).numpy())
			# check whether the target scene has the same labels as the state scene
			if labels != target_labels:
				raise ValueError('Target scene has different labels than the state scene')
		
		self.initial_x = self.current_x.clone()

		if use_stack is False:
			if torch.sum(self.initial_x[:, Indices.RELATION]) > 0:
				raise ValueError('Initial scene has stacks in Non-stack mode')
			if torch.sum(self.target_x[:, Indices.RELATION]) > 0:
				raise ValueError('Target scene has stacks in Non-stack mode')

		self.init()

	def render(self, fig_size: float=4.4, show_manipulator: bool=False):
		"""
		Renders the current and target scenes side-by-side.
		- show_manipulator: if True, overlay the manipulator’s footprint in red on the current scene.
		"""
		current = self.current_x
		target = self.target_x
		manip = self.manipulator.tolist()

		# print manipulator pos
		if self.verbose > 0:
			print(f'Manipulator at {manip}')

		# Build manipulator markers
		markers = torch.tensor([], dtype=torch.long)
		if show_manipulator:
			h, w = (5, 5)
			hh, hw = h//2, w//2

			xr = torch.arange(manip[0]-hh, manip[0]+hh+1)
			yr = torch.arange(manip[1]-hw, manip[1]+hw+1)
			xg, yg = torch.meshgrid(xr, yr, indexing='ij')
			pts = torch.stack((xg.flatten(), yg.flatten()), dim=1)

			# clip to bounds
			H, W = self.grid_size
			mask = (
				(pts[:,0]>=0) & (pts[:,0]<H) & (pts[:,1]>=0) & (pts[:,1]<W)
			)
			markers = pts[mask]

		# Draw side-by-side
		scale = max(self.grid_size)/min(self.grid_size)
		fig, (ax1, ax2) = plt.subplots(
			1, 2,
			figsize=(fig_size*2*scale, fig_size)
		)
		plot_scene(current, self.grid_size, ax1, markers=markers)
		plot_scene(target, self.grid_size, ax2)
		plt.tight_layout()
		plt.show()

	## --occupancy table--
	def _build_table(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Builds a 2D occupancy grid (table) from the object state tensor.
		Only base objects (those not stacked on anything) are marked on the table.

		Args:
			x: [N, D] tensor of obj features (LABEL, SIZE, COORD, RELATION…).

		Returns:
			[H, W] occupancy table with 0=empty, (i+1)=object i.
		"""
		# Identify base objects
		rel          = x[:, Indices.RELATION]         # [N, N]
		stacked_mask = rel.any(dim=1)                 # [N]
		base_idx     = (~stacked_mask).nonzero()[:,0]  # [Nb]

		# Gather centers & sizes for bases
		coords     = x[base_idx, Indices.COORD]  # [Nb,2]
		sizes      = x[base_idx, Indices.SIZE]   # [Nb,2]
		half_sizes = sizes // 2                         # [Nb,2]

		# Initialize the background table
		table = torch.zeros(self.grid_size, dtype=torch.uint8)

		# Broadcast mask build
		cx = coords[:,0].view(-1,1,1)
		cy = coords[:,1].view(-1,1,1)
		hh = half_sizes[:,0].view(-1,1,1)
		hw = half_sizes[:,1].view(-1,1,1)

		in_x = (self._i >= (cx - hh)) & (self._i <= (cx + hh))
		in_y = (self._j >= (cy - hw)) & (self._j <= (cy + hw))
		mask = in_x & in_y  # [Nb, H, W]

		ids       = base_idx + 1                         # [Nb]
		obj_vals  = ids.view(-1,1,1).expand_as(mask)
		placed    = torch.where(mask, obj_vals, torch.zeros_like(obj_vals))
		table, _  = placed.max(dim=0)                    # collapse Nb → [H,W]

		return table

	def _erase_from_table(self, obj: int):
		"""
		Sets the footprint of 'obj' in the current occupancy table to 0 (empty).
		Args:
			obj: The index of the object whose footprint is to be erased.
		"""
		coor = self.current_x[obj, Indices.COORD].tolist()
		size = self.current_x[obj, Indices.SIZE].tolist()
		self.current_table[get_patch_slice(coor, size, self.grid_size)] = 0

	def _draw_to_table(self, obj: int):
		"""
		Fills the footprint of 'obj' in the current occupancy table with (obj + 1).
		Args:
			obj: The index of the object whose footprint is to be drawn.
		"""
		coor = self.current_x[obj, Indices.COORD].tolist()
		size = self.current_x[obj, Indices.SIZE].tolist()
		self.current_table[get_patch_slice(coor, size, self.grid_size)] = obj + 1

	def decode_action(self, action: int) -> Tuple[str, int, int, torch.Tensor]:
		"""
		Decodes a single integer action into its type, involved objects, and coordinates.

		Args:
			action: A single integer representing the encoded action.

		Returns:
			A tuple: (action_type, start_obj, target_obj, coord)
			- action_type: 'stack' or 'move'
			- start_obj: The index of the object to be moved or stacked.
			- target_obj: The index of the target object for 'stack' actions (meaningless for 'move').
			- coord: The (x,y) destination coordinate for 'move' actions (meaningless for 'stack').
		"""
		H, W = self.grid_size
		m = H * W
		stack_offset = self.N * (self.N - 1)

		if action < stack_offset:
			action_type = 'stack'
			start_obj = action // (self.N - 1)
			target_obj = action % (self.N - 1)
			# repair the “skip self” offset
			if target_obj >= start_obj:
				target_obj += 1
			coord = torch.tensor([0, 0], dtype=torch.long)
		else:
			action_type = 'move'
			adjusted = action - stack_offset
			start_obj = adjusted // m
			flat = adjusted % m
			target_obj = start_obj

			# invert flatten:  row = flat // W, col = flat % W
			row = flat // W
			col = flat % W
			coord = torch.tensor([row, col], dtype=torch.long)
			
		return action_type, start_obj, target_obj, coord

	## --aciton encode/decode--
	def encode_move(
			self,
			start_obj: Union[int, List[int], torch.Tensor],
			coords: torch.Tensor
		) -> Union[int, torch.Tensor]:
		"""
		Encodes a move action into a single integer or a batch of integers.
		Args:
			start_obj: The ID(s) of the object(s) to move. Can be an int, list of int, or torch.Tensor.
			coords: The (x,y) destination coordinate(s). Can be a [2] or [B,2] torch.Tensor.
		Returns:
			An integer action code if start_obj is int, or a torch.Tensor of action codes otherwise.
		"""
		H, W     = self.grid_size
		stack_offset = self.N * (self.N - 1)
		m        = H * W

		# Build batched starts & targets of shape [B]
		if isinstance(start_obj, int):
			start = torch.full((1,), start_obj, dtype=torch.long)
		else:
			start = torch.as_tensor(start_obj, dtype=torch.long).view(-1)

		# Flatten coords
		coords = coords.long()
		if coords.ndim == 1 and coords.size(0) == 2:
			# single‐point → make it a [1,2] batch
			coords = coords.view(1, 2)
		if coords.ndim == 2 and coords.size(1) == 2:
			# general batch
			flat = coords[:, 0] * W + coords[:, 1]  # [B]
		else:
			raise ValueError("coords must be shape [2] or [B,2]")

		codes = stack_offset + start * m + flat     # [B]

		if isinstance(start_obj, int):
			return codes.item()
		return codes

	def encode_stack(
			self,
			start_obj: Union[int, List[int], torch.Tensor],
			target_obj: Union[int, List[int], torch.Tensor],
		) -> Union[int, torch.Tensor]:
		"""
		Encodes a stack action into a single integer or a batch of integers.
		Args:
			start_obj: The ID(s) of the object(s) to stack. Can be an int, list of int, or torch.Tensor.
			target_obj: The ID(s) of the object(s) to stack on. Can be an int, list of int, or torch.Tensor.
		Returns:
			An integer action code if start_obj is int, or a torch.Tensor of action codes otherwise.
		"""
		# Build batched starts & targets of shape [B]
		if isinstance(start_obj, int):
			start = torch.full((1,), start_obj, dtype=torch.long)
			B = 1
		else:
			start = torch.as_tensor(start_obj, dtype=torch.long).view(-1)
			B = start.size(0)

		if isinstance(target_obj, int):
			target = torch.full((B,), target_obj, dtype=torch.long)
		else:
			target = torch.as_tensor(target_obj, dtype=torch.long).view(-1)
			if target.size(0) != B:
				raise ValueError("start_obj and target_obj batch sizes disagree")

		# adjust target index down by 1 when target>=start
		adj   = (target >= start).long()      # [B]
		t_adj = target - adj                  # [B]
		codes = start * (self.N - 1) + t_adj       # [B]

		if isinstance(start_obj, int):
			return int(codes.item())
		return codes

	def encode_action(
			self,
			action_type: str,
			start_obj: Union[int, List[int], torch.Tensor],
			target_obj: Union[int, List[int], torch.Tensor],
			coords: torch.Tensor,
		) -> Union[int, torch.Tensor]:
		"""
		Batch‐unified action encoder. Internally everything is a batch of size B.

		Args:
			action_type: "move" or "stack".
			start_obj: The ID(s) of the object(s) involved in the action. Can be int or sequence.
			target_obj: The ID(s) of the target object(s) for 'stack' actions (ignored for 'move').
						Can be int or sequence.
			coords: The (x,y) destination coordinate(s) for 'move' actions (ignored for 'stack').
					Tensor of shape [2] or [B,2].
		Returns:
			An integer action code or a torch.Tensor of action codes.
		"""
		
		# Compute action codes as a tensor of shape [B]
		if action_type == 'stack':
			codes = self.encode_stack(start_obj, target_obj)
		elif action_type == 'move':
			codes = self.encode_move(start_obj, coords)
		else:
			raise ValueError(f"Invalid action_type: {action_type!r}")

		return codes

	## --state management--
	def get_state(self) -> Dict[str, torch.Tensor]:
		"""
		Returns a dictionary representing the current state of the environment.
		"""
		return {
			'current': self.current_x.clone(), 
			'table': self.current_table.clone(),
			# 'target': self.target_x.clone(),
			'manipulator': self.manipulator.clone(),
		}

	def set_state(self, state: Dict[str, torch.Tensor]):
		"""
		Sets the environment's state from a given state dictionary.
		Args:
			state: A dictionary containing 'current' scene tensor, 'table' occupancy grid,
				and 'manipulator' position.
		"""
		self.current_x = state['current']
		self.current_table = state['table']
		# self.target_x = state['target']
		self.manipulator = state['manipulator']

	## --occupancy and placement checks--
	def valid_center_mask(self, obj: int) -> torch.Tensor:
		"""
		Returns a BoolTensor mask of shape (H, W) where True at (x, y)
		indicates that placing `obj`’s center at (x, y) keeps its h×w
		footprint in-bounds and free of any other object.
		This implementation leverages a 2D prefix-sum (integral image)
		so that we can test every h×w window in O(H·W) time.
		Args:
			obj: The index of the object to be placed.
		Returns:
			A BoolTensor mask of shape (H, W).
		"""
		H, W = self.grid_size
		h, w = self.current_x[obj, Indices.SIZE]
		hh, hw = h//2, w//2

		# Build a boolean map of all “other-object” occupancy.
		# 1.0 where the grid is occupied by someone ≠ obj, else 0.0.
		occ = self.current_table
		obj_id = obj + 1
		occ_others = ((occ != 0) & (occ != obj_id)).to(torch.float32)

		# Compute integral image (prefix‐sum) with a zero row/col pad so that
		#   S[i+1, j+1] = sum of occ_others[:i+1, :j+1]
		S = occ_others.cumsum(dim=0).cumsum(dim=1)
		S = F.pad(S, (1, 0, 1, 0), value=0.0)

		# Extract submatrices for the four corners of each potential window:
		H2 = H - h + 1
		W2 = W - w + 1
		br = S[h:   h+H2, w:   w+W2]  # bottom-right corners
		tr = S[0:   H2,   w:   w+W2]  # top-right
		bl = S[h:   h+H2, 0:   W2  ]  # bottom-left
		tl = S[0:   H2,   0:   W2  ]  # top-left

		# Window-sum = br - tr - bl + tl
		# If that sum is zero, the window is entirely free.
		counts = br - tr - bl + tl    # [H-h+1, W-w+1]

		# Collect all top-left coordinates whose window-sum == 0.
		free_tl = torch.nonzero(counts == 0, as_tuple=False)
		if free_tl.numel() == 0:
			# No valid placements: return an all-False mask.
			return torch.zeros((H, W), dtype=torch.bool)

		# Convert top-left coords to center coords by adding half-sizes.
		centers = free_tl + torch.tensor([hh, hw], dtype=torch.long)

		# Scatter into a full (H, W) boolean mask.
		mask = torch.zeros((H, W), dtype=torch.bool)
		mask[centers[:,0], centers[:,1]] = True

		# Exclude the object’s current center (we typically don’t want no-op moves).
		cx, cy = self.current_x[obj, Indices.COORD].tolist()
		mask[cx, cy] = False

		return mask

	def is_invalid_center(self, center: torch.Tensor, obj: int) -> bool:
		"""
		Checks if a single center coordinate (x, y) for object `obj` is invalid.
		A placement is invalid if it's out-of-bounds or collides with another object.

		Args:
			center: The (x, y) coordinates of the proposed center.
			obj: The index of the object to be placed.

		Returns:
			True if placement is out-of-bounds or collides, False otherwise.
		"""
		H, W = self.grid_size
		coor = center.tolist()
		size = self.current_x[obj, Indices.SIZE].tolist()
		hh, hw = size[0] // 2, size[1] // 2

		# Out-of-bounds check
		if not (hh <= coor[0] < H-hh and hw <= coor[1] < W-hw):
			return True

		# Slice the occupancy map (excluding this object)
		obj_id = obj + 1
		footprint = self.current_table[get_patch_slice(coor, size, self.grid_size)]
		occupied = ((footprint != 0) & (footprint != obj_id)).any()

		return bool(occupied.item())

	## --mobile manipulator--
	def get_boundary_index(self, boundary_coord: torch.Tensor) -> int:
		"""
		Given a boundary (b_x,b_y) coordinate, return its index in [0..P).
		Assumes coord is exactly on the boundary.
		Args:
			coord: A torch.Tensor of shape [2] representing (b_x, b_y) coordinates.
		Returns:
			The 1D index of the coordinate on the boundary.
		"""
		H, W = self.grid_size
		b_x, b_y = boundary_coord.tolist()

		# top row
		if b_x == 0 and b_y < W-1:
			return b_y
		# right col
		if b_y == W-1 and b_x < H-1:
			return W + b_x - 1
		# bottom row
		if b_x == H-1:
			return W + (H - 1) + (W - 1 - b_y)
		# left col
		return self._P - b_x + 1

	def cal_manipulator_cost(self, coord_to: torch.Tensor) -> float:
		"""
		Calculates the cost of moving the manipulator from its current position
		(self.manipulator) to `coord_to`, and updates self.manipulator.

		- In 'stationary' mode: Euclidean distance, manipulator moves directly to coord_to.
		- In 'mobile' mode: Projects coord_to to its nearest boundary point,
		moves the manipulator along the boundary (ring distance), and sets
		manipulator to that boundary point.

		Args:
			coord_to: The target (x,y) coordinate for the manipulator.

		Returns:
			The cost of the manipulator movement.
		"""
		H, W = self.grid_size

		if self.mode == 'stationary':
			# direct Euclidean
			dist = torch.norm((self.manipulator - coord_to).float()).item()
			self.manipulator = coord_to.clone()
			return dist * self.normalization_factor

		# --- mobile mode ---
		x, y = coord_to

		# Map each edge to its boundary point and distance
		edge_projections = {
			'top':    (torch.tensor([0,   y], dtype=torch.long), x    ),
			'right':  (torch.tensor([x, W-1], dtype=torch.long), W-1-y),
			'bottom': (torch.tensor([H-1, y], dtype=torch.long), H-1-x),
			'left':   (torch.tensor([x,   0], dtype=torch.long), y    )
		}

		# Find the closest edge
		closest_edge = min(edge_projections.keys(), key=lambda edge: edge_projections[edge][1])
		best_pt, _ = edge_projections[closest_edge]

		# Calculate ring distance and update manipulator
		b_from = self.get_boundary_index(self.manipulator)
		b_to   = self.get_boundary_index(best_pt)

		d = abs(b_from - b_to)
		dist = float(min(d, self._P - d))

		self.manipulator = best_pt
		return dist * self.normalization_factor

	## --manipulation functions--
	def move_object_chain(self, obj: int, new_center: torch.Tensor):
		"""
		Moves `obj` (and any objects stacked on it, recursively) to `new_center`.
		This is a direct update of the object's coordinates in the internal state.
		Args:
			obj: The index of the object to move.
			new_center: The new (x, y) center coordinate for the object.
		"""
		# Build a parent-of map for the current state
		child_of = build_child_of(self.current_x)  # [N]

		# Walk the chain from `obj` up to the topmost object
		desc_mask = torch.zeros(self.N, dtype=torch.bool)
		cur = obj
		while cur >= 0:
			desc_mask[cur] = True
			cur = int(child_of[cur])   # moves to the one object on top, or -1 stops

		# Bulk‐assign new center to every object in the chain
		self.current_x[desc_mask, Indices.COORD] = new_center.clone().view(1,2)

	def move_func(self, start_obj: int, coord: torch.Tensor) -> float:
		"""
		Executes a 'move' action for `start_obj` to `coord`.
		Calculates the cost of the move, updates the scene state and manipulator position.

		Args:
			start_obj: The index of the object to move.
			coord: The target (x,y) coordinate for the object's center.

		Returns:
			The cost of the move action.
		"""
		if self.is_invalid_center(coord, start_obj):
			raise ValueError(f'Invalid placement: position {coord.numpy()} is occupied or out of bounds')

		prev_below = get_object_below(self.current_x, start_obj)
		if prev_below is None: 
			# start_obj is a base object, so remove its footprint before moving
			self._erase_from_table(start_obj)
		else:
			# Remove any old stacking-relation from the object below
			self.current_x[start_obj][Indices.RELATION.start+prev_below] = 0

		# -- Calculate manipulator movement costs --
		cost = 0.0

		# Cost of moving manipulator to start_obj
		prev_coord = self.current_x[start_obj, Indices.COORD]
		cost += self.cal_manipulator_cost(prev_coord)

		# Cost of moving manipulator from prev_coord to coord
		self.move_object_chain(start_obj, coord)
		cost += self.cal_manipulator_cost(coord)

		# Redraw the object's footprint at its new position
		self._draw_to_table(start_obj)

		return cost

	def stack_func(self, start_obj: int, target_obj: int) -> float:
		"""
		Executes a 'stack' action: `start_obj` is stacked on `target_obj`.
		Calculates the cost of the stack, updates the scene state and manipulator position.

		Args:
			start_obj: The index of the object to stack.
			target_obj: The index of the object to stack upon.

		Returns:
			The cost of the stack action.
		"""
		# Stability check
		if not self.stability_mask[start_obj, target_obj]:
			raise ValueError(f'not stable {start_obj} -> {target_obj}')

		# Target‐empty check (no one currently sits on target_obj)
		rel = self.current_x[:, Indices.RELATION]
		if rel[:, target_obj].any():
			raise ValueError(f'obj {target_obj} is not empty')

		prev_below = get_object_below(self.current_x, start_obj)
		if prev_below is None:
			# start_obj is a base object, so remove its footprint before moving
			self._erase_from_table(start_obj)
		else:
			# Remove any old stacking-relation from the object below
			self.current_x[start_obj][Indices.RELATION.start+prev_below] = 0

		# -- Calculate manipulator movement costs --
		cost = 0.0

		# Cost of moving manipulator to start_obj
		prev_coord = self.current_x[start_obj, Indices.COORD]
		cost += self.cal_manipulator_cost(prev_coord)

		# Stack start_obj on top of target_obj by updating relation
		self.current_x[start_obj][Indices.RELATION.start+target_obj] = 1

		# Cost of moving manipulator from prev_coord to destination
		destination  = self.current_x[target_obj, Indices.COORD]
		self.move_object_chain(start_obj, destination)
		cost += self.cal_manipulator_cost(destination)

		return cost

	## --empty positions and objects--
	def get_empty_objs(self, ref_obj: int, n: int=1) -> List[int]:
		"""
		Returns a list of up to `n` object indices that can serve as valid
		bases for stacking `ref_obj`. Objects must be "empty" (nothing stacked on them)
		and stable enough to support `ref_obj`.

		Args:
			ref_obj: The index of the object that intends to stack.
			n: The maximum number of empty objects to return.

		Returns:
			A list of integer object indices.
		"""
		# Which objects are empty and stable?
		rel       = self.current_x[:, Indices.RELATION]      # [N, N] one-hot
		empty_j   = ~rel.any(dim=0)             # [N] bool
		stable_j = self.stability_mask[ref_obj]  # [N] bool
		valid_j = empty_j & stable_j            # [N] bool

		# Extract their indices
		candidates = torch.nonzero(valid_j, as_tuple=False).view(-1)  # [K]

		if candidates.numel() == 0:
			return []

		# Always shuffle, then take up to n
		perm       = torch.randperm(candidates.size(0))
		shuffled   = candidates[perm]
		picked     = shuffled[:n]

		return picked.tolist()

	def get_empty_positions(self, ref_obj: int, n: int=1, sort: bool=False) -> torch.LongTensor:
		"""
		Returns up to `n` valid center positions (x, y) for placing `ref_obj`.
		If sort=False, it returns a random shuffle of valid positions.
		If sort=True, it performs a weighted random sample where positions
		with better (lower) scores have a higher probability of being chosen.

		Args:
			ref_obj: The index of the object for which to find empty positions.
			n: The maximum number of positions to return.
			sort: If True, performs weighted sampling based on score.

		Returns:
			A torch.LongTensor of shape [K, 2] where K <= n.
		"""
		# Get the valid placement mask
		mask = self.valid_center_mask(ref_obj)  # [H, W] BoolTensor

		# Extract coordinates of all True positions
		valid_coords = mask.nonzero(as_tuple=False)  # [K, 2]

		if valid_coords.size(0) == 0:
			return torch.LongTensor([])

		# If we need fewer positions than available, proceed with selection
		if valid_coords.size(0) > n:
			if not sort:
				# Random shuffle for non-sort mode
				perm = torch.randperm(valid_coords.size(0))
				chosen_indices = perm[:n]
				chosen = valid_coords[chosen_indices]
			else:
				# Calculate scores (lower is better)
				sc = self.score(valid_coords, ref_obj)  # [K] composite scores

				# Convert scores to weights (higher is better)
				# We invert the scores so that low scores get high weights.
				# `max(sc) - sc` makes the best score (lowest) have the highest weight.
				# We add a small epsilon for numerical stability if all scores are identical.
				weights = (torch.max(sc) - sc) + 1e-6

				# Sample `n` indices using the weights.
				# `replacement=False` ensures we don't pick the same position twice.
				sampled_indices = torch.multinomial(weights, num_samples=n, replacement=False)

				# Select the coordinates based on the sampled indices
				chosen = valid_coords[sampled_indices]
		else:
			# If n is larger or equal to all available positions, return them all.
			# Optionally shuffle them if not in sort mode.
			if not sort:
				perm = torch.randperm(valid_coords.size(0))
				valid_coords = valid_coords[perm]
			chosen = valid_coords

		return chosen

	def get_empty_positions_with_target(self, ref_obj: int, n: int = 1, sort: bool = False) -> torch.LongTensor:
		"""
		Returns up to n valid flattened positions, preferring the target center
		if it is free, else random/ordered empties.

		Args:
			ref_obj: The index of the object for which to find empty positions.
			n: The maximum number of positions to return.
			sort: If True, sort other candidate positions by score (best first).

		Returns:
			A torch.LongTensor of shape [K, 2] where K <= n.
		"""
		# Compute target center and its flat index
		target_pos  = self.target_x[ref_obj, Indices.COORD]

		# Check if target is free
		if not self.is_invalid_center(target_pos, ref_obj):
			return target_pos.view(1, 2)

		# Otherwise, fetch n candidate empties
		candidates = self.get_empty_positions(ref_obj=ref_obj, n=n, sort=sort)

		return candidates

	def find_blocking_objects(self, obj: int) -> List[int]:
		"""
		For self.current_table (current occupancy) and self.target_x (goal layout),
		find which objects in current_table occupy the target footprint of `obj`.
		Excludes `obj` itself.

		Args:
			obj: The index of the object whose target footprint to check.

		Returns:
			A list of integer indices of objects currently blocking the target footprint.
		"""
		table = self.current_table    # torch.Tensor [H,W], values 0=empty or (idx+1)

		# Compute target footprint bounds
		coor_t = self.target_x[obj, Indices.COORD].tolist()
		size = self.current_x[obj, Indices.SIZE].tolist()
		
		# Slice and gather unique occupiers
		sub    = table[get_patch_slice(coor_t, size, self.grid_size)]          # footprint region
		uniq   = torch.unique(sub)            # all IDs in that region
		others = uniq[uniq > 0] - 1           # convert ID+1 → idx

		# Exclude the object itself
		blockers = others[others != obj]

		# Return as a Python list
		return blockers.tolist()

	## --scoring functions--
	def occupied_score(self, centers: torch.Tensor, obj: int) -> torch.Tensor:
		"""
		For each candidate center in `centers` ([M,2]), compute the fraction
		of that object's h×w footprint occupied by *other* objects in the
		target scene. Returns a FloatTensor of shape [M].

		Args:
			centers: A torch.Tensor of shape [M,2] representing candidate (x,y) centers.
			obj: The index of the object to be placed.

		Returns:
			A FloatTensor of shape [M] with occupancy scores.
		"""
		H, W = self.grid_size
		h, w = self.current_x[obj, Indices.SIZE]

		# Build a boolean map of all “other-object” occupancy.
		# 1.0 where the grid is occupied by someone ≠ obj, else 0.0.
		occ = self.target_table
		obj_id = obj + 1
		occ_others = ((occ != 0) & (occ != obj_id)).to(torch.float32)

		# Compute integral image (prefix‐sum) with a zero row/col pad so that
		#   S[i+1, j+1] = sum of occupied_by_others[:i+1, :j+1]
		S = occ_others.cumsum(dim=0).cumsum(dim=1)	# [H, W]
		S = F.pad(S, (1, 0, 1, 0), value=0.0)		# [H+1, W+1]

		# Extract submatrices for the four corners of each potential window:
		H2 = H - h + 1
		W2 = W - w + 1
		br = S[h:   h+H2, w:   w+W2]  # bottom-right corners
		tr = S[0:   H2,   w:   w+W2]  # top-right
		bl = S[h:   h+H2, 0:   W2  ]  # bottom-left
		tl = S[0:   H2,   0:   W2  ]  # top-left

		# Window-sum = br - tr - bl + tl
		# If that sum is zero, the window is entirely free.
		counts = br - tr - bl + tl    # [H-h+1, W-w+1]

		# Clamp each center to its top‐left footprint origin
		pts = centers.long()
		if pts.ndim == 1:
			pts = pts.view(1, 2)
		ti = (pts[:, 0] - h // 2).clamp(0, H2 - 1)
		tj = (pts[:, 1] - w // 2).clamp(0, W2 - 1)

		# Gather and normalize
		occ_counts = counts[ti, tj]	# [M]
		area = float(h * w)
		return occ_counts / area			# FloatTensor [M]

	def score(self, centers: torch.Tensor, obj: int) -> torch.Tensor:
		"""
		Vectorized composite score for placing `obj` at each center in `centers`.
		The score is a combination of the occupied fraction of the footprint
		in the target scene and the normalized Euclidean distance to the center 
		between the current and target positions.
		Lower scores are better.

		Args:
			centers: [M,2] long or float tensor of candidate center coordinates.
			obj: The index of the object to score.

		Returns:
			[M] float tensor of composite scores.
		"""
		# Occupancy fraction at each placement
		occ_frac = self.occupied_score(centers, obj)  # [M]

		# Euclidean distance from each center to the mid
		cur_center = self.current_x[obj, Indices.COORD]  # [2]
		tgt_center = self.target_x[obj, Indices.COORD]   # [2]
		mid = (cur_center + tgt_center) / 2.0            # [2]

		diffs = centers.float() - mid.float().unsqueeze(0)		# [M,2]
		distances  = torch.norm(diffs, dim=1)                   # [M]

		return occ_frac + distances * self.normalization_factor	# [M]

	## --valid actions--
	def get_valid_stacks(self) -> List[int]:
		"""
		Generates a list of all valid 'stack' actions from the current scene state.
		Considers stability rules and whether the target object is already occupied.
		Applies `static_stack` rule: if True, `start_obj` must have nothing stacked on it.

		Returns:
			A list of encoded integer action codes for valid stack actions.
		"""
		# Which objects k are empty
		rel     = self.current_x[:, Indices.RELATION]       # [N,N]
		empty_k = ~rel.any(dim=0)              # [N]

		# Combine & apply static_stack
		mask = self.stability_mask & empty_k.view(1, self.N)
		if self.static_stack:
			mask &= empty_k.view(self.N, 1)
		mask.fill_diagonal_(False)

		# Extract and encode
		pairs = torch.nonzero(mask, as_tuple=False)  # [M,2]
		if pairs.numel() == 0:
			return []

		# Batch‐encode them
		starts = pairs[:,0]
		targets = pairs[:,1]
		batch_codes = self.encode_stack(starts, targets)
		return batch_codes.tolist()

	def get_valid_moves(self, max_move_num: int=10) -> List[int]:
		"""
		For each object `k`, gathers up to `max_move_num` free grid centers
		where it can be moved (ignoring stacking initially).
		Applies `static_stack` by skipping any `k` that has something on top of it.

		Args:
			max_move_num: The maximum number of valid move positions to consider for each object.

		Returns:
			A list of encoded integer action codes for valid move actions.
		"""
		actions = []

		# Which k are allowed (static_stack skips non‐empty actors)
		if self.static_stack:
			rel     = self.current_x[:, Indices.RELATION]
			empty_k = ~rel.any(dim=0)                  # True if k has no one on top
			ks      = empty_k.nonzero(as_tuple=False).view(-1)
		else:
			ks = torch.arange(self.N)

		# For each k, gather valid centers and batch‐encode
		for k in ks.tolist():
			mask = self.valid_center_mask(k)			# [H, W]
			idxs = torch.nonzero(mask, as_tuple=False)  # [M_k,2]
			M_k   = idxs.size(0)
			if M_k == 0:
				continue

			# Subsample up to max_move_num
			if M_k > max_move_num:
				perm = torch.randperm(M_k)[:max_move_num]
				idxs = idxs[perm]

			# Batch‐encode: starts and targets are both k
			B = idxs.size(0)
			starts  = torch.full((B,), k, dtype=torch.long)
			codes   = self.encode_move(starts, idxs)
			actions.extend(codes.tolist())

		return actions

	def get_valid_actions(self, max_move_num: int=10) -> List[int]:
		"""
		Returns a combined list of all valid 'stack' and 'move' actions.

		Args:
			max_move_num: The maximum number of valid move positions to consider for each object.

		Returns:
			A list of encoded integer action codes.
		"""
		return self.get_valid_stacks() + self.get_valid_moves(max_move_num=max_move_num)

	## --step functions--
	def is_terminal_state(self, state: Optional[Dict[str, torch.Tensor]]=None) -> bool:
		"""
		Checks if the current scene state matches the target scene state.
		Returns: True if current_x is identical to target_x, False otherwise.
		"""
		if state is not None:
			return torch.equal(state['current'], self.target_x)
		return torch.equal(self.current_x, self.target_x)

	def step_move(self, obj: int, coord: torch.Tensor, log: bool=True) -> Tuple[float, Dict[str, torch.Tensor]]:
		"""
		Performs a 'move' action.
		Args:
			obj: The index of the object to move.
			coord: The target (x,y) coordinate for the object.
			log: If True, prints action details and cost.
		Returns:
			A tuple: (cost, new_state).
		"""
		action = int(self.encode_move(obj, coord))
		return self.step(action, log)

	def step_stack(self, start_obj: int, target_obj: int, log: bool=True) -> Tuple[float, Dict[str, torch.Tensor]]:
		"""
		Performs a 'stack' action.
		Args:
			start_obj: The index of the object to stack.
			target_obj: The index of the object to stack upon.
			log: If True, prints action details and cost.
		Returns:
			A tuple: (cost, new_state).
		"""
		action = int(self.encode_stack(start_obj, target_obj))
		return self.step(action, log)

	def step(self, action: int, log: bool=False) -> Tuple[float, Dict[str, torch.Tensor]]:
		"""
		Executes a single action in the environment and updates the state.

		Args:
			action: An integer representing the encoded action.
			log: If True, prints details of the action and its cost.

		Returns:
			A tuple:
			- cost (float): The cost incurred by taking the action.
			- new_state (dict): A dictionary representing the new environment state.
		"""
		action_type, start_obj, target_obj, coord = self.decode_action(action)
		return self._step(action_type, start_obj, target_obj, coord, log=log)

	def _step(self, action_type: str, start_obj: int, target_obj: int, coord: torch.Tensor, log: bool=False) -> Tuple[float, Dict[str, torch.Tensor]]:
		"""
		Internal helper to execute a single action.

		Args:
			action_type: 'move' or 'stack'.
			start_obj: The object index for the action.
			target_obj: The target object index for 'stack' actions.
			coord: The target coordinate for 'move' actions.
			log: If True, prints action details and cost.

		Returns:
			A tuple: (cost, new_state).
		"""
		cost = 0.0
		terminated = False

		rel = self.current_x[:, Indices.RELATION]

		if action_type == 'move':
			if self.static_stack and rel[:, start_obj].any():
				raise ValueError(f'can not move non-empty objects')
			cost += self.move_func(start_obj, coord)
		elif action_type == 'stack':
			if self.static_stack and rel[:, start_obj].any():
				raise ValueError(f'can not stack non-empty objects')
			cost += self.stack_func(start_obj, target_obj)

		cost += self.pp_cost # Add per-pickup cost

		if self.is_terminal_state():
			if self.terminal_cost:
				# Add cost for manipulator to return to initial position
				cost += self.cal_manipulator_cost(self.manipulator_init_pos)
			self.manipulator = self.manipulator_init_pos.clone()
			terminated = True

		if log:
			if action_type == 'move':
				print(f'Moved {start_obj} to: {coord.numpy()} | cost: {cost:.3f} | done: {terminated}')
			else:
				print(f'Stacked {start_obj} -> {target_obj} | cost: {cost:.3f} | done: {terminated}')

		return cost, self.get_state()
