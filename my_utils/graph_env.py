import math
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from torch_geometric.data import Data
import numpy as np

_OBJECTS_SET_1 = {
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

_OBJECTS_SET_2 = {
    0: {"name": "fork", "color": "yellow", "category": "cat1", "size": (7, 7)},
    1: {"name": "apple", "color": "orange", "category": "cat2", "size": (11, 11)},
    2: {"name": "banana", "color": "orange", "category": "cat2", "size": (17, 17)},
    3: {"name": "paper-cup", "color": "green", "category": "cat3", "size": (11, 11)},
    4: {"name": "bowl", "color": "green", "category": "cat3", "size": (17, 17)},
    5: {"name": "basket", "color": "blue", "category": "cat4", "size": (21, 21)},
    6: {"name": "pan", "color": "blue", "category": "cat4", "size": (27, 27)}
}

OBJECTS = _OBJECTS_SET_1	# Default value

def set_objects(version='set1'):
    global OBJECTS
    if version == 'set1':
        OBJECTS = _OBJECTS_SET_1
    elif version == 'set2':
        OBJECTS = _OBJECTS_SET_2

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
	# COORD = slice(0, 2)
	# RELATION = slice(2, None)

def flatten_pos(pos, grid_size: tuple):
	return int(pos[0] * grid_size[1] + pos[1])

def unflatten_pos(flat: int, grid_size: tuple):
	row = flat // grid_size[1]
	col = flat % grid_size[1]
	return torch.tensor(np.array([row, col]), dtype=torch.float32)

def copy_graph(graph):
	return graph.clone().detach()

def is_in_env(coor, size, grid_size):
	if coor[0] - size[0]//2 < 0 or coor[0] + size[0]//2 >= grid_size[0]:
		return False
	if coor[1] - size[1]//2 < 0 or coor[1] + size[1]//2 >= grid_size[1]:
		return False
	return True

def get_object_bellow(x, node):
	i = (x[node, Indices.RELATION] == 1).nonzero(as_tuple=True)[0]
	if len(i):
		return i.item()
	return None

def get_object_above(x, node):
	i = (x[:, Indices.RELATION.start + node] == 1).nonzero(as_tuple=True)[0]
	if len(i):
		return i.item()
	return None

def is_empty_object(x, node):
	return get_object_above(x, node) is None

def is_stacked_on_object(x, node1, node2):
	if x[node1, Indices.RELATION.start + node2] == 0:
		return False
	return True

def is_stacked(x, node):
	return get_object_bellow(x, node) is not None

def get_object_base(x, node):
	while is_stacked(x, node):
		node = get_object_bellow(x, node)
	return node

def get_obj_pos(x, node):
	return x[node, Indices.COORD].clone()

def get_obj_size(x, node):
	return x[node, Indices.SIZE].clone().tolist()

def get_obj_label(x, node):
	return int(x[node, Indices.LABEL].item())

def get_obj_relation(x, node):
	return x[node, Indices.RELATION].clone().tolist()

def in_table_index(center, size):
    """
    Helper function to generate a mask for placing an object on the table.
    """
    x, y = int(center[0]), int(center[1])
    half_w, half_h = int(size[0] // 2), int(size[1] // 2)
    return slice(x - half_w, x + half_w + 1), slice(y - half_h, y + half_h + 1)

def get_node_poses(num_nodes):
    if num_nodes < 3:
        raise ValueError("Number of nodes must be at least 3")

    radius = 1  # You can adjust the radius to change the polygon size
    angle_step = 2 * math.pi / num_nodes
    positions = []

    for i in range(num_nodes):
        angle = i * angle_step
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        positions.append([x, y])

    return positions

def is_stable(x, node1, node2):
    label1 = get_obj_label(x, node1)
    label2 = get_obj_label(x, node2)

    cat1 = OBJECTS[label1]['category']
    cat2 = OBJECTS[label2]['category']

    return cat2 in CATEGORY_STABILITY[cat1]

def plot_graph(graph, grid_size, ax=None, fig_size=2.5, title=None, constraints=[]):
	if ax is None:
		fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size * (grid_size[1] / grid_size[0])))

	color_by_label = {k + 1: v['color'] for k, v in OBJECTS.items()}
	color_by_label[0] = 'white'  # Background
	color_by_label[max(color_by_label.keys()) + 1] = 'red'  # Constraints

	all_sizes = graph.x[:, Indices.SIZE]
	all_labels = sorted(color_by_label.keys())
	color_list = [color_by_label[label] for label in all_labels]

	mapped_table = -1 * np.ones(grid_size, dtype=int)
	unrendered_nodes = list(range(graph.num_nodes))
	while len(unrendered_nodes) > 0:
		i = unrendered_nodes.pop(0)
		label_i = get_obj_label(graph.x, i)
		coor_i = get_obj_pos(graph.x, i)
		size_i = get_obj_size(graph.x, i)

		if is_stacked(graph.x, i):
			child = get_object_bellow(graph.x, i)
			if child in unrendered_nodes:
				unrendered_nodes.append(i)
				continue

			child_label = get_obj_label(graph.x, child)
			if OBJECTS[label_i]['category'] == 'cat1' and OBJECTS[child_label]['category'] == 'cat3':
				size_i = [size_i[0] // 4, size_i[1] // 4]
			elif torch.all(all_sizes == all_sizes[0]):
				size_i = [size_i[0] // 2, size_i[1] // 2]

		mapped_table[in_table_index(coor_i, size_i)] = label_i + 1

	for c in constraints:
		if isinstance(c, torch.Tensor):
			c = c.numpy()
		c_x, c_y = map(int, c)
		mapped_table[c_x, c_y] = list(color_by_label.keys())[-1]

	cmap = ListedColormap(color_list)
	bounds = np.arange(len(color_list)+1)
	norm = BoundaryNorm(bounds, cmap.N)
	ax.imshow(mapped_table, cmap=cmap, norm=norm, origin='upper')

	ax.set_xticks(np.arange(-0.5, mapped_table.shape[1], 1), minor=True)
	ax.set_yticks(np.arange(-0.5, mapped_table.shape[0], 1), minor=True)
	ax.tick_params(which='minor', bottom=False, left=False)
	ax.set_xticks([])
	ax.set_yticks([])

	for i in range(graph.num_nodes):
		label_i = get_obj_label(graph.x, i)
		coor_i = get_obj_pos(graph.x, i)
		size_i = get_obj_size(graph.x, i)
		if is_stacked(graph.x, i):
			child = get_object_bellow(graph.x, i)
			child_label = get_obj_label(graph.x, child)
			if OBJECTS[label_i]['category'] == 'cat1' and OBJECTS[child_label]['category'] == 'cat3':
				size_i = [size_i[0] // 4, size_i[1] // 4]
			elif torch.all(all_sizes == all_sizes[0]):
				size_i = [size_i[0] // 2, size_i[1] // 2]

		ax.text(coor_i[1] - size_i[1] // 2, coor_i[0] - size_i[0] // 2, str(i),
				ha='center', va='center', color='black')

	if title is not None:
		ax.set_title(title)

def cal_density(graph, grid_size):
	phi = 0
	for i in range(graph.num_nodes):
		size_i = get_obj_size(graph.x, i)
		phi += (size_i[0] * size_i[1])

	return phi / (grid_size[0] * grid_size[1])

def x_to_edge_index(x):
	edge_index = torch.tensor([], dtype=torch.long)
	for i in range(len(x)):
		for j in range(len(x)):
			if x[i][Indices.RELATION.start+j] == 1:
				edge_index = torch.cat((edge_index, torch.tensor([[i], [j]])), dim=1)
	return edge_index

def generate_random_coordinates(grid_size: tuple, sizes: list, side_assignments: list=None, max_attempts: int=100) -> list:
	"""
	Generate random coordinates for objects.

	Args:
		grid_size (tuple): Dimensions (height, width) of the grid.
		sizes (list of tuple): List of (height, width) sizes (must be odd integers) for each object.
		max_attempts (int): Maximum number of attempts.
		side_assignments (list of str): List of 'left' or 'right' for each object.

	Returns:
		list of torch.Tensor: List of coordinates for the placed objects.
	"""

	for _ in range(max_attempts):  # Keep trying until all objects are successfully placed
		table = np.zeros(grid_size, dtype=int)
		success = True
		coords = []

		for i, size in enumerate(sizes):
			placed = False
			for _ in range(max_attempts):
				x_min, x_max = size[0] // 2, grid_size[0] - size[0] // 2 - 1

				# Determine bounds based on side assignment
				if side_assignments is not None:
					half_width = grid_size[1] // 2
					if side_assignments[i] == 'left':
						y_min, y_max = size[1] // 2, (grid_size[1]) // 3 - size[1] // 2 - 1
					else:
						y_min, y_max = (grid_size[1]*2) // 3 + size[1] // 2, grid_size[1] - size[1] // 2 - 1
				else:
					y_min, y_max = size[1] // 2, grid_size[1] - size[1] // 2 - 1

				# Generate random coordinates for the object's center
				coor = torch.tensor([
					random.randint(x_min, x_max),
					random.randint(y_min, y_max)
				], dtype=torch.float32)

				# Check if the placement is valid
				x_slice, y_slice = in_table_index(coor, size)
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

def create_graph(
		num_nodes: int, 
		grid_size: tuple, 
		num_labels: int, 
		object_sizes: dict, 
		labels: list=None, 
		stack_prob: float=0.6, 
		max_attempts: int=10,
		use_sides: bool=False,
		prev_sides: list=None,
		switch_prob: float=0.5,
	) -> Data:

	assert num_nodes >= 2

	if max_attempts == 0:
		raise ValueError('Failed to create a graph with the given parameters')

	label_flag = True
	if labels is None:
		label_flag = False
		labels = [random.randint(0, num_labels-1) for _ in range(num_nodes)]
		if np.sum([object_sizes[labels[id]][0]*object_sizes[labels[id]][1] for id in range(num_nodes)]) > grid_size[0]*grid_size[1]:
			print('Error: The combined object sizes exceed grid capacity')
			return create_graph(num_nodes, grid_size, num_labels, object_sizes, None, stack_prob, max_attempts-1)
	elif len(labels) != num_nodes:
		raise ValueError('Number of labels must be equal to the number of nodes')
	else:
		if np.sum([object_sizes[labels[id]][0]*object_sizes[labels[id]][1] for id in range(num_nodes)]) > grid_size[0]*grid_size[1]:
			raise ValueError('Man! The objects are too big!')

	edge_index = torch.tensor([], dtype=torch.long)
	x_arr = torch.zeros(num_nodes, num_nodes+4)
	x_arr = torch.cat([torch.tensor(labels, dtype=torch.float).reshape(-1, 1), x_arr], dim=1)

	# Random label for each node
	nodes = list(range(num_nodes))
	for node in nodes:
		x_arr[node][Indices.LABEL] = labels[node]
		x_arr[node][Indices.SIZE] = torch.tensor(object_sizes[labels[node]], dtype=torch.float32)

	random.shuffle(nodes)

	# Randomly create edges between nodes based on stacking probability
	for node in nodes:
		if random.random() < stack_prob:
			node1 = node
			node2 = node1
			while node2 == node1:
				node2 = random.randint(0, num_nodes-1)

			if not is_stable(x_arr, node1, node2):
				continue

			# Avoid stacking multiple objects on top of the same base
			if torch.sum(x_arr[:, Indices.RELATION.start + node2]) >= 1:
				continue

			edge_index = torch.cat([edge_index, torch.tensor([[node1], [node2]], dtype=torch.long)], dim=1)
			x_arr[node1, Indices.RELATION.start + node2] = 1

	base_nodes = [i for i in range(num_nodes) if not is_stacked(x_arr, i)]

	# For target_graph: maybe switch side
	side_assignments = None
	if use_sides:
		if prev_sides is None:
			side_assignments_all = [random.choice(['left', 'right']) for _ in range(num_nodes)]
		else:
			side_assignments_all = []
			for i in range(num_nodes):
				if random.random() < switch_prob:
					side = 'right' if prev_sides[i] == 'left' else 'left'
					if i in base_nodes:
						print(f'{i} switched to {side}')
				else:
					side = prev_sides[i]
				side_assignments_all.append(side)
		side_assignments = [side_assignments_all[i] for i in base_nodes]

	# Allocate random positions to base nodes
	base_sizes = [get_obj_size(x_arr, i) for i in base_nodes]
	coords = generate_random_coordinates(grid_size, base_sizes, side_assignments=side_assignments, max_attempts=300)
	if coords is None:
		print('Trying again!')
		if label_flag:
			return create_graph(
				num_nodes=num_nodes,
				grid_size=grid_size, 
				num_labels=num_labels,
				object_sizes=object_sizes,
				labels=labels,
				stack_prob=stack_prob,
				max_attempts=max_attempts-1,
				use_sides=use_sides,
				prev_sides=prev_sides,
				switch_prob=switch_prob,
			)
		else:
			return create_graph(
				num_nodes=num_nodes,
				grid_size=grid_size, 
				num_labels=num_labels,
				object_sizes=object_sizes,
				labels=None,
				stack_prob=stack_prob,
				max_attempts=max_attempts-1,
				use_sides=use_sides,
				prev_sides=prev_sides,
				switch_prob=switch_prob,
			)
	
	unallocated_nodes = list(range(num_nodes))
	for i in base_nodes:
		x_arr[i][Indices.COORD] = coords.pop(0).clone()
		unallocated_nodes.remove(i)

	while len(unallocated_nodes) > 0:
		i = unallocated_nodes.pop(0)
		if not is_stacked(x_arr, i):
			raise ValueError('Unknown error in creating graph')
		target_node = get_object_bellow(x_arr, i)
		if target_node in unallocated_nodes:
			unallocated_nodes.append(i)
			continue
		x_arr[i][Indices.COORD] = get_obj_pos(x_arr, target_node)

	return Data(x=x_arr, edge_index=edge_index, pos=get_node_poses(num_nodes))
