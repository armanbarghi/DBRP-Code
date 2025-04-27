import math
import torch
import random
import numpy as np
import networkx as nx
import torch_geometric
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import numpy as np
from typing import Union

class Indices:
	LABEL = slice(0, 1)
	SIZE = slice(1, 3)
	COORD = slice(3, 5)
	RELATION = slice(5, None)
	# COORD = slice(0, 2)
	# RELATION = slice(2, None)

def is_stable(x, node1, node2):
	label_1 = x[node1][Indices.LABEL]
	label_2 = x[node2][Indices.LABEL]
	if label_1 == 0 and (label_2 in [3, 4, 5, 6]):
		return True
	elif (label_2 in [5, 6]) and (label_1 in [0, 1, 2, 3, 4]):
		return True
	return False

def flatten_pos(pos, grid_size: tuple):
	return int(pos[0] * grid_size[1] + pos[1])

def unflatten_pos(flat: int, grid_size: tuple):
	row = flat // grid_size[1]
	col = flat % grid_size[1]
	return torch.tensor(np.array([row, col]), dtype=torch.float32)

def copy_graph(graph):
	return graph.clone().detach()

def find_target_obj(x, node):
	i = (x[node, Indices.RELATION] == 1).nonzero(as_tuple=True)[0]
	if len(i):
		return i.item()
	return None

def find_start_obj(x, node):
	i = (x[:, Indices.RELATION.start + node] == 1).nonzero(as_tuple=True)[0]
	if len(i):
		return i.item()
	return None

def is_empty_object(x, node):
	return find_start_obj(x, node) is None

def is_edge_in_graph(x, node1, node2):
	if x[node1, Indices.RELATION.start + node2] == 0:
		return False
	return True

def is_stacked_object(x, node):
	return x[node, Indices.RELATION].sum() > 0

def find_base_obj(x, node):
	while is_stacked_object(x, node):
		node = find_target_obj(x, node)
	return node

def get_obj_pos(x, node):
	return x[node, Indices.COORD].clone()

def get_obj_size(x, node):
	return x[node, Indices.SIZE].clone().tolist()

def get_obj_label(x, node):
	return x[node, Indices.LABEL].item()

def get_obj_relation(x, node):
	return x[node, Indices.RELATION].clone().tolist()

def in_table_index(coor, size):
    """
    Helper function to generate a mask for placing an object on the table.
    """
    x, y = int(coor[0]), int(coor[1])
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

### Random Graph Generation

def generate_random_coordinates(grid_size: tuple, sizes: list, max_attempts: int=100) -> list:
	"""
	Generate random coordinates for objects, restarting if a placement fails.

	Args:
		grid_size (int, int): Size of the table.
		sizes (list of tuple): List of (height, width) for each object (must be odd numbers).
		values (list of int): List of values to assign to the objects in the table.
		max_attempts (int): Maximum number of attempts before restarting.
		
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

def generate_random_coordinates_with_ratio(grid_size: tuple, sizes: list, ratio: float, max_attempts: int=100) -> list:
	"""
	Generate random coordinates for objects, ensuring a placement ratio across grid halves.

	Args:
		grid_size (int, int): Size of the table.
		sizes (list of tuple): List of (height, width) for each object (must be odd numbers).
		ratio (float): Proportion of objects to be placed in the left half of the grid (0 <= ratio <= 1).
		max_attempts (int): Maximum number of attempts before restarting placement.
		adjustment_factor (float): The amount to adjust the ratio if placement is impossible.

	Returns:
		list of torch.Tensor: List of coordinates for the placed objects.
	"""
	assert 0 <= ratio <= 1, "Ratio must be between 0 and 1"

	for _ in range(max_attempts):  # Keep trying until all objects are successfully placed
		table = np.zeros(grid_size, dtype=int)
		success = True
		coords = []

		for i, size in enumerate(sizes):

			placed = False
			for _ in range(max_attempts):

				# Determine whether to place in the left or right half based on the current ratio
				x_min, x_max = size[0] // 2, grid_size[0] - size[0] // 2 - 1
				# y_min, y_max = size[1] // 2, grid_size[1] - size[1] // 2 - 1

				if random.random() < ratio:
					y_min, y_max = size[1] // 2, grid_size[1] // 2 - 1
				else:
					y_min, y_max = grid_size[1] // 2, grid_size[1] - size[1] // 2 - 1

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

def create_graph(num_nodes: int, grid_size: tuple, num_labels: int, object_sizes: dict, labels=None, p: float=0.6, ratio: float=0.5, max_attempts: int=10) -> Data:
	if max_attempts == 0:
		raise ValueError('Failed to create a graph with the given parameters')
	
	# if grid_size[0] > grid_size[1]:
	# 	min_size = min([object_sizes[i][0] for i in range(num_labels)])
	# 	if min_size * num_nodes > grid_size[0]:
	# 		raise ValueError('Grid size is too small for the objects')
	# else:
	# 	min_size = min([object_sizes[i][1] for i in range(num_labels)])
	# 	if min_size * num_nodes > grid_size[1]:
	# 		raise ValueError('Grid size is too small for the objects')
	# max_size_x = max([object_sizes[i][0] for i in range(num_labels)])
	# max_size_y = max([object_sizes[i][1] for i in range(num_labels)])
	# if max_size_x > grid_size[0] or max_size_y > grid_size[1]:
	# 	raise ValueError('Grid size is too small for the objects')
	assert num_nodes >= 2

	label_flag = True
	if labels is None:
		label_flag = False
		labels = [random.randint(0, num_labels-1) for _ in range(num_nodes)]
		if np.sum([object_sizes[labels[i]][0]*object_sizes[labels[i]][1] for i in range(num_nodes)]) > grid_size[0]*grid_size[1]:
			print('Man! The objects are too big!')
			return create_graph(num_nodes, grid_size, num_labels, object_sizes, None, p, ratio, max_attempts-1)
	elif len(labels) != num_nodes:
		raise ValueError('Number of labels must be equal to the number of nodes')
	else:
		if np.sum([object_sizes[labels[i]][0]*object_sizes[labels[i]][1] for i in range(num_nodes)]) > grid_size[0]*grid_size[1]:
			raise ValueError('Man! The objects are too big!')

	edge_index = torch.tensor([], dtype=torch.long)
	x_arr = torch.zeros(num_nodes, num_nodes+4)
	x_arr = torch.cat([torch.tensor(labels, dtype=torch.float).reshape(-1, 1), x_arr], dim=1)

	# Random label for each node
	nodes = list(range(num_nodes))
	random.shuffle(nodes)
	for node in nodes:
		x_arr[node][Indices.LABEL] = labels[node]
		x_arr[node][Indices.SIZE] = torch.tensor(object_sizes[labels[node]], dtype=torch.float32)

	# Random connection between edges
	for node in nodes:
		if random.random() < p:
			node1 = node
			node2 = node1
			while node2 == node1:
				node2 = random.randint(0, num_nodes-1)

			if not is_stable(x_arr, node1, node2):
				continue
			if torch.sum(x_arr[:, Indices.RELATION.start + node2]) >= 1:
				continue

			edge_index = torch.cat([edge_index, torch.tensor([[node1], [node2]], dtype=torch.long)], dim=1)
			x_arr[node1, Indices.RELATION.start + node2] = 1

	# Allocate random positions to base nodes
	base_sizes = [get_obj_size(x_arr, i) for i in range(num_nodes) if not is_stacked_object(x_arr, i)]

	if ratio == 0.5:
		coords = generate_random_coordinates(grid_size, base_sizes, 300)
	else:
		coords = generate_random_coordinates_with_ratio(grid_size, base_sizes, ratio, 300)

	if coords is None:
		print('Trying again!')
		if label_flag:
			return create_graph(num_nodes, grid_size, num_labels, object_sizes, labels, p, ratio, max_attempts-1)
		else:
			return create_graph(num_nodes, grid_size, num_labels, object_sizes, None, p, ratio, max_attempts-1)
	
	unallocated_nodes = list(range(num_nodes))
	for i in range(num_nodes):
		if not is_stacked_object(x_arr, i):
			x_arr[i][Indices.COORD] = coords.pop(0).clone()
			unallocated_nodes.remove(i)

	while len(unallocated_nodes) > 0:
		i = unallocated_nodes.pop(0)
		if not is_stacked_object(x_arr, i):
			raise ValueError('Unknown error in creating graph')
		
		target_node = find_target_obj(x_arr, i)
		if target_node in unallocated_nodes:
			unallocated_nodes.append(i)
			continue
		x_arr[i][Indices.COORD] = get_obj_pos(x_arr, target_node)

	return Data(x=x_arr, edge_index=edge_index, pos=get_node_poses(num_nodes))
