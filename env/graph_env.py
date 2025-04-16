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
	COORD = slice(1, 3)
	RELATION = slice(3, None)
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

# def is_stable(x, node1, node2):
# 	label_1 = x[node1][Indices.LABEL]
# 	label_2 = x[node2][Indices.LABEL]
# 	if label_1 == 0 and (label_2 == 2 or label_2 == 3):
# 		return True
# 	elif label_2 == 3 and (label_1 == 1 or label_1 == 2):
# 		return True
# 	return False

def flatten_pos(pos, grid_size: tuple):
	return int(pos[0] * grid_size[1] + pos[1])

def unflatten_pos(flat: int, grid_size: tuple):
	row = flat // grid_size[1]
	col = flat % grid_size[1]
	return torch.tensor(np.array([row, col]), dtype=torch.float32)

def copy_graph(graph):
	return graph.clone().detach()

### Random Graph Generation

def in_table_index(coor, size):
    """
    Helper function to generate a mask for placing an object on the table.
    """
    x, y = int(coor[0]), int(coor[1])
    half_w, half_h = size[0] // 2, size[1] // 2
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
			attempts = 0

			while not placed:
				attempts += 1
				if attempts > max_attempts:
					# Restart the entire placement process
					success = False
					break

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

			if not success:
				break  # Restart placement process

		if success:
			return coords  # Successfully placed all objects
	return None  # Failed to place all objects after max attempts

def generate_random_coordinates_with_ratio(grid_size: tuple, sizes: list, ratio: float, max_attempts: int=100, adjustment_factor: float=0.01) -> list:
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
		current_ratio = ratio  # Start with the initial ratio

		for i, size in enumerate(sizes):
			placed = False
			attempts = 0

			while not placed:
				attempts += 1
				if attempts > max_attempts:
					# Adjust the ratio slightly and restart
					if current_ratio > 0.5:
						current_ratio = max(0.5, current_ratio - adjustment_factor)
					else:
						current_ratio = min(0.5, current_ratio + adjustment_factor)
					success = False
					break

				# Determine whether to place in the left or right half based on the current ratio
				x_min, x_max = size[0] // 2, grid_size[0] - size[0] // 2 - 1
				# y_min, y_max = size[1] // 2, grid_size[1] - size[1] // 2 - 1

				if random.random() < current_ratio:
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

			if not success:
				break  # Restart placement process with adjusted ratio

		if success:
			return coords  # Successfully placed all objects
	return None  # Failed to place all objects after max attempts

def create_graph_label_continuous(num_nodes: int, grid_size: tuple, num_labels: int, object_sizes: list, labels=None, p: float=0.6, ratio: float=0.5, iterations: int=100) -> Data:
	if iterations == 0:
		raise ValueError('Failed to create a graph with the given parameters')
	
	# if grid_size[0] > grid_size[1]:
	# 	min_size = min([object_sizes[i][0] for i in range(num_labels)])
	# 	if min_size * num_nodes > grid_size[0]:
	# 		raise ValueError('Grid size is too small for the objects')
	# else:
	# 	min_size = min([object_sizes[i][1] for i in range(num_labels)])
	# 	if min_size * num_nodes > grid_size[1]:
	# 		raise ValueError('Grid size is too small for the objects')
	max_size_x = max([object_sizes[i][0] for i in range(num_labels)])
	max_size_y = max([object_sizes[i][1] for i in range(num_labels)])
	if max_size_x > grid_size[0] or max_size_y > grid_size[1]:
		raise ValueError('Grid size is too small for the objects')

	if labels is None:
		labels = [random.randint(0, num_labels-1) for _ in range(num_nodes)]
		if np.sum([object_sizes[labels[i]][0]*object_sizes[labels[i]][0] for i in range(num_nodes)]) > grid_size[0]*grid_size[1]:
			return create_graph_label_continuous(num_nodes, grid_size, num_labels, object_sizes, None, p, iterations-1)
	elif len(labels) != num_nodes:
		raise ValueError('Number of labels must be equal to the number of nodes')
	else:
		if np.sum([object_sizes[labels[i]][0]*object_sizes[labels[i]][0] for i in range(num_nodes)]) > grid_size[0]*grid_size[1]:
			raise ValueError('objects are too big')

	edge_index = torch.tensor([], dtype=torch.long)
	x_arr = torch.zeros(num_nodes, num_nodes+2)
	x_arr = torch.cat([torch.tensor(labels, dtype=torch.float).reshape(-1, 1), x_arr], dim=1)

	# Shuffle nodes
	nodes = [i for i in range(num_nodes)]
	random.shuffle(nodes)

	# Random label for each node
	for node in nodes:
		x_arr[node][Indices.LABEL] = labels[node]

	# Random connection between edges
	for node in nodes:
		if random.random() < p:
			node1 = node
			node2 = random.randint(0, num_nodes-1)
			if node1 != node2:
				if torch.any(torch.all(edge_index == torch.tensor([[node2], [node1]], dtype=torch.long), dim=0)):
					continue
				if not is_stable(x_arr, node1, node2):
					continue
				if torch.sum(x_arr[:, Indices.RELATION.start+node2]) >= 1:
					continue
				edge_index = torch.cat([edge_index, torch.tensor([[node1], [node2]], dtype=torch.long)], dim=1)
				x_arr[node1][Indices.RELATION.start+node2] = 1

	# Allocate random positions to base nodes
	sizes = [object_sizes[labels[i]] for i in range(num_nodes) if x_arr[i, Indices.RELATION].sum() == 0]
	if ratio == 0.5:
		coords = generate_random_coordinates(grid_size, sizes, 300)
	else:
		coords = generate_random_coordinates_with_ratio(grid_size, sizes, ratio, 300)
	unallocated_nodes = list(range(num_nodes))
	for i in range(num_nodes):
		if x_arr[i, Indices.RELATION].sum() == 0:
			x_arr[i][Indices.COORD] = coords.pop(0).clone()
			unallocated_nodes.remove(i)

	while len(unallocated_nodes) > 0:
		i = unallocated_nodes.pop(0)
		if x_arr[i, Indices.RELATION].sum() == 0:
			raise ValueError('Unknown error in creating graph')
		child_node = (x_arr[i, Indices.RELATION] == 1).nonzero(as_tuple=True)[0].item()
		if child_node in unallocated_nodes:
			unallocated_nodes.append(i)
			continue
		x_arr[i][Indices.COORD] = x_arr[child_node][Indices.COORD].clone()

	return Data(x=x_arr, edge_index=edge_index, pos=get_node_poses(num_nodes))
