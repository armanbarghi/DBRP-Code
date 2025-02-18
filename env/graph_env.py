import torch
import random
import networkx as nx
import torch_geometric
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import numpy as np


class Indices:
	LABEL = slice(0, 1)
	COORD = slice(1, 3)
	RELATION = slice(3, None)
	# COORD = slice(0, 2)
	# RELATION = slice(2, None)

'''
non-container
0: yellow: spoon, fork, knife
1: orange: apple, pear, banana

container
2: green: cup
3: green: mug, bowl
4: blue: box, pan, basket
'''

def plot_graph(graph, ax=None, figsize=(2.5, 2.5), title=None, print_coords=None):
	if ax is None:
		fig, ax = plt.subplots(1, 1, figsize=figsize)
	graph_nx = torch_geometric.utils.to_networkx(graph)
	if print_coords:
		labels = {i: f"{i}\n{graph.x[i, Indices.COORD].numpy().astype(int)}" for i in range(graph.num_nodes)}
		if hasattr(Indices, 'LABEL'):
			node_colors = [graph.x[i, Indices.LABEL].item() for i in range(graph.num_nodes)]
			nx.draw(graph_nx, pos=graph.pos, labels=labels, with_labels=True, node_size=1000, font_size=7, font_weight='bold', ax=ax, node_color=node_colors, cmap=plt.cm.tab20)
		else:
			nx.draw(graph_nx, pos=graph.pos, labels=labels, with_labels=True, node_size=1000, font_size=7, font_weight='bold', ax=ax)
	else:
		nx.draw(graph_nx, pos=graph.pos, with_labels=True, ax=ax)
	if title is not None:
		ax.set_title(title)

def copy_graph(graph):
	return graph.clone().detach()

def dfs_cycle_detection(node, visited, rec_stack, edge_index, num_nodes):
	if not visited[node]:
		# Mark the current node as visited and part of the recursion stack
		visited[node] = True
		rec_stack[node] = True

		# Recur for all the vertices adjacent to this vertex
		for neighbor in edge_index[1][edge_index[0] == node]:
			if not visited[neighbor]:
				if dfs_cycle_detection(neighbor, visited, rec_stack, edge_index, num_nodes):
					return True
			elif rec_stack[neighbor]:
				return True

	# The node needs to be popped from recursion stack before function ends
	rec_stack[node] = False
	return False

def has_cycles(data):
	num_nodes = data.num_nodes
	edge_index = data.edge_index
	if len(edge_index) == 0:
		return False

	# Mark all the vertices as not visited and not part of recursion stack
	visited = [False] * num_nodes
	rec_stack = [False] * num_nodes

	# Call the recursive helper function to detect cycle in different DFS trees
	for node in range(num_nodes):
		if not visited[node]:
			if dfs_cycle_detection(node, visited, rec_stack, edge_index, num_nodes):
				return True
	return False

def get_parents(node, edge_index):
	if len(edge_index) == 0:
		return []
	outgoing_edges = edge_index[1] == node
	return edge_index[0][outgoing_edges].tolist()


### Random Graph Generation

def in_table_index(coor, size):
    """
    Helper function to generate a mask for placing an object on the table.
    """
    x, y = int(coor[0]), int(coor[1])
    half_w, half_h = size[0] // 2, size[1] // 2
    return slice(x - half_w, x + half_w + 1), slice(y - half_h, y + half_h + 1)

def is_stable(x, node1, node2):
	label_1 = x[node1][Indices.LABEL]
	label_2 = x[node2][Indices.LABEL]
	if label_1 == 0 and (label_2 == 2 or label_2 == 3 or label_2 == 4):
		return True
	elif label_2 == 4 and (label_1 == 1 or label_1 == 2 or label_1 == 3):
		return True
	return False

def get_node_poses(num_nodes):
	if num_nodes <= 4:
		return [[0,1],[1,1],[1,0],[0,0]]
	elif num_nodes == 5:
		return [[0,1],[0.5,1.5],[1,1],[1,0],[0,0]]
	elif num_nodes == 6:
		return [[0,1],[0.5,1.5],[1,1],[1,0],[0.5,-0.5],[0,0]]
	return None

def random_points(num_nodes, dim):
    if num_nodes > dim[0] * dim[1]:
        raise ValueError('Number of nodes is greater than the grid size')
    
    points_set = set()  # Use a set to store unique grid positions
    while len(points_set) < num_nodes:
        x = random.randint(0, dim[0]-1)  # X-coordinate range
        y = random.randint(0, dim[1]-1)  # Y-coordinate range
        points_set.add((x, y))  # Add as a tuple to ensure uniqueness
    
    # Convert the set of points into a tensor
    points = torch.tensor(list(points_set), dtype=torch.int32)
    return points

def create_graph(num_nodes, grid_size, p=0.6, not_empty=False):
	edge_index = torch.tensor([], dtype=torch.long)
	x = torch.zeros(num_nodes, num_nodes)
	x = torch.cat([random_points(num_nodes, (grid_size, grid_size)), x], dim=1)
	if hasattr(Indices, 'LABEL'):
		x = torch.cat([torch.zeros(num_nodes, 1), x], dim=1)

	# Shuffle nodes
	nodes = [i for i in range(num_nodes)]
	random.shuffle(nodes)

	# Random connection between edges
	for node in nodes:
		if random.random() < p:
			node1 = node
			node2 = random.randint(0, num_nodes-1)
			if node1 != node2:
				if torch.any(torch.all(edge_index == torch.tensor([[node2], [node1]], dtype=torch.long), dim=0)):
					continue
				edge_index = torch.cat([edge_index, torch.tensor([[node1], [node2]], dtype=torch.long)], dim=1)
				x[node1][Indices.RELATION.start+node2] = 1

	# Find all parents of the node and their parents recursively, then, move them
	for node in nodes:
		visited = [False] * num_nodes
		visited[node] = True
		stack = [node]
		while stack:
			node = stack.pop()
			parents = get_parents(node, edge_index)
			for i in parents:
				if not visited[i]:
					visited[i] = True
					stack.append(i)
					x[i][Indices.COORD] = x[node][Indices.COORD].clone()

	if not_empty and len(edge_index) == 0:
		return create_graph(num_nodes, grid_size, p, not_empty)

	graph = Data(x=x, edge_index=edge_index, pos=get_node_poses(num_nodes))
	if has_cycles(graph):
		return create_graph(num_nodes, grid_size, p, not_empty)
	return graph

def create_graph_label(num_nodes, grid_size, num_labels, labels=None, p=0.6, not_empty=False):
	if labels is None:
		labels = [random.randint(0, num_labels-1) for _ in range(num_nodes)]
	elif len(labels) != num_nodes:
		raise ValueError('Number of labels must be equal to the number of nodes')
	
	edge_index = torch.tensor([], dtype=torch.long)
	x = torch.zeros(num_nodes, num_nodes)
	x = torch.cat([random_points(num_nodes, (grid_size, grid_size)), x], dim=1)
	x = torch.cat([torch.tensor(labels, dtype=torch.float).reshape(-1, 1), x], dim=1)

	# Shuffle nodes
	nodes = [i for i in range(num_nodes)]
	random.shuffle(nodes)

	# Random label for each node
	for node in nodes:
		x[node][Indices.LABEL] = labels[node]

	# Random connection between edges
	for node in nodes:
		if random.random() < p:
			node1 = node
			node2 = random.randint(0, num_nodes-1)
			if node1 != node2:
				if torch.any(torch.all(edge_index == torch.tensor([[node2], [node1]], dtype=torch.long), dim=0)):
					continue
				if not is_stable(x, node1, node2):
					continue
				if torch.sum(x[:, Indices.RELATION.start+node2]) >= 1:
					continue
				edge_index = torch.cat([edge_index, torch.tensor([[node1], [node2]], dtype=torch.long)], dim=1)
				x[node1][Indices.RELATION.start+node2] = 1

	# Find all parents of the node and their parents recursively, then, move them
	for node in nodes:
		visited = [False] * num_nodes
		visited[node] = True
		stack = [node]
		while stack:
			node = stack.pop()
			parents = get_parents(node, edge_index)
			for i in parents:
				if not visited[i]:
					visited[i] = True
					stack.append(i)
					x[i][Indices.COORD] = x[node][Indices.COORD].clone()

	if not_empty and len(edge_index) == 0:
		return create_graph_label(num_nodes, grid_size, num_labels, labels, p, not_empty)

	graph = Data(x=x, edge_index=edge_index, pos=get_node_poses(num_nodes))
	if has_cycles(graph):
		return create_graph_label(num_nodes, grid_size, num_labels, labels, p, not_empty)
	return graph

def generate_random_coordinates(grid_size, sizes, max_attempts=1000):
	"""
	Generate random coordinates for objects, restarting if a placement fails.

	Args:
		grid_size (int): Size of the square table (grid_size x grid_size).
		sizes (list of tuple): List of (width, height) for each object (must be odd numbers).
		values (list of int): List of values to assign to the objects in the table.
		max_attempts (int): Maximum number of attempts before restarting.
		
	Returns:
		np.ndarray: Table with placed objects.
	"""
	while True:  # Keep trying until all objects are successfully placed
		table = np.zeros((grid_size, grid_size), dtype=int)
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

				x_min, x_max = size[0] // 2, grid_size - size[0] // 2 - 1
				y_min, y_max = size[1] // 2, grid_size - size[1] // 2 - 1

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

def generate_random_coordinates_with_ratio(grid_size, sizes, ratio, max_attempts=100, adjustment_factor=0.01):
	"""
	Generate random coordinates for objects, ensuring a placement ratio across grid halves.

	Args:
		grid_size (int): Size of the square grid (grid_size x grid_size).
		sizes (list of tuple): List of (width, height) for each object (must be odd numbers).
		ratio (float): Proportion of objects to be placed in the left half of the grid (0 <= ratio <= 1).
		max_attempts (int): Maximum number of attempts before restarting placement.
		adjustment_factor (float): The amount to adjust the ratio if placement is impossible.

	Returns:
		list of torch.Tensor: List of coordinates for the placed objects.
	"""
	assert 0 <= ratio <= 1, "Ratio must be between 0 and 1"

	while True:  # Keep trying until all objects are successfully placed
		table = np.zeros((grid_size, grid_size), dtype=int)
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
				x_min, x_max = size[0] // 2, grid_size - size[0] // 2 - 1
				# y_min, y_max = size[1] // 2, grid_size - size[1] // 2 - 1

				if random.random() < current_ratio:
					y_min, y_max = size[1] // 2, grid_size // 2 - 1
				else:
					y_min, y_max = grid_size // 2, grid_size - size[1] // 2 - 1

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

def create_graph_label_continuous(num_nodes, grid_size, num_labels, object_sizes, labels=None, p=0.6, ratio=0.5):
	min_size = min([max(object_sizes[i][0], object_sizes[i][1]) for i in range(num_labels)])
	if min_size * num_nodes > grid_size:
		raise ValueError('Grid size is too small for the objects')
	if labels is None:
		labels = [random.randint(0, num_labels-1) for _ in range(num_nodes)]
		if np.sum([object_sizes[labels[i]][0]*object_sizes[labels[i]][0] for i in range(num_nodes)]) > grid_size*grid_size:
			return create_graph_label_continuous(num_nodes, grid_size, num_labels, object_sizes, None, p)
	elif len(labels) != num_nodes:
		raise ValueError('Number of labels must be equal to the number of nodes')
	else:
		if np.sum([object_sizes[labels[i]][0]*object_sizes[labels[i]][0] for i in range(num_nodes)]) > grid_size*grid_size:
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
		coords = generate_random_coordinates(grid_size, sizes, 1000)
	else:
		coords = generate_random_coordinates_with_ratio(grid_size, sizes, ratio, 1000)
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


### Graph Environment

class BaseGraphEnv():
	def __init__(self, num_nodes, grid_size, max_steps=200):
		self.max_steps = max_steps
		self.num_nodes = num_nodes
		self.grid_size = grid_size

	def create_graph(self):
		raise NotImplementedError

	def is_terminal_state(self):
		return torch.equal(self.state_graph.x, self.target_graph.x)

	def find_connected_node(self, node):
		if len(self.state_graph.edge_index) == 0:
			return None
		outgoing_edges = self.state_graph.edge_index[0] == node
		connected_nodes = self.state_graph.edge_index[1][outgoing_edges].tolist()
		return connected_nodes[0] if connected_nodes else None

	def is_in_state_graph(self, node1, node2):
		if len(self.state_graph.edge_index) == 0:
			return False
		return ((self.state_graph.edge_index[0] == node1) & (self.state_graph.edge_index[1] == node2)).any()

	def is_in_target_graph(self, node1, node2):
		if len(self.target_graph.edge_index) == 0:
			return False
		return ((self.target_graph.edge_index[0] == node1) & (self.target_graph.edge_index[1] == node2)).any()

	def is_coor_occupied(self, coor):
		n = self.state_graph.num_nodes
		for i in range(n):
			if torch.equal(self.state_graph.x[i, Indices.COORD], coor):
				return True
		return False

	def same_coor_in_target_graph(self, node, coor):
		return torch.equal(self.target_graph.x[node, Indices.COORD], coor)

	def check_loop_by_replacing_edge(self, start_node, prev_target, target_node):
		edge_index = self.state_graph.edge_index.clone()
		if prev_target is not None:
			mask = ~((edge_index[0] == start_node) & (edge_index[1] == prev_target))
			edge_index = edge_index[:, mask]
		new_edge = torch.tensor([[start_node], [target_node]], dtype=torch.long)
		edge_index = torch.cat([edge_index, new_edge], dim=1)
		return has_cycles(Data(x=self.state_graph.x, edge_index=edge_index))

	def render(self, with_target=True, figsize=(2.5, 2.5), return_fig=False):
		if with_target:
			fig, ax = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))
			plot_graph(self.state_graph, ax=ax[0], figsize=figsize, title='State Graph', print_coords=True)
			plot_graph(self.target_graph, ax=ax[1], figsize=figsize, title='Target Graph', print_coords=True)
		else:
			fig, ax = plt.subplots(1, 1, figsize=figsize)
			plot_graph(self.state_graph, ax=ax, figsize=figsize, title='State Graph', print_coords=True)

		if return_fig:
			plt.close()
			return fig
		else:
			plt.show()

	def reset(self, state_graph=None, target_graph=None, seed=None):
		super().reset(seed=seed)
		self.steps = 0
		self.state_graph = self.create_graph() if state_graph is None else copy_graph(state_graph)
		self.initial_graph = copy_graph(self.state_graph)
		if target_graph is None:
			self.target_graph = self.create_graph()
			while torch.equal(self.state_graph.x, self.target_graph.x):
				self.target_graph = self.create_graph()
		else:
			self.target_graph = copy_graph(target_graph)

		info = {}
		return self._get_obs(), info

	def _get_obs(self):
		raise NotImplementedError


class GraphEnv(BaseGraphEnv):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.node_features = self.create_graph().x.shape[1]

	def remove_edge(self, node1, node2):
		mask = ~((self.state_graph.edge_index[0] == node1) & (self.state_graph.edge_index[1] == node2))
		self.state_graph.edge_index = self.state_graph.edge_index[:, mask]
		self.state_graph.x[node1][Indices.RELATION.start+node2] = 0

	def add_edge(self, node1, node2):
		new_edge = torch.tensor([[node1], [node2]], dtype=torch.long)
		self.state_graph.edge_index = torch.cat([self.state_graph.edge_index, new_edge], dim=1)
		self.state_graph.x[node1][Indices.RELATION.start+node2] = 1
		destination = self.state_graph.x[node2, Indices.COORD].clone()
		return self.move_node_with_parents(node1, destination)

	def decode_action(self, action):
		n = self.state_graph.num_nodes
		m = self.grid_size**2
		num_edge_actions = n * (n - 1)
		coordinates = 0

		if action < num_edge_actions:
			action_type = 'on'
			start_node = action // (n - 1)
			target_node = action % (n - 1)
			if target_node >= start_node:
				target_node += 1
		else:
			action_type = 'move'
			adjusted_action = action - num_edge_actions
			start_node = adjusted_action // m
			coordinates = adjusted_action % m
			target_node = start_node

		return action_type, start_node, target_node, coordinates

	def encode_action(self, action_type, start_node, target_node, coordinates):
		if isinstance(coordinates, torch.Tensor):
			coordinates = coordinates[0] * self.grid_size + coordinates[1]
			coordinates = int(coordinates.item())
		elif isinstance(coordinates, list) or isinstance(coordinates, np.ndarray):
			coordinates = coordinates[0] * self.grid_size + coordinates[1]
			coordinates = int(coordinates)

		n = self.state_graph.num_nodes
		m = self.grid_size**2
		num_edge_actions = n * (n - 1)

		if action_type == 'on':
			if target_node >= start_node:
				target_node -= 1
			action = start_node * (n - 1) + target_node
		elif action_type == 'move':
			action = num_edge_actions + start_node * m + coordinates
		else:
			raise ValueError('Invalid action type')

		return action

	def move_node_with_parents(self, node, coordinates):
		reward = 0
		n = self.state_graph.num_nodes
		# Find all parents of the node and their parents recursively and move them
		visited = [False] * n
		visited[node] = True
		stack = [node]
		while len(stack) > 0:
			node = stack.pop()
			prev_coordinates = self.state_graph.x[node, Indices.COORD].clone()
			self.state_graph.x[node, Indices.COORD] = coordinates.clone()
			if not torch.equal(prev_coordinates, coordinates):
				if self.same_coor_in_target_graph(node, prev_coordinates):
					reward -= 1
				elif self.same_coor_in_target_graph(node, coordinates):
					reward += 1
			parents = get_parents(node, self.state_graph.edge_index)
			for i in parents:
				if not visited[i]:
					stack.append(i)
					visited[i] = True
		return reward

	def step_move(self, node, coordinates, log=True):
		coordinates = coordinates[0] * self.grid_size + coordinates[1]
		action = self.encode_action('move', node, node, coordinates)
		self.step(action, log)

	def step_on(self, start_node, target_node, log=True):
		action = self.encode_action('on', start_node, target_node, 0)
		self.step(action, log)

	def step(self, action, log=False):
		action_type, start_node, target_node, coordinates = self.decode_action(action)
		coordinates = torch.tensor([coordinates // self.grid_size, coordinates % self.grid_size], dtype=torch.float32)

		reward, truncated, terminated = -1, False, False

		self.steps += 1
		if self.steps >= self.max_steps:
			truncated = True

		if action_type == 'move':
			if self.is_coor_occupied(coordinates):
				pass
			else:
				prev_target = self.find_connected_node(start_node)
				if prev_target is not None:
					self.remove_edge(start_node, prev_target)
					if self.is_in_target_graph(start_node, prev_target):
						reward -= 1
					else:
						reward += 1
				reward += self.move_node_with_parents(start_node, coordinates)
		elif action_type == 'on':
			if self.is_in_state_graph(target_node, start_node):
				pass
			else:
				prev_target = self.find_connected_node(start_node)
				if prev_target == target_node:
					pass
				elif self.check_loop_by_replacing_edge(start_node, prev_target, target_node):
					pass
				elif prev_target is None:
					reward += self.add_edge(start_node, target_node)
					if self.is_in_target_graph(start_node, target_node):
						reward += 1
				else:
					self.remove_edge(start_node, prev_target)
					reward += self.add_edge(start_node, target_node)
					if self.is_in_target_graph(start_node, prev_target):
						reward -= 1
					elif self.is_in_target_graph(start_node, target_node):
						reward += 1

		# normalize reward in [-1, 0]
		# min_reward = -5   # -time - num_nodes
		# max_reward = 0   # -time + 1
		# reward = (reward - min_reward) / (max_reward - min_reward) - 1

		if self.is_terminal_state():
			terminated = True
			reward = 100

		if log:
			if action_type == 'move':
				print(f'Moved {start_node} to: {coordinates.numpy()} | reward: {reward:.3f} | done: {terminated or truncated}')
			else:
				print(f'{start_node} -> {target_node} | reward: {reward:.3f} | done: {terminated or truncated}')

		info = {}
		return self._get_obs(), reward, terminated, truncated, info
