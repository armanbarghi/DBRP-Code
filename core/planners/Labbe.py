import torch
import time
import numpy as np
from core.planners.utils import BaseSearch, copy_state, reconstruct_path
from core.env.scene_manager import (
	Indices, copy_state,
    get_object_below, get_object_above, get_object_base, build_parent_of
)
from typing import List, Optional, Dict
from tqdm import tqdm


class LabbeNode:
	node_counter = 0  # Static variable to assign unique IDs to each node

	def __init__(self, 
			state: Dict[str, torch.Tensor], 
			remaining_objs: List[int], 
			obj: Optional[int]=None, 
			parent: Optional[int]=None, 
			action: Optional[int]=None, 
			c: float=1., 
			depth: int=0
		):
		self.state = state
		self.obj = obj
		self.parent = parent
		self.action = action
		self.children = {}
		self.n = 0
		self.w = 0.0
		self.c = c
		self.remaining_objs = remaining_objs
		self.depth = depth
		
		# Assign a unique ID to each node
		self.id = LabbeNode.node_counter
		LabbeNode.node_counter += 1

	def get_state(self) -> Dict[str, torch.Tensor]:
		return copy_state(self.state)

	def is_fully_expanded(self) -> bool:
		return len(self.remaining_objs) == 0

	def ucb(self) -> float:
		expected_value = self.w / self.n
		exploration_term = self.c * np.sqrt(2 * np.log(self.parent.n) / self.n)

		return expected_value + exploration_term

class LabbeMCTS(BaseSearch):
	def __init__(self, env):
		super().__init__(env, LabbeNode)

	def get_remaining_objs(self, state: Dict[str, torch.Tensor]) -> List[int]:
		raise NotImplementedError

	def evaluate_state(self, state: Dict[str, torch.Tensor]) -> float:
		raise NotImplementedError

	def get_action_move_obj_away(self, k: int) -> Optional[int]:
		raise NotImplementedError

	def get_motion(self, k: int) -> Optional[int]:
		raise NotImplementedError

	def select(self, node):
		# Accesses child nodes for best selection
		return max(node.children.values(), key=lambda child: child.ucb())

	def expand(self, node):
		# Prevents further expansion if no actions remain
		if node.is_fully_expanded():
			return None

		# If the node is too deep, we stop expanding to avoid infinite loops
		if node.depth > 2*self.env.N + 2:
			return None

		action = self.get_motion(node.remaining_objs.pop())
		if action is None:
			return self.expand(node)

		action_type, start_obj, target_obj, coord = self.env.decode_action(action)

		# Continue expanding if the last changed obj is the same as the current obj
		# if node.obj == start_obj:
		# 	return self.expand(node)

		_, child_state = self.env._step(action_type, start_obj, target_obj, coord)

		if torch.equal(child_state['current'], node.state['current']):
			raise ValueError('State has not changed')

		child_node = self.node_class(
			state=child_state, 
			remaining_objs=self.get_remaining_objs(child_state), 
			obj=start_obj, 
			parent=node, 
			action=action, 
			c=node.c,
			depth=node.depth+1
		)
		node.children[action] = child_node

		return child_node

	def backup_search(self, node, value: float):
		while node is not None:
			node.n += 1
			node.w += value
			node = node.parent

	def print_tree(self, node, depth: int=0):
		# Print the current node with indentation to show its depth in the tree
		indent = "    " * depth  # Four spaces per level of depth
		if node.id == 0:
			print(f"Root | Visits: {node.n} | Value: {node.w:.2f}")
		else:
			print(f"{indent}ID: {node.id} | Action: {node.action} | "
					f"Visits: {node.n} | Value: {node.w:.2f}")

		# Sort children by value estimate
		children = sorted(node.children.values(), key=lambda child: child.w / child.n if child.n > 0 else float('inf'))

		# Recurse on children
		for child in children:
			self.print_tree(child, depth + 1)

	def loop(self):
		node = self.root_node

		# Selection
		while node.is_fully_expanded():
			node = self.select(node)

		# Expansion
		self.env.set_state(node.get_state())
		child_node = self.expand(node)
		if child_node is not None:
			node = child_node
			if self.env.is_terminal_state(node.state):
				self.terminal_node = node
				return
			value = self.evaluate_state(node.state)
		else:
			# It means it fully expanded
			while len(node.children) == 0 and node.parent:
				node.parent.children.pop(node.action)
				node = node.parent
			value = 0

		# Backpropagation
		self.backup_search(node, value)

	def _solve(self, c: float=0.1, verbose: int=0, time_limit: int=1000):
		start_time = time.time()
		self.terminal_node = None
		LabbeNode.node_counter = 0
		steps = 0

		if self.env.is_terminal_state():
			raise ValueError('The initial scene is already in the target state.')

		root_state = self.env.get_state()
		self.root_node = self.node_class(
			state=root_state, 
			remaining_objs=self.get_remaining_objs(root_state), 
			c=c
		)

		if verbose > 0:
			pbar = tqdm(total=None, unit='iterations')

		while self.terminal_node is None:
			# Check if the elapsed time has exceeded the limit
			if time.time()-start_time > time_limit:
				if verbose > 0:
					pbar.close() # type: ignore
				return None, steps, time.time()-start_time

			steps += 1
			self.loop()
			if verbose > 0:
				pbar.update(1) # type: ignore
			
			if len(self.root_node.children) == 0:
				# print('No more actions to expand for root node')
				if verbose > 0:
					pbar.close() # type: ignore
				return None, steps, time.time()-start_time

		if verbose > 0:
			pbar.close() # type: ignore

		return reconstruct_path(self.terminal_node), steps, time.time()-start_time

class Labbe(LabbeMCTS):
	def get_remaining_objs(self, state: Dict[str, torch.Tensor]) -> List[int]:
		"""
		Objects whose current center ≠ target center.
		"""
		current_x, target_x = state['current'], self.env.target_x

		# All current vs. target centers: [N,2]
		cur_centers = current_x[:, Indices.COORD]
		tgt_centers = target_x[:, Indices.COORD]

		# Base condition for objs that should be on the table
		satisfied = (cur_centers == tgt_centers).all(dim=1)  # [N]

		# Remaining = those not satisfied
		rem = torch.nonzero(~satisfied, as_tuple=False).view(-1)  # [R]

		if rem.numel() == 0:
			return []

		# Shuffle the remaining indices
		perm = torch.randperm(rem.size(0))
		return rem[perm].tolist()

	def evaluate_state(self, state: Dict[str, torch.Tensor]) -> float:
		"""
		Fraction of objects exactly at their target centers.
		"""
		current_x, target_x = state['current'], self.env.target_x

		cur_centers = current_x[:, Indices.COORD]       
		tgt_centers = target_x[:, Indices.COORD]  

		satisfied = (cur_centers == tgt_centers).all(dim=1)  # [N]
		return satisfied.float().mean().item()

	def get_action_move_obj_away(self, k: int) -> Optional[int]:
		"""
		Moves object k to it's target position,
		o.w. moves it randomly.
		"""
		# Is target position free?
		TK = self.env.target_x[k, Indices.COORD]
		if not self.env.is_invalid_center(TK, k):
			return self.env.encode_move(k, TK)

		# Move away k
		free_positions = self.env.get_empty_positions(ref_obj=k, n=1)
		if free_positions.numel() > 0:
			# move to a random position
			return self.env.encode_move(k, free_positions[0])
		return None

	def get_motion(self, k: int) -> Optional[int]:
		"""
		Move object k to its target position if it was free,
		o.w. move away one of its blocking objects
		"""
		CK = self.env.current_x[k, Indices.COORD]
		TK = self.env.target_x[k, Indices.COORD]
		if torch.equal(TK, CK):
			raise ValueError(f'The obj {k} is already in its target position')
		
		# Is target position free?
		if not self.env.is_invalid_center(TK, k):
			return self.env.encode_move(k, TK)

		# Move away one of its blocking objects
		occupants = self.env.find_blocking_objects(k)
		if len(occupants) == 0:
			raise ValueError('No obj is occupying the target position')
		for j in occupants:
			action_away = self.get_action_move_obj_away(j)
			if action_away is not None:
				return action_away
		return None

	def solve(self, c: float=0.1, verbose: int=0, time_limit: int=1000):
		if torch.sum(self.env.initial_x[:, Indices.RELATION]) > 0:
			raise ValueError('Initial scene has stacks in Non-stack mode')
		if torch.sum(self.env.current_x[:, Indices.RELATION]) > 0:
			raise ValueError('Current scene has stacks in Non-stack mode')
		if torch.sum(self.env.target_x[:, Indices.RELATION]) > 0:
			raise ValueError('Target scene has stacks in Non-stack mode')
		return self._solve(c, verbose, time_limit)

class Labbe_S(LabbeMCTS):
	def get_remaining_objs(self, state: Dict[str, torch.Tensor]) -> List[int]:
		"""
		• Objects whose aren't stacked on their target objects.
		• Base objects whose current center ≠ target center.
		"""
		current_x, target_x = state['current'], self.env.target_x

		# Build parent‐of maps once
		cur_parent = build_parent_of(current_x)

		# Stacking condition for objs that should be stacked
		cond_stacked = (self.tgt_parent >= 0) & (cur_parent == self.tgt_parent)

		# Base condition for objs that should be on the table
		cur_centers = current_x[:, Indices.COORD]
		tgt_centers = target_x[:, Indices.COORD]
		base_match = (cur_centers == tgt_centers).all(dim=1)  # [N]
		cond_base    = (self.tgt_parent < 0) & (cur_parent < 0) & base_match

		# Satisfied = either stacking OK or base OK
		satisfied = cond_stacked | cond_base       # [N]

		# Remaining = those not satisfied
		rem = torch.nonzero(~satisfied, as_tuple=False).view(-1)  # [R]

		if rem.numel() == 0:
			return []

		# Shuffle the remaining indices
		perm = torch.randperm(rem.size(0))
		return rem[perm].tolist()

	def evaluate_state(self, state: Dict[str, torch.Tensor]) -> float:
		"""
		Fraction of nodes satisfying the same two conditions:
		• stacked correctly, or
		• matched centers of base objects.
		"""
		current_x, target_x = state['current'], self.env.target_x

		cur_parent = build_parent_of(current_x)

		cur_centers = current_x[:, Indices.COORD]
		tgt_centers = target_x[:, Indices.COORD]
		base_match = (cur_centers == tgt_centers).all(dim=1)  # [N]

		cond_stacked = (self.tgt_parent >= 0) & (cur_parent == self.tgt_parent)
		cond_base    = (self.tgt_parent < 0) & (cur_parent < 0) & base_match

		satisfied = cond_stacked | cond_base
		return satisfied.float().mean().item()

	def get_action_move_obj_away(self, k: int) -> Optional[int]:
		"""
		Moves object k to it's target position or target stack,
		o.w. moves it randomly or stacks it randomly.
		"""
		# Non-empty objects cannot be moved in static_stack mode
		if self.static_stack:
			j = get_object_above(self.env.current_x, k)
			if j is not None:
				return None

		i = get_object_below(self.env.target_x, k)
		if i is not None:
			# Is target object empty?
			if get_object_above(self.env.current_x, i) is None:
				return self.env.encode_stack(k, i)
		else:
			# Is target position free?
			TK = self.env.target_x[k, Indices.COORD]
			if not self.env.is_invalid_center(TK, k):
				return self.env.encode_move(k, TK)

		# Move away k
		free_positions = self.env.get_empty_positions(ref_obj=k, n=1)
		free_objects = self.env.get_empty_objs(ref_obj=k, n=1)

		if len(free_objects) > 0 and free_positions.numel() > 0:
			if np.random.rand() < 0.5:
				# move to a random position
				return self.env.encode_move(k, free_positions[0])
			else:
				# stack on a random object
				return self.env.encode_stack(k, free_objects[0])
		elif len(free_objects) == 0 and free_positions.numel() > 0:
			# move to a random position
			return self.env.encode_move(k, free_positions[0])
		elif free_positions.numel() == 0 and len(free_objects) > 0:
			# stack on a random object
			return self.env.encode_stack(k, free_objects[0])
		return None

	def get_motion(self, k: int) -> Optional[int]:
		"""
		Moves object k to it's target position or target stack,
		o.w. move away one of it's blocking objects.
		"""
		# Non-empty objects cannot be moved in static_stack mode
		if self.static_stack:
			j = get_object_above(self.env.current_x, k)
			if j is not None:
				return self.get_action_move_obj_away(j)

		i = get_object_below(self.env.target_x, k)
		if i is not None:
			# Is target object empty?
			j = get_object_above(self.env.current_x, i)
			if j is not None:
				return self.get_action_move_obj_away(j)
			return self.env.encode_stack(k, i)

		CK = self.env.current_x[k, Indices.COORD]
		TK = self.env.target_x[k, Indices.COORD]
		if torch.equal(TK, CK):
			j = get_object_below(self.env.current_x, k)
			if j is None:
				raise ValueError(f'The obj {k} is already in its target position')

			if self.static_stack:
				return self.get_action_move_obj_away(k)
			else:
				j = get_object_base(self.env.current_x, j)
				return self.get_action_move_obj_away(j)

		# Is target position free?
		if not self.env.is_invalid_center(TK, k):
			return self.env.encode_move(k, TK)

		# Move away one of its blocking objects
		blockers = self.env.find_blocking_objects(k)
		if len(blockers) == 0:
			raise ValueError(f'No obj is blocking the target position of {k}')
		for j in blockers:
			action_away = self.get_action_move_obj_away(j)
			if action_away is not None:
				return action_away
		return None

	def solve(self, c: float=0.1, static_stack: bool=False, verbose: int=0, time_limit: int=1000):
		self.static_stack = static_stack
		self.env.static_stack = static_stack
		self.tgt_parent = build_parent_of(self.env.target_x)
		return self._solve(c, verbose, time_limit)
