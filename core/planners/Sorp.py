import torch
import time
import numpy as np
from tqdm import tqdm
from typing import Union, List, Tuple, Optional, Dict
from core.planners.utils import BaseSearch, copy_state, reconstruct_path
from core.planners.Labbe import Labbe_S
from core.env.scene_manager import Indices, build_parent_of


class MctsNode:
	node_counter = 0  # Static variable to assign unique IDs to each node

	def __init__(self, state, valid_actions, parent=None, action=None, cost=0.0, cost_to_come=0.0, c=1, depth=0):
		self.state = state
		self.parent = parent
		self.action = action
		self.children = {}
		self.n = 0
		self.w = np.inf
		self.c = c
		self.cost = cost
		self.cum_cost = cost_to_come
		self.unexpanded_actions = valid_actions
		self.depth = depth
		
		# Assign a unique ID to each node
		self.id = MctsNode.node_counter
		MctsNode.node_counter += 1

	def get_state(self):
		return copy_state(self.state)

	def is_fully_expanded(self):
		return len(self.unexpanded_actions) == 0

	def uct(self, c_min=0, c_max=1):
		n = self.n

		# expected_value = ( (self.cum_cost + self.w / n) - c_min ) / (c_max - c_min)
		expected_value = ( (self.cum_cost + self.w) - c_min ) / (c_max - c_min)
		exploration_term = np.sqrt(2 * np.log(self.parent.n) / n)

		return expected_value - self.c * exploration_term  # Minimization form

class Sorp(BaseSearch):
	def __init__(self, env):
		super().__init__(env, MctsNode)
	
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

	def get_valid_actions(self, state: Dict[str, torch.Tensor]) -> List[int]:
		"""
		...
		"""
		# restore the env to this state
		self.env.set_state(copy_state(state))

		# which objects remain to be placed?
		rem = self.get_remaining_objs(state)

		# Which k are allowed (static_stack skips non‐empty actors)
		if self.static_stack:
			rem = torch.tensor(rem, dtype=torch.long)
			rel     = self.env.current_x[:, Indices.RELATION]
			empty_k = ~rel.any(dim=0)                  # True if k has no one on top
			mask = empty_k[rem]
			ks = rem[mask].tolist()
		else:
			ks = rem

		valid_actions = []
		stack_nums = max(int(0.6 * self.num_buffers), 1)

		for k in ks:
			valid_stacks = []
			empty_objs = self.env.get_empty_objs(ref_obj=k, n=stack_nums)
			if len(empty_objs) > 0:
				M      = len(empty_objs)
				starts = torch.full((M,), k, dtype=torch.long)
				targets= torch.tensor(empty_objs, dtype=torch.long)
				valid_stacks = self.env.encode_stack(starts, targets).tolist()

			valid_moves = []
			coords = self.env.get_empty_positions_with_target(
				ref_obj=k,
				n=self.num_buffers-len(valid_stacks),
				sort=self.score_sorting
			)

			if coords.numel() > 0:
				M      = coords.size(0)
				starts = torch.full((M,), k, dtype=torch.long)
				valid_moves  = self.env.encode_move(starts, coords).tolist()

			valid_actions += valid_stacks + valid_moves

		return valid_actions

	def select(self, node):
		# Accesses child nodes for best selection
		if self.c_min == np.inf or self.c_max == self.c_min:
			return min(node.children.values(), key=lambda child: child.uct())
		else:
			return min(node.children.values(), key=lambda child: child.uct(self.c_min, self.c_max))

	def expand(self, node):
		if node.is_fully_expanded():
			return None  # Prevents further expansion if no actions remain

		action = node.unexpanded_actions.pop()
		self.env.set_state(node.get_state())
		action_type, start_obj, target_obj, coordinates = self.env.decode_action(action)

		# Continue expanding if the last changed node is the same as the current node
		if node.action is not None:
			_, last_obj, _, _ = self.env.decode_action(node.action)
			if last_obj == start_obj:
				return self.expand(node)

		cost, child_state = self.env._step(action_type, start_obj, target_obj, coordinates)

		# Continue expanding if the state hasn't changed
		if torch.equal(child_state['current'], node.state['current']):
			raise ValueError('State has not changed')

		child_node = self.node_class(
			state=child_state, 
			valid_actions=self.get_valid_actions(self.env.get_state()), 
			parent=node, 
			action=action, 
			cost=cost, 
			cost_to_come=cost+node.cum_cost,
			c=node.c, 
			depth=node.depth+1
		)
		node.children[action] = child_node

		return child_node

	def rollout_one(self, node):
		costs = 0

		self.env.set_state(node.get_state())
		sim_time_limit = (self.time_limit - time.time() + self.start_time) / 4
		feasible_path, steps, _ = Labbe_S(self.env).solve(time_limit=sim_time_limit, static_stack=self.static_stack)
		self.env.set_state(node.get_state())

		if feasible_path:
			for i, action in enumerate(feasible_path):
				cost, child_state = self.env.step(action)
				costs += cost
				if i == 0:
					# remove action from the node's unexpanded actions
					if action in node.unexpanded_actions:
						node.unexpanded_actions.remove(action)
					# add the new child to the node
					child_node = self.node_class(
						state=child_state, 
						valid_actions=self.get_valid_actions(child_state), 
						parent=node, 
						action=action, 
						cost=cost, 
						cost_to_come=cost+node.cum_cost,
						c=node.c, 
						depth=node.depth+1
					)
					node.children[action] = child_node
					node = child_node

		return costs, steps, feasible_path, node

	def rollout_whole(self, node):
		costs = 0

		self.env.set_state(node.get_state())
		sim_time_limit = (self.time_limit - time.time() + self.start_time) / 4
		feasible_path, steps, _ = Labbe_S(self.env).solve(time_limit=sim_time_limit, static_stack=self.static_stack)
		self.env.set_state(node.get_state())

		if feasible_path:
			for action in feasible_path:
				cost, child_state = self.env.step(action)
				costs += cost
				# remove action from the node's unexpanded actions
				if action in node.unexpanded_actions:
					node.unexpanded_actions.remove(action)
				# add the new child to the node
				child_node = self.node_class(
					state=child_state, 
					valid_actions=self.get_valid_actions(child_state), 
					parent=node, 
					action=action, 
					cost=cost, 
					cost_to_come=cost+node.cum_cost,
					c=node.c, 
					depth=node.depth+1
				)
				node.children[action] = child_node
				node = child_node

		return costs, steps, feasible_path, node

	def rollout(self, node):
		if self.one_step:
			return self.rollout_one(node)
		else:
			return self.rollout_whole(node)

	def backup_search(self, node, value):
		while node is not None:
			node.n += 1
			# node.w += value
			node.w = min(node.w, value)
			value += node.cost
			node = node.parent

	def print_tree(self, node, depth=0, max_depth=float('inf'), ter=False):
		if depth >= max_depth:
			return

		# Print the current node with indentation to show its depth in the tree
		indent = "    " * depth  # Four spaces per level of depth
		if node.id == 0:
			print(f"Root | n: {node.n} | w: {node.w:.2f}")
		else:
			n = node.n
			if self.c_max == self.c_min:
				# expected_value = node.cum_cost + node.w / n
				expected_value = node.cum_cost + node.w
			else:
				# expected_value = ( (node.cum_cost + node.w / n) - self.c_min ) / (self.c_max - self.c_min)
				expected_value = ( (node.cum_cost + node.w) - self.c_min ) / (self.c_max - self.c_min)
			exploration_term = np.sqrt(2 * np.log(node.parent.n) / n)

			print(f"{indent}ID: {node.id} | a: {node.action} | c: {node.cost:.2f} | ctc: {node.cum_cost:.2f} | "
					f"n: {node.n} | w: {node.w:.2f} | expe: {expected_value:.4f} | expl: {exploration_term:.4f}")

		# Sort children by w estimate
		if ter:
			accepted_children = [child for child in node.children.values()]
			children = sorted(accepted_children, key=lambda child: child.w)
		else:
			children = sorted(node.children.values(), key=lambda child: child.w)

		# Recurse on children
		for child in children:
			self.print_tree(child, depth + 1, max_depth, ter)

	def find_best_path(self):
		# self.print_tree(self.root_node, max_depth=5)
		if self.best_plan is None or self.best_plan[1] is None:
			return None
		return reconstruct_path(self.best_plan[0]) + self.best_plan[1]

	def loop(self):
		node = self.root_node

		# Selection
		self.env.set_state(node.get_state())
		while node.is_fully_expanded() and not self.env.is_terminal_state():
			node = self.select(node)
			self.env.set_state(node.get_state())

		# Expansion
		if not self.env.is_terminal_state():
			child_node = self.expand(node)
			if child_node is None:
				while len(node.children) == 0 and node.is_fully_expanded():
					node.parent.children.pop(node.action)
					node = node.parent
					print('oooooooooooooooooooo laaaaaaaa laaaaaaaaaaaaa')
				return 1
			node = child_node

		# Simulation (Rollout)
		self.env.set_state(node.get_state())
		steps = 0
		if self.env.is_terminal_state():
			value = 0
		else:
			c_rollout, steps, feasible_plan, child_node = self.rollout(node)

			if feasible_plan is None:
				if node.parent is None:
					return -1
				node.parent.children.pop(node.action)
				node = node.parent
				while len(node.children) == 0 and node.is_fully_expanded():
					node.parent.children.pop(node.action)
					node = node.parent
					if node.parent is None:
						return -1
				return steps

			new_cost = c_rollout + node.cum_cost
			self.c_max = max(self.c_max, new_cost)
			if new_cost < self.c_min:
				self.best_plan = (node, feasible_plan)
				self.c_min = new_cost
			
			node = child_node
			if self.one_step:
				value = c_rollout - node.cost
				# value = c_rollout
			else:
				value = 0

		# Backpropagation
		self.backup_search(node, value)

		return steps

	def solve(
			self, iterations: int=1000, 
			num_buffers: int=4, score_sorting: bool=False,
			c: float=1, verbose: int=0, 
			one_step: bool=True, static_stack: bool=False, 
			time_limit: int=1000
		):
		self.start_time = time.time()
		self.static_stack = static_stack
		self.score_sorting = score_sorting
		self.tgt_parent = build_parent_of(self.env.target_x)

		MctsNode.node_counter = 0
		self.num_buffers = num_buffers
		self.time_limit = time_limit
		self.one_step = one_step
		self.c_max = -np.inf
		self.c_min = np.inf
		self.best_plan = None
		window_last_values = []
		self.root_node = self.node_class(
			state=self.env.get_state(), 
			valid_actions=self.get_valid_actions(self.env.get_state()), 
			c=c
		)
		
		steps, iteration = 0, 0
		if verbose > 0:
			pbar = tqdm(total=None, unit='iterations')
		
		while iteration < iterations:
			# Check if the elapsed time has exceeded the limit
			if time.time()-self.start_time > time_limit:
				if verbose > 0:
					pbar.close()
				# print('Time limit exceeded')
				return self.find_best_path(), steps, time.time()-self.start_time
			
			iteration += 1
			step = self.loop()
			if step == -1:
				return self.find_best_path(), steps, time.time()-self.start_time
			
			steps += step
			if verbose > 0:
				pbar.update(1)
			
			# if iteration != 0 and iteration % 10 == 0:
			# 	v_root = self.root_node.w
			# 	print(f'v_root: {v_root:.3f} | c_min: {self.c_min:.3f} | c_max: {self.c_max:.3f}')
			# 	self.print_tree(self.root_node, max_depth=3)
			# 	window_last_values.append(self.c_min)
			# 	if len(window_last_values) > 5:
			# 		window_last_values.pop(0)
			# 		if len(set(window_last_values)) == 1:
			# 			break

		if verbose > 0:
			pbar.close()

		return self.find_best_path(), steps, time.time()-self.start_time

# evaluate_alg(
# 	env, Sorp, initial_scene, target_scene, 
# 	num_runs=1, score_sorting=False,
# 	iterations=1000, num_buffers=4, one_step=True,
# 	c=0.5, time_limit=20, verbose=1
# )
