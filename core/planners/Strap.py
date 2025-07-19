import time
import torch
import heapq
from typing import Union, List, Tuple, Optional, Dict
from core.planners.planning_utils import BaseSearch, reconstruct_path
from core.env.scene_manager import (
	Indices, copy_state, state_to_hashable, build_parent_of
)
from core.planners.Labbe import Labbe, Labbe_S


class AstarNode:
	def __init__(self, 
			state: Dict[str, torch.Tensor], 
			parent: Optional[int]=None, 
			action: Optional[int]=None, 
			g_cost: float=0.0, 
			h_cost: float=0.0, 
			depth: int=0
		):
		self.state = state
		self.parent = parent
		self.action = action
		self.g_cost = g_cost	# cost-to-come
		self.h_cost = h_cost	# cost-to-go
		self.total_cost = self.g_cost + self.h_cost
		self.depth = depth

	def __lt__(self, other):
		return self.total_cost < other.total_cost  # Lower cost first

	def get_state(self) -> Dict[str, torch.Tensor]:
		return copy_state(self.state)

class Astar(BaseSearch):
	def __init__(self, env):
		super().__init__(env, AstarNode)
	
	def get_remaining_objs(self, state: Dict[str, torch.Tensor]) -> List[int]:
		raise NotImplementedError
	
	def get_valid_actions(self, state: Dict[str, torch.Tensor]) -> List[int]:
		raise NotImplementedError
	
	def evaluate_state(self, state: Dict[str, torch.Tensor]) -> float:
		raise NotImplementedError
	
	def _solve(self, time_limit: int=1000):
		start_time = time.time()
		
		steps = 0
		root_state = self.env.get_state()
		root_node = self.node_class(
			state=root_state, 
			g_cost=0, 
			h_cost=self.evaluate_state(root_state)
		)

		self.queue = []
		heapq.heappush(self.queue, root_node)
		visited = {}

		while self.queue:
			current_node = heapq.heappop(self.queue)

			# Check if the current node's state matches the target state
			if self.env.is_terminal_state(current_node.state):
				return reconstruct_path(current_node), steps, time.time()-start_time

			# Check if the elapsed time has exceeded the limit
			if time.time()-start_time > time_limit:
				# print('Time limit exceeded')
				return None, steps, time.time()-start_time

			last_obj = self.env.decode_action(current_node.action)[1] if current_node.action is not None else None
			for action in self.get_valid_actions(current_node.state):
				action_type, start_obj, target_obj, coordinates = self.env.decode_action(action)

				# If the last changed obj is the same as the current obj, continue
				if start_obj == last_obj:
					continue

				steps += 1
				self.env.set_state(current_node.get_state())
				cost, child_state = self.env._step(action_type, start_obj, target_obj, coordinates)

				# If state hasn't changed, continue
				if torch.equal(child_state['current'], current_node.state['current']):
					raise ValueError('State has not changed')

				child_hash = state_to_hashable(child_state)

				# Calculate the accumulated cost for the current path
				new_g_cost = current_node.g_cost + cost
				h_cost = self.evaluate_state(child_state)
				new_total_cost = new_g_cost + h_cost

				# Retain the node with better cost
				if child_hash not in visited or visited[child_hash] > new_total_cost:
					visited[child_hash] = new_total_cost
					child_node = self.node_class(
						state=child_state, 
						parent=current_node, 
						action=action, 
						g_cost=new_g_cost, 
						h_cost=h_cost,
						depth=current_node.depth+1
					)

					heapq.heappush(self.queue, child_node)

		return None, steps, time.time()-start_time


class Strap(Astar):
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

	def get_valid_actions(self, state: Dict[str, torch.Tensor]) -> List[int]:
		"""
		For each object k that’s not yet in place,
		gather up to self.num_buffers candidate coords
		(including its target if free), and batch‐encode
		all those 'move' actions at once.
		"""
		# restore the env to this state
		self.env.set_state(copy_state(state))

		# which objects remain to be placed?
		rem = self.get_remaining_objs(state)

		# for each remaining object, batch‐fetch positions & encode
		valid_actions = []
		for k in rem:
			# coords: Tensor of shape [M,2], dtype long
			coords = self.env.get_empty_positions_with_target(
				ref_obj=k,
				n=self.num_buffers,
				sort=self.score_sorting
			)

			if coords.numel() == 0:
				continue

			# build batched start/target vectors of length M
			M      = coords.size(0)
			starts = torch.full((M,), k, dtype=torch.long)

			# vectorized call: returns LongTensor[M]
			codes  = self.env.encode_move(starts, coords)
			valid_actions.append(codes)

		if len(valid_actions) == 0:
			return []

		# concatenate all batches and return Python ints
		return torch.cat(valid_actions).tolist()

	def evaluate_state(self, state: Dict[str, torch.Tensor]) -> float:
		"""
		Heuristic = sum over all remaining k of:
		pp_cost + normalized_distance( current_pos[k], target_pos[k] ).
		"""
		current_x, target_x = state['current'], self.env.target_x

		# 1) remaining object indices [R]
		rem_nodes = torch.tensor(self.get_remaining_objs(state), dtype=torch.long)
		if rem_nodes.numel() == 0:
			return 0.0

		# 2) Gather current and target centers for those nodes: [R,2]
		cur_ctr = current_x[rem_nodes, Indices.COORD].float()
		tgt_ctr = target_x[rem_nodes, Indices.COORD].float()

		# 3) Euclidean distances [R]
		dists   = torch.cdist(cur_ctr, tgt_ctr, p=2).diag()  # [R]
		
		# 4) Heuristic = R * pp_cost + sum(dists) * normalization
		R       = float(rem_nodes.size(0))
		pp      = self.env.pp_cost
		norm    = self.env.normalization_factor

		return R * pp + (dists.sum().item() * norm)

	def solve(self, num_buffers: int=3, score_sorting: bool=False, time_limit: int=1000):
		if torch.sum(self.env.initial_x[:, Indices.RELATION]) > 0:
			raise ValueError('Initial scene has stacks in Non-stack mode')
		if torch.sum(self.env.current_x[:, Indices.RELATION]) > 0:
			raise ValueError('Current scene has stacks in Non-stack mode')
		if torch.sum(self.env.target_x[:, Indices.RELATION]) > 0:
			raise ValueError('Target scene has stacks in Non-stack mode')
		
		self.score_sorting = score_sorting
		self.num_buffers = num_buffers
		return self._solve(time_limit)

class StrapGA(Strap):
	def goal_attempt(self, node, time_limit: int) -> int:
		self.env.set_state(node.get_state())
		plan_to_go, steps, _ = Labbe(self.env).solve(time_limit=time_limit)

		# no feasible plan was found in the time_limit
		if plan_to_go is None:
			return steps

		# --- Stage 1: Immediate Redundancy Removal ---
		decoded_plan = []
		for action in plan_to_go:
			decoded_action = (action, self.env.decode_action(action))
			# each entry: (action, (action_type, start_obj, target_obj, coord))
			# remove redundant action on the latest manipulated object
			# This check is for consecutive actions on the SAME object.
			# It keeps the last action on that object and discards previous consecutive ones.
			if len(decoded_plan) > 0 and decoded_action[1][1] == decoded_plan[-1][1][1]:
				decoded_plan[-1] = decoded_action
			else:
				decoded_plan.append(decoded_action)

		# --- Stage 2: State-Checking Redundancy (Simulate and Filter) ---
		# This stage is only required after the immediate redundancy removal
		self.env.set_state(node.get_state())
		feasible_path_cost = node.g_cost
		refined_plan = []

		for i, decoded_action in enumerate(decoded_plan):
			action = decoded_action[0]
			action_type, start_obj, target_obj, coordinates = decoded_action[1]
			
			if action_type == 'stack':
				# If the object is already stacked, skip this action
				if self.env.current_x[start_obj, Indices.RELATION.start + target_obj] == 1:
					continue
			elif action_type == 'move':
				# If the object is ALREADY at the target coordinates for this move action.
				current_coord = self.env.current_x[start_obj, Indices.COORD]
				if torch.equal(current_coord, coordinates):
					continue

			refined_plan.append(action)
			cost, child_state = self.env._step(action_type, start_obj, target_obj, coordinates)

			feasible_path_cost += cost
			if i == 0:
				first_child = child_state
				first_action = action
				first_cost = feasible_path_cost

		# --- Stage 3: add the first child node if the plan is the best so far ---
		if feasible_path_cost < self.best_cost:
			self.best_plan = reconstruct_path(node) + list(refined_plan)
			self.best_cost = feasible_path_cost
			child_node = self.node_class(
				state=first_child,
				parent=node,
				action=first_action,
				g_cost=first_cost,
				h_cost=self.evaluate_state(first_child),
				depth=node.depth+1
			)

			heapq.heappush(self.queue, child_node)

		# Remove all the nodes with their total cost is greater than the feasible path cost
		# for node in self.queue:
		# 	if node.total_cost > feasible_path_cost:
		# 		self.queue.remove(node)

		return steps

	def _solve(self, time_limit: int=1000):
		start_time = time.time()
		self.best_plan = None
		self.best_cost = float('inf')

		steps = 0
		root_state = self.env.get_state()
		root_node = self.node_class(
			state=root_state, 
			g_cost=0, 
			h_cost=self.evaluate_state(root_state)
		)

		self.queue = []
		heapq.heappush(self.queue, root_node)
		visited = {}

		while self.queue:
			current_node = heapq.heappop(self.queue)

			# Check if the current node's state matches the target state
			if self.env.is_terminal_state(current_node.state):
				if current_node.total_cost < self.best_cost:
					return reconstruct_path(current_node), steps, time.time()-start_time
				return self.best_plan, steps, time.time()-start_time

			# Check if the elapsed time has exceeded the limit
			if time.time()-start_time > time_limit:
				# print('Time limit exceeded')
				return self.best_plan, steps, time.time()-start_time

			last_obj = self.env.decode_action(current_node.action)[1] if current_node.action is not None else None
			for action in self.get_valid_actions(current_node.state):
				action_type, start_obj, target_obj, coordinates = self.env.decode_action(action)

				# If the last changed obj is the same as the current obj, continue
				if start_obj == last_obj:
					continue

				steps += 1
				self.env.set_state(current_node.get_state())
				cost, child_state = self.env._step(action_type, start_obj, target_obj, coordinates)

				# If state hasn't changed, continue
				if torch.equal(child_state['current'], current_node.state['current']):
					raise ValueError('State has not changed')

				child_hash = state_to_hashable(child_state)

				# Calculate the accumulated cost for the current path
				new_g_cost = current_node.g_cost + cost
				h_cost = self.evaluate_state(child_state)
				new_total_cost = new_g_cost + h_cost

				# Retain the node with better cost
				if child_hash not in visited or visited[child_hash] > new_total_cost:
					visited[child_hash] = new_total_cost
					child_node = self.node_class(
						state=child_state, 
						parent=current_node, 
						action=action, 
						g_cost=new_g_cost, 
						h_cost=h_cost,
						depth=current_node.depth+1
					)

					heapq.heappush(self.queue, child_node)

			if time.time()-start_time > time_limit:
				# print('Time limit exceeded')
				return self.best_plan, steps, time.time()-start_time

			# Goal Attempting
			sim_time_limit = (time_limit - time.time() + start_time) / 5
			steps += self.goal_attempt(current_node, sim_time_limit)

		return self.best_plan, steps, time.time()-start_time


class Strap_S(Astar):
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

	def evaluate_state2(self, state: Dict[str, torch.Tensor]) -> float:
		"""
		Heuristic for each remaining object k:
			Aₖ = dist(Cₖ → Tₖ) * NF
			Bₖ = min_j [ dist(Cₖ → Cⱼ) + dist(Tⱼ → Tₖ) ] * NF + pp_cost
				(only over supporting j)
		Final h = sum_k ( min(Aₖ, Bₖ) + pp_cost )
		"""
		cur_x, tgt_x  = state['current'], self.env.target_x
		pp_cost, norm = self.env.pp_cost, self.env.normalization_factor

		# Remaining indices
		rem = torch.tensor(self.get_remaining_objs(state), dtype=torch.long)
		if rem.numel() == 0:
			return 0.0

		# Gather current & target centers for remaining: [R,2]
		cur_ctr = cur_x[rem, Indices.COORD].float()  # [R,2]
		tgt_ctr = tgt_x[rem, Indices.COORD].float()  # [R,2]

		# Compute A = direct move distance
		A = torch.cdist(cur_ctr, tgt_ctr, p=2).diag() * norm     # [R]

		# Precompute all object centers
		all_cur = cur_x[:, Indices.COORD].float()    # [N,2]
		all_tgt = tgt_x[:, Indices.COORD].float()    # [N,2]

		# Pairwise distances
		D_c = torch.cdist(cur_ctr, all_cur, p=2)  # cost C_k→C_j: [R,N]
		D_t = torch.cdist(tgt_ctr, all_tgt, p=2)  # cost T_k→T_j: [R,N]

		# Stability mask for rem rows
		S_rem = self.env.stability_mask[rem]      # [R,N]

		# Compute stack‐move costs and invalidate unsupportable pairs
		costs = D_c + D_t                   # [R,N]
		costs[~S_rem] = float('inf')        # forbid non‐stable

		# B = min_j costs[r,j] * norm + pp_cost
		B = costs.min(dim=1).values * norm + pp_cost           # [R]

		# Per‐object best cost = min(A, B) + pp_cost
		best = torch.minimum(A, B) + pp_cost                 # [R]

		# Final h_cost
		return best.sum()

	def evaluate_state(self, state: Dict[str, torch.Tensor]) -> float:
		"""
		Heuristic for each remaining object k:
			Aₖ = dist(Cₖ → Tₖ) * NF
			Bₖ = min_j [ dist(Cₖ → Cⱼ) ] * NF + pp_cost
				(only over supporting j)
		Final h = sum_k ( min(Aₖ, Bₖ) + pp_cost )
		"""
		cur_x, tgt_x  = state['current'], self.env.target_x
		pp_cost, norm = self.env.pp_cost, self.env.normalization_factor

		# Remaining indices
		rem = torch.tensor(self.get_remaining_objs(state), dtype=torch.long)
		if rem.numel() == 0:
			return 0.0

		# Gather current & target centers for remaining: [R,2]
		cur_ctr = cur_x[rem, Indices.COORD].float()  # [R,2]
		tgt_ctr = tgt_x[rem, Indices.COORD].float()  # [R,2]

		# Compute A = direct move distance
		A = torch.cdist(cur_ctr, tgt_ctr, p=2).diag() * norm     # [R]

		# Precompute all object centers
		all_cur = cur_x[:, Indices.COORD].float()    # [N,2]

		# Pairwise distances
		costs = torch.cdist(cur_ctr, all_cur, p=2)  # cost C_k→C_j: [R,N]

		# Stability mask for rem rows
		S_rem = self.env.stability_mask[rem]      # [R,N]

		# Compute stack‐move costs and invalidate unsupportable pairs
		costs[~S_rem] = float('inf')        # forbid non‐stable

		# B = min_j costs[r,j] * norm + pp_cost
		B = costs.min(dim=1).values * norm + pp_cost           # [R]

		# Per‐object best cost = min(A, B) + pp_cost
		best = torch.minimum(A, B) + pp_cost                 # [R]

		# Final h_cost
		return best.sum()

	def solve(self, num_buffers: int=4, score_sorting: bool=False, time_limit: int=1000, static_stack: bool=False):
		self.tgt_parent = build_parent_of(self.env.target_x)
		self.score_sorting = score_sorting
		self.num_buffers = num_buffers
		self.static_stack = static_stack
		self.env.static_stack = static_stack
		return self._solve(time_limit)

class StrapGA_S(Strap_S):
	def goal_attempt(self, node, time_limit: int) -> int:
		self.env.set_state(node.get_state())
		plan_to_go, steps, _ = Labbe_S(self.env).solve(time_limit=time_limit, static_stack=self.static_stack)

		# no feasible plan was found in the time_limit
		if plan_to_go is None:
			return steps

		# --- Stage 1: Immediate Redundancy Removal ---
		decoded_plan = []
		for action in plan_to_go:
			decoded_action = (action, self.env.decode_action(action))
			# each entry: (action, (action_type, start_obj, target_obj, coord))
			# remove redundant action on the latest manipulated object
			# This check is for consecutive actions on the SAME object.
			# It keeps the last action on that object and discards previous consecutive ones.
			if len(decoded_plan) > 0 and decoded_action[1][1] == decoded_plan[-1][1][1]:
				decoded_plan[-1] = decoded_action
			else:
				decoded_plan.append(decoded_action)

		# --- Stage 2: State-Checking Redundancy (Simulate and Filter) ---
		# This stage is only required after the immediate redundancy removal
		self.env.set_state(node.get_state())
		feasible_path_cost = node.g_cost
		refined_plan = []

		for i, decoded_action in enumerate(decoded_plan):
			action = decoded_action[0]
			action_type, start_obj, target_obj, coordinates = decoded_action[1]
			
			if action_type == 'stack':
				# If the object is already stacked, skip this action
				if self.env.current_x[start_obj, Indices.RELATION.start + target_obj] == 1:
					continue
			elif action_type == 'move':
				# If the object is ALREADY at the target coordinates for this move action.
				current_coord = self.env.current_x[start_obj, Indices.COORD]
				if torch.equal(current_coord, coordinates):
					continue

			refined_plan.append(action)
			cost, child_state = self.env._step(action_type, start_obj, target_obj, coordinates)

			feasible_path_cost += cost
			if i == 0:
				first_child = child_state
				first_action = action
				first_cost = feasible_path_cost

		# --- Stage 3: add the first child node if the plan is the best so far ---
		if feasible_path_cost < self.best_cost:
			self.best_plan = reconstruct_path(node) + list(refined_plan)
			self.best_cost = feasible_path_cost
			child_node = self.node_class(
				state=first_child,
				parent=node,
				action=first_action,
				g_cost=first_cost,
				h_cost=self.evaluate_state(first_child),
				depth=node.depth+1
			)

			heapq.heappush(self.queue, child_node)

		# Remove all the nodes with their total cost is greater than the feasible path cost
		# for node in self.queue:
		# 	if node.total_cost > feasible_path_cost:
		# 		self.queue.remove(node)

		return steps

	def _solve(self, time_limit: int=1000):
		start_time = time.time()
		self.best_plan = None
		self.best_cost = float('inf')

		steps = 0
		root_state = self.env.get_state()
		root_node = self.node_class(
			state=root_state, 
			g_cost=0, 
			h_cost=self.evaluate_state(root_state)
		)

		self.queue = []
		heapq.heappush(self.queue, root_node)
		visited = {}

		while self.queue:
			current_node = heapq.heappop(self.queue)

			# Check if the current node's state matches the target state
			if self.env.is_terminal_state(current_node.state):
				if current_node.total_cost < self.best_cost:
					return reconstruct_path(current_node), steps, time.time()-start_time
				return self.best_plan, steps, time.time()-start_time

			# Check if the elapsed time has exceeded the limit
			if time.time()-start_time > time_limit:
				# print('Time limit exceeded')
				return self.best_plan, steps, time.time()-start_time

			last_obj = self.env.decode_action(current_node.action)[1] if current_node.action is not None else None
			for action in self.get_valid_actions(current_node.state):
				action_type, start_obj, target_obj, coordinates = self.env.decode_action(action)

				# If the last changed obj is the same as the current obj, continue
				if start_obj == last_obj:
					continue

				steps += 1
				self.env.set_state(current_node.get_state())
				cost, child_state = self.env._step(action_type, start_obj, target_obj, coordinates)

				# If state hasn't changed, continue
				if torch.equal(child_state['current'], current_node.state['current']):
					raise ValueError('State has not changed')

				child_hash = state_to_hashable(child_state)

				# Calculate the accumulated cost for the current path
				new_g_cost = current_node.g_cost + cost
				h_cost = self.evaluate_state(child_state)
				new_total_cost = new_g_cost + h_cost

				# Retain the node with better cost
				if child_hash not in visited or visited[child_hash] > new_total_cost:
					visited[child_hash] = new_total_cost
					child_node = self.node_class(
						state=child_state, 
						parent=current_node, 
						action=action, 
						g_cost=new_g_cost, 
						h_cost=h_cost,
						depth=current_node.depth+1
					)

					heapq.heappush(self.queue, child_node)

			if time.time()-start_time > time_limit:
				# print('Time limit exceeded')
				return self.best_plan, steps, time.time()-start_time

			# Goal Attempting
			sim_time_limit = (time_limit - time.time() + start_time) / 5
			steps += self.goal_attempt(current_node, sim_time_limit)

		return self.best_plan, steps, time.time()-start_time
