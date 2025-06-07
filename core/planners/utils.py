import time
import torch
import numpy as np
from tqdm import tqdm
from core.env.scene_manager import copy_state, Indices

def env_cost(env, actions, initial_scene, target_scene, log=True):
	env.reset(initial_scene, target_scene)
	if actions is None:
		return None
	
	ep_cost = 0
	for action in actions:
		ep_cost += env.step(action, log=log)[0]
	if log:
		print(f'episode cost: {ep_cost:.3f}')
	env.reset(initial_scene, target_scene)
	return ep_cost

def evaluate_alg(env, alg, initial_scene, target_scene, num_runs=1, **kwargs):
	plan = None
	steps, costs, elapsed_time = [], [], []
	best_cost = np.inf

	print(f"--------{alg.__name__}--------")
	if num_runs > 1:
		pbar = tqdm(total=num_runs, desc=f"Evaluating {alg.__name__}", unit="run")
	
	for _ in range(num_runs):
		env.reset(initial_scene, target_scene)
		cost = None
		plan_i, steps_i, elapsed_time_i = alg(env).solve(**kwargs)
		if plan_i:
			cost = env_cost(env, plan_i, initial_scene, target_scene, log=False)
			if cost < best_cost:
				plan = plan_i
			costs.append(cost)
			steps.append(steps_i)
			elapsed_time.append(elapsed_time_i)

		if num_runs > 1:
			pbar.update(1)
			pbar.set_postfix(cost=cost, steps=steps_i, elapsed_time=elapsed_time_i)
		else:
			elapsed_time = elapsed_time_i
			steps = steps_i
	
	if num_runs > 1:
		pbar.close()
		print(f'mean cost: {np.mean(costs):.2f} | mean elapsed_time: {np.mean(elapsed_time):.3f}s | mean steps: {np.mean(steps):.2f}')
	else:
		print(f'plan: {plan}')
		print(f'elapsed_time: {elapsed_time:.3f}s')
		print(f'steps: {steps}')

		if plan is not None:
			env_cost(env, plan, initial_scene, target_scene)

	env.reset(initial_scene, target_scene)
	return plan

def reconstruct_path(node):
    path = []
    while node.parent is not None:
        path.append(node.action)
        node = node.parent
    path.reverse()
    return path

class BaseSearch:
	def __init__(self, env, node_class):
		"""
		Base class for search algorithms.
		:param env: The environment in which the search is performed.
		:param node_class: The class used for representing nodes in the search.
		"""
		self.env = env
		self.node_class = node_class  # Generalized node class


def refine_until_convergence(env, plan, initial_scene, target_scene, alg, verbose=0):
	start_time = time.time()
	best_plan = plan
	best_cost = env_cost(env, plan, initial_scene, target_scene, log=False)
	if plan is None:
		return best_plan, best_cost, time.time() - start_time
	while True:
		if alg in ['Labbe', 'Strap', 'StrapGA']:
			refined_plan = plan_refinement(env, plan, initial_scene, target_scene, verbose=verbose)
		else:
			refined_plan = plan_refinement_stack(env, plan, initial_scene, target_scene, verbose=verbose)

		if plan == refined_plan:
			break

		cost = env_cost(env, refined_plan, initial_scene, target_scene, log=False)
		if cost < best_cost:
			if verbose > 0:
				print(f'cost got better from {best_cost:.3f} to {cost:.3f}')
			best_cost = cost
			best_plan = refined_plan

		plan = refined_plan

	return best_plan, best_cost, time.time() - start_time

def plan_refinement(env, plan, initial_scene, target_scene, verbose=0):
	# --- Stage 1: Immediate Redundancy Removal ---
	decoded_plan = []
	for action in plan:
		decoded_action = (action, env.decode_action(action))
		if decoded_action[1][0] == 'stack':
			print("There is a stack action in simple refinement")
			return plan
		# each entry: (action, (action_type, start_obj, target_obj, coord))
		# remove redundant action on the latest manipulated object
		# This check is for consecutive actions on the SAME object.
		# It keeps the last action on that object and discards previous consecutive ones.
		if len(decoded_plan) > 0 and decoded_action[1][1] == decoded_plan[-1][1][1]:
			if verbose > 0:
				print(f'Redundant action, obj {decoded_plan[-1][1][1]} moving to {decoded_plan[-1][1][3]}, was removed')
			decoded_plan[-1] = decoded_action
		else:
			decoded_plan.append(decoded_action)

	# --- Stage 2: State-Checking Redundancy (Simulate and Filter) ---
	# This stage is only required after the immediate redundancy removal
	env.reset(initial_scene, target_scene)
	action_seq = []
	plan = []

	for decoded_action in decoded_plan:
		action = decoded_action[0]
		action_type, start_obj, target_obj, coordinates = decoded_action[1]
		
		# If the object is ALREADY at the target coordinates for this move action.
		current_coord = env.current_x[start_obj, Indices.COORD]
		if torch.equal(current_coord, coordinates):
			if verbose > 0:
				print(f"Skipping Move action (obj {start_obj} to {coordinates.tolist()}): Already at target position.")
			continue

		p_place = coordinates.clone()
		p_pick = env.current_x[start_obj, Indices.COORD].clone()

		plan.append(action)
		action_seq.append({
			'k': start_obj,
			'p_pick': p_pick,
			'p_place': p_place
		})

		env._step(action_type, start_obj, target_obj, coordinates)

	env.reset(initial_scene, target_scene)
	B = {}
	H = {0: env.get_state()}		# arrangement history
	for i in range(len(action_seq)):
		k = action_seq[i]['k']

		if k in B:			# if object k was moved
			bIdx = B[k]		# previous action index on k

			# Valid mask form the action index bIdx to i-1
			valid_mask = torch.ones(env.grid_size, dtype=torch.bool)
			for j in range(bIdx-1, i):
				env.set_state(copy_state(H[j+1]))
				valid_mask &= env.valid_center_mask(k)

			env.set_state(copy_state(H[i]))

			P = valid_mask.nonzero(as_tuple=False).float()
			if len(P) == 0:
				if verbose > 0:
					print(f'No feasible buffer set')

			# Distances
			p1 = action_seq[bIdx]['p_pick'].float()
			p2 = action_seq[i-1]['p_place'].float()
			p3 = action_seq[bIdx+1]['p_pick'].float()
			p4 = action_seq[i]['p_place'].float()

			# Current cost
			p = action_seq[bIdx]['p_place'].float()
			min_cost = torch.norm(p1 - p)
			min_cost += torch.norm(p - p3)
			min_cost += torch.norm(p2 - p)
			min_cost += torch.norm(p - p4)
			min_cost = (min_cost * env.normalization_factor).item()

			# Find the best buffer
			best_p = None
			for p in P:
				cost = torch.norm(p1 - p)
				cost += torch.norm(p - p3)
				cost += torch.norm(p2 - p)
				cost += torch.norm(p - p4)
				cost = (cost * env.normalization_factor).item()
				if cost < min_cost:
					best_p = p.clone().to(torch.long)
					min_cost = cost

			# Update the best buffer
			if best_p is not None:
				if verbose > 0:
					last_pos = action_seq[bIdx]['p_place']
					print(f'Buffer of obj {k} changed from pos {last_pos.tolist()} to pos {best_p.tolist()}')

				action_seq[bIdx] = {
					'k': k,
					'p_pick': action_seq[bIdx]['p_pick'],
					'p_place': best_p
				}
				if torch.equal(best_p, action_seq[i]['p_place']):
					if verbose > 0:
						print(f'New static buffer is the same as the current one, so the action {i} is removed')
					del action_seq[i]
				else:
					action_seq[i] = {
						'k': k,
						'p_pick': best_p,
						'p_place': action_seq[i]['p_place']
					}
				break

		env.step(plan[i])
		H[i+1] = env.get_state()
		B[k] = i

	# Generate the refined plan
	refined_plan = [env.encode_move(a['k'], a['p_place']) for a in action_seq]
	return refined_plan

def plan_refinement_stack(env, plan, initial_scene, target_scene, verbose=0):
	env.static_stack = False
	# --- Stage 1: Immediate Redundancy Removal ---
	decoded_plan = []
	for action in plan:
		decoded_action = (action, env.decode_action(action))
		# each entry: (action, (action_type, start_obj, target_obj, coord))
		# remove redundant action on the latest manipulated object
		# This check is for consecutive actions on the SAME object.
		# It keeps the last action on that object and discards previous consecutive ones.
		if len(decoded_plan) > 0 and decoded_action[1][1] == decoded_plan[-1][1][1]:
			if verbose > 0:
				if decoded_plan[-1][1][0] == 'move':
					print(f'Redundant action, obj {decoded_plan[-1][1][1]} moving to {decoded_plan[-1][1][3]}, was removed')
				elif decoded_plan[-1][1][0] == 'stack':
					print(f'Redundant action, obj {decoded_plan[-1][1][1]} stacking on {decoded_plan[-1][1][2]}, was removed')
			decoded_plan[-1] = decoded_action
		else:
			decoded_plan.append(decoded_action)

	# --- Stage 2: State-Checking Redundancy (Simulate and Filter) ---
	# This stage is only required after the immediate redundancy removal
	env.reset(initial_scene, target_scene)
	action_seq = []
	plan = []

	for decoded_action in decoded_plan:
		action = decoded_action[0]
		action_type, start_obj, target_obj, coordinates = decoded_action[1]
		
		if action_type == 'stack':
			# If the object is already stacked, skip this action
			if env.current_x[start_obj, Indices.RELATION.start + target_obj] == 1:
				if verbose > 0:
					print(f"Skipping Stack action (obj {start_obj} -> {target_obj}): Object already stacked on target.")
				continue
		elif action_type == 'move':
			# If the object is ALREADY at the target coordinates for this move action.
			current_coord = env.current_x[start_obj, Indices.COORD]
			if torch.equal(current_coord, coordinates):
				if verbose > 0:
					print(f"Skipping Move action (obj {start_obj} to {coordinates.tolist()}): Already at target position.")
				continue

		if action_type == 'stack':
			p_place = env.current_x[target_obj, Indices.COORD].clone()
		else:
			p_place = coordinates.clone()
		p_pick = env.current_x[start_obj, Indices.COORD].clone()
		plan.append(action)
		action_seq.append({
			'type': action_type,
			'k': start_obj,
			'l': target_obj,
			'p_pick': p_pick,
			'p_place': p_place
		})

		env._step(action_type, start_obj, target_obj, coordinates)

	env.reset(initial_scene, target_scene)
	B = {}
	H = {0: env.get_state()}		# arrangement history
	for i in range(len(action_seq)):
		k = action_seq[i]['k']

		if k in B:			# if object k was moved
			bIdx = B[k]		# previous action index on k

			# Valid mask form the action index bIdx to i-1
			valid_mask = torch.ones(env.grid_size, dtype=torch.bool)
			for j in range(bIdx, i+1):
				env.set_state(copy_state(H[j]))
				valid_mask &= env.valid_center_mask(k)

			env.set_state(copy_state(H[i]))

			P = valid_mask.nonzero(as_tuple=False).float()
			if len(P) == 0:
				if verbose > 0:
					print(f'No feasible buffer set')

			# Occupied moving buffers in the action index bIdx to i-1
			empty_objs = []
			stable_j = env.stability_mask[k]
			candidates = torch.nonzero(stable_j, as_tuple=False).view(-1).tolist()
			for obj in candidates:
				# if obj is empty in the whole bIdx to i-1 period
				is_empty = True
				for j in range(bIdx, i+1):
					if H[j]['current'][:, obj].any():
						is_empty = False
						break
				if is_empty:
					empty_objs.append(obj)

			# Distances
			p1 = action_seq[bIdx]['p_pick'].float()
			p2 = action_seq[i-1]['p_place'].float()
			p3 = action_seq[bIdx+1]['p_pick'].float()
			p4 = action_seq[i]['p_place'].float()

			# Current cost
			if action_seq[bIdx]['type'] == 'stack':
				p_to_buff = action_seq[bIdx]['p_place'].float()
				p_i = H[i]['current'][k, Indices.COORD]
				min_cost = torch.norm(p1 - p_to_buff)
				min_cost += torch.norm(p_to_buff - p3)
				min_cost += torch.norm(p2 - p_i)
				min_cost += torch.norm(p_i - p4)
				min_cost = (min_cost * env.normalization_factor).item()
			else:
				p = action_seq[bIdx]['p_place'].float()
				min_cost = torch.norm(p1 - p)
				min_cost += torch.norm(p - p3)
				min_cost += torch.norm(p2 - p)
				min_cost += torch.norm(p - p4)
				min_cost = (min_cost * env.normalization_factor).item()

			# Find the best buffer
			best_p = None
			for p in P:
				cost = torch.norm(p1 - p)
				cost += torch.norm(p - p3)
				cost += torch.norm(p2 - p)
				cost += torch.norm(p - p4)
				cost = (cost * env.normalization_factor).item()
				if cost < min_cost:
					best_p = p.clone().to(torch.long)
					min_cost = cost

			best_obj = None
			for empty_obj in empty_objs:
				p_to_buff = H[bIdx]['current'][empty_obj, Indices.COORD].float()
				p_i = H[i]['current'][empty_obj, Indices.COORD].float()
				cost = torch.norm(p1 - p_to_buff)
				cost += torch.norm(p_to_buff - p3)
				cost += torch.norm(p2 - p_i)
				cost += torch.norm(p_i - p4)
				cost = (cost * env.normalization_factor).item()
				if cost < min_cost:
					min_cost = cost
					best_obj = empty_obj

			# Update the best buffer
			if best_obj is not None:
				if verbose > 0:
					if action_seq[bIdx]['type'] == 'stack':
						last_obj = action_seq[bIdx]['l']
						print(f'Buffer of obj {k} changed from obj {last_obj} to obj {best_obj}')
					else:
						last_pos = action_seq[bIdx]['p_place']
						print(f'Buffer of obj {k} changed from pos {last_pos.tolist()} to obj {best_obj}')

				action_seq[bIdx] = {
					'type': 'stack',
					'k': k,
					'l': best_obj,
					'p_pick': action_seq[bIdx]['p_pick'].clone(),
					'p_place': H[bIdx]['current'][best_obj, Indices.COORD].clone(),
				}
				if action_seq[i]['type'] == 'stack' and action_seq[i]['l'] == best_obj:
					if verbose > 0:
						print(f'New moving buffer is the same as the current one, so the action {i} is removed')
					del action_seq[i]
				else:
					action_seq[i]['p_pick'] = H[i]['current'][best_obj, Indices.COORD].clone()
				break
			elif best_p is not None:
				if verbose > 0:
					if action_seq[bIdx]['type'] == 'stack':
						last_obj = action_seq[bIdx]['l']
						print(f'Buffer of obj {k} changed from obj {last_obj} to pos {best_p.tolist()}')
					else:
						last_pos = action_seq[bIdx]['p_place']
						print(f'Buffer of obj {k} changed from pos {last_pos.tolist()} to pos {best_p.tolist()}')

				action_seq[bIdx] = {
					'type': 'move',
					'k': k,
					'l': k,
					'p_pick': action_seq[bIdx]['p_pick'].clone(),
					'p_place': best_p
				}
				if action_seq[i]['type'] == 'move' and torch.equal(best_p, action_seq[i]['p_place']):
					if verbose > 0:
						print(f'New static buffer is the same as the current one, so the action {i} is removed')
					del action_seq[i]
				else:
					action_seq[i]['p_pick'] = best_p
				break

		env.step(plan[i])
		H[i+1] = env.get_state()
		B[k] = i

	refined_plan = []
	for a in action_seq:
		if a['type'] == 'stack':
			refined_plan.append(env.encode_stack(a['k'], a['l']))
		else:
			refined_plan.append(env.encode_move(a['k'], a['p_place']))

	return refined_plan
