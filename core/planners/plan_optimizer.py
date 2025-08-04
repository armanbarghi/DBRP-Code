import time
import torch
from core.env.scene_manager import Indices, copy_state
from core.planners.planning_utils import env_cost


def redundancy_pruning(env, plan, initial_scene, target_scene, verbose=0):
	"""
	Performs the first stage of refinement: immediate redundancies and non-consecutive redundancies.
	Returns a clean action_seq.
	"""
	# --- Immediate Redundancy ---
	decoded_plan = []
	for action in plan:
		decoded_action = (action, env.decode_action(action))
		# Each entry: (action, (action_type, start_obj, target_obj, coord))
		# Remove redundant action on the latest manipulated object
		# This check is for consecutive actions on the SAME object.
		# It keeps the last action on that object and discards previous consecutive ones.
		if len(decoded_plan) > 0 and decoded_action[1][1] == decoded_plan[-1][1][1]:
			if verbose > 0:
				if decoded_plan[-1][1][0] == 'move':
					print(f'Redundant action, obj {decoded_plan[-1][1][1]} moving to {decoded_plan[-1][1][3].tolist()}, was removed')
				elif decoded_plan[-1][1][0] == 'stack':
					print(f'Redundant action, obj {decoded_plan[-1][1][1]} stacking on {decoded_plan[-1][1][2]}, was removed')
			decoded_plan[-1] = decoded_action
		else:
			decoded_plan.append(decoded_action)

	# --- Non-Consecutive Redundancies (Simulate and Filter) ---
	env.reset(initial_scene, target_scene)
	action_seq = []

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

		action_seq.append({
			'type': action_type,
			'k': start_obj,
			'l': target_obj,
			'p_pick': p_pick,
			'p_place': p_place
		})

		env._step(action_type, start_obj, target_obj, coordinates)

	return action_seq

def plan_refinement(env, plan, initial_scene, target_scene, refine_mode=None, verbose=0):
	env.static_stack = False
	start_time = time.time()
	cost = env_cost(env, plan, initial_scene, target_scene, log=False)
	if plan is None:
		return plan, cost, time.time() - start_time

	# Perform redundancy pruning once at the beginning
	action_seq = redundancy_pruning(env, plan, initial_scene, target_scene, verbose=verbose)
	current_plan = action_seq_to_plan(env, action_seq)
	best_plan = current_plan
	best_cost = env_cost(env, best_plan, initial_scene, target_scene, log=False)
	if best_cost > cost:
		if verbose > 0:
			print(f'Initial cost: {cost:.3f}, after pruning: {best_cost:.3f}')

	while True:
		if refine_mode == "stack":
			action_seq = buffer_optimality_stack(env, action_seq, initial_scene, target_scene, verbose=verbose)
		elif refine_mode == "move":
			action_seq = buffer_optimality_move(env, action_seq, initial_scene, target_scene, verbose=verbose)
		else:
			raise ValueError(f'Unknown refinement mode: {refine_mode}')

		# Convert to plan for convergence check
		refined_plan = action_seq_to_plan(env, action_seq)

		# Check convergence by comparing plans
		if current_plan == refined_plan:
			break
		
		# Evaluate cost of refined plan
		cost = env_cost(env, refined_plan, initial_scene, target_scene, log=False)
		if cost < best_cost:
			if verbose > 0:
				print(f'cost got better from {best_cost:.3f} to {cost:.3f}')
			best_cost = cost
			best_plan = refined_plan

		current_plan = refined_plan

	return best_plan, best_cost, time.time() - start_time

def action_seq_to_plan(env, action_seq):
    """Convert action_seq back to plan using unified format"""
    plan = []
    for a in action_seq:
        if a['type'] == 'stack':
            plan.append(env.encode_stack(a['k'], a['l']))
        else:
            plan.append(env.encode_move(a['k'], a['p_place']))
    return plan

def buffer_optimality_move(env, action_seq, initial_scene, target_scene, verbose=0):
	"""Core refinement logic for simple (move-only) plans"""
	env.reset(initial_scene, target_scene)
	plan = action_seq_to_plan(env, action_seq)
	B = {}
	H = {0: env.get_state()}		# arrangement history
	for i in range(len(action_seq)):
		k = action_seq[i]['k']

		if k in B:			# if object k was moved
			b = B[k]		# previous action index on k

			# Valid mask form the action index b to i-1
			valid_mask = torch.ones(env.grid_size, dtype=torch.bool)
			for j in range(b-1, i):
				env.set_state(copy_state(H[j+1]))
				valid_mask &= env.valid_center_mask(k)
			P = valid_mask.nonzero(as_tuple=False).float()

			env.set_state(copy_state(H[i]))

			if len(P) == 0:
				if verbose > 0:
					print(f'No feasible buffer set')
				env.step(plan[i])
				H[i+1] = env.get_state()
				B[k] = i
				continue

			# Distances
			p1 = action_seq[b]['p_pick'].float()
			p2 = action_seq[i-1]['p_place'].float()
			p3 = action_seq[b+1]['p_pick'].float()
			p4 = action_seq[i]['p_place'].float()

			# Current cost
			if action_seq[b]['type'] == 'stack':
				p_to_buff = action_seq[b]['p_place'].float()
				p_i = H[i]['current'][k, Indices.COORD]
				min_cost = torch.norm(p1 - p_to_buff)
				min_cost += torch.norm(p_to_buff - p3)
				min_cost += torch.norm(p2 - p_i)
				min_cost += torch.norm(p_i - p4)
				min_cost = (min_cost * env.normalization_factor).item()
			else:
				p = action_seq[b]['p_place'].float()
				min_cost = torch.norm(p1 - p)
				min_cost += torch.norm(p - p3)
				min_cost += torch.norm(p2 - p)
				min_cost += torch.norm(p - p4)
				min_cost = (min_cost * env.normalization_factor).item()

			# Find the best static buffer
			best_p = None
			for p in P:
				# Skip if this position is the same as the final placement position
				if action_seq[i]['type'] == 'move' and torch.equal(p.to(torch.long), action_seq[i]['p_place']):
					continue

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
					if action_seq[b]['type'] == 'stack':
						last_obj = action_seq[b]['l']
						print(f'Buffer of obj {k} changed from obj {last_obj} to pos {best_p.tolist()}')
					else:
						last_pos = action_seq[b]['p_place'].tolist()
						print(f'Buffer of obj {k} changed from pos {last_pos} to pos {best_p.tolist()}')

				action_seq[b]['type'] = 'move'
				action_seq[b]['l'] = k  # For move actions, target is same as object
				action_seq[b]['p_place'] = best_p
				action_seq[i]['p_pick'] = best_p
				break

		env.step(plan[i])
		H[i+1] = env.get_state()
		B[k] = i

	return action_seq

def buffer_optimality_stack(env, action_seq, initial_scene, target_scene, verbose=0):
	"""Core stack refinement logic"""
	env.reset(initial_scene, target_scene)
	plan = action_seq_to_plan(env, action_seq)
	B = {}
	H = {0: env.get_state()}		# arrangement history
	for i in range(len(action_seq)):
		k = action_seq[i]['k']

		if k in B:			# if object k was moved
			b = B[k]		# previous action index on k

			# Valid mask form the action index b to i-1
			valid_mask = torch.ones(env.grid_size, dtype=torch.bool)
			for j in range(b, i+1):
				env.set_state(copy_state(H[j]))
				valid_mask &= env.valid_center_mask(k)
			P = valid_mask.nonzero(as_tuple=False).float()

			env.set_state(copy_state(H[i]))

			# Occupied dynamic buffers in the action index b to i-1
			empty_objs = []
			stable_j = env.stability_mask[k]
			candidates = torch.nonzero(stable_j, as_tuple=False).view(-1).tolist()
			for obj in candidates:
				if obj == action_seq[i]['l']:
					continue
				# if obj is empty within b to i-1 period
				is_empty = True
				for j in range(b, i+1):
					rel = H[j]['current'][:, Indices.RELATION]
					if rel[:, obj].any():
						is_empty = False
						break
				if is_empty:
					empty_objs.append(obj)

			if len(P) == 0 and len(empty_objs) == 0:
				if verbose > 0:
					print(f'No feasible buffer set')
				env.step(plan[i])
				H[i+1] = env.get_state()
				B[k] = i
				continue
			
			# Distances
			p1 = action_seq[b]['p_pick'].float()
			p2 = action_seq[i-1]['p_place'].float()
			p3 = action_seq[b+1]['p_pick'].float()
			p4 = action_seq[i]['p_place'].float()

			# Current cost
			if action_seq[b]['type'] == 'stack':
				p_to_buff = action_seq[b]['p_place'].float()
				p_i = H[i]['current'][k, Indices.COORD]
				min_cost = torch.norm(p1 - p_to_buff)
				min_cost += torch.norm(p_to_buff - p3)
				min_cost += torch.norm(p2 - p_i)
				min_cost += torch.norm(p_i - p4)
				min_cost = (min_cost * env.normalization_factor).item()
			else:
				p = action_seq[b]['p_place'].float()
				min_cost = torch.norm(p1 - p)
				min_cost += torch.norm(p - p3)
				min_cost += torch.norm(p2 - p)
				min_cost += torch.norm(p - p4)
				min_cost = (min_cost * env.normalization_factor).item()

			# Find the best static buffer
			best_p = None
			for p in P:
				# Skip if this position is the same as the final placement position (for move actions)
				if action_seq[i]['type'] == 'move' and torch.equal(p.to(torch.long), action_seq[i]['p_place']):
					continue

				cost = torch.norm(p1 - p)
				cost += torch.norm(p - p3)
				cost += torch.norm(p2 - p)
				cost += torch.norm(p - p4)
				cost = (cost * env.normalization_factor).item()
				if cost < min_cost:
					best_p = p.clone().to(torch.long)
					min_cost = cost

			# Find the best dynamic buffer
			best_obj = None
			for empty_obj in empty_objs:
				# Skip if this object is the same as the final placement object (for stack actions)
				if action_seq[i]['type'] == 'stack' and action_seq[i]['l'] == empty_obj:
					continue

				p_to_buff = H[b]['current'][empty_obj, Indices.COORD].float()
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
					if action_seq[b]['type'] == 'stack':
						last_obj = action_seq[b]['l']
						print(f'Buffer of obj {k} changed from obj {last_obj} to obj {best_obj}')
					else:
						last_pos = action_seq[b]['p_place'].tolist()
						print(f'Buffer of obj {k} changed from pos {last_pos} to obj {best_obj}')

				action_seq[b]['type'] = 'stack'
				action_seq[b]['l'] = best_obj
				action_seq[b]['p_place'] = H[b]['current'][best_obj, Indices.COORD].clone()
				action_seq[i]['p_pick'] = H[i]['current'][best_obj, Indices.COORD].clone()
				break
			elif best_p is not None:
				if verbose > 0:
					if action_seq[b]['type'] == 'stack':
						last_obj = action_seq[b]['l']
						print(f'Buffer of obj {k} changed from obj {last_obj} to pos {best_p.tolist()}')
					else:
						last_pos = action_seq[b]['p_place'].tolist()
						print(f'Buffer of obj {k} changed from pos {last_pos} to pos {best_p.tolist()}')

				action_seq[b]['type'] = 'move'
				action_seq[b]['l'] = k  # For move actions, target is same as object
				action_seq[b]['p_place'] = best_p
				action_seq[i]['p_pick'] = best_p
				break

		env.step(plan[i])
		H[i+1] = env.get_state()
		B[k] = i

	return action_seq
