import numpy as np
from tqdm import tqdm


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
