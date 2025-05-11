import os
import time
import random
import numpy as np
import pybullet as p
import pybullet_data
import trimesh
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_object(name, pos, orn=[0, 0, 0], scale=1, bodyType=None, verbose=0):
    if bodyType is None:
        bodyType = random.randint(1, len(os.listdir(f"objects/{name}")))
    urdf_path = f"objects/{name}/{name}_{bodyType}/{name}_{bodyType}.urdf"
    orn = p.getQuaternionFromEuler(orn)
    bodyId = p.loadURDF(urdf_path, pos, orn, useFixedBase=False,
                        globalScaling=scale, flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
    p.changeVisualShape(bodyId, -1, rgbaColor=[1, 1, 1, 1])
    if verbose:
        print(f"Loaded object {name} at {pos}")
    return bodyId, bodyType

def load_table(gridSize, pos=[0, 0, 0], verbose=0):
    if gridSize == (100, 100):
        file_path = "objects/table/table_1.1/table_1.1.urdf"
    elif gridSize == (100, 300):
        file_path = "objects/table/table_1.3/table_1.3.urdf"
    else:
        raise ValueError("Invalid grid size. Supported sizes are (100, 100) and (100, 300).")
    bodyId = p.loadURDF(file_path, pos, useFixedBase=True)
    p.changeVisualShape(bodyId, -1, rgbaColor=[1, 1, 1, 1])
    if verbose > 0:
        print(f"Loaded table with grid size {gridSize}")
    return bodyId

def get_size(obj_id):
	# Get the bounding box of the object
	bounding_box = p.getAABB(obj_id)
	# Calculate the size of the bounding box
	size = np.array(bounding_box[1]) - np.array(bounding_box[0])
	return size

class PyBulletSim:
	def __init__(self, mode=p.DIRECT):
		try:
			p.disconnect()
		except:
			pass
		
		if mode == p.DIRECT:
			self.client = p.connect(p.DIRECT, options="--egl")
		else:
			self.client = p.connect(mode)
			
		p.setPhysicsEngineParameter(enableFileCaching=0)
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		p.resetSimulation()
		p.setGravity(0, 0, -9.8)
		p.setRealTimeSimulation(0)
		p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
		p.loadURDF("plane.urdf")

		self.time_step = p.getPhysicsEngineParameters()['fixedTimeStep']

	def step(self, duration=0):
		for _ in range(int(duration / self.time_step)):
			p.stepSimulation()

	def close(self):
		if p.getConnectionInfo(self.client)['connectionMethod'] == p.GUI:
			while p.isConnected():
				p.stepSimulation()
				time.sleep(self.time_step)
		try:
			p.disconnect()
		except:
			pass

class Camera:
	def __init__(self, target_pos, distance, yaw, pitch, roll, width=640, height=480,
					near=0.01, far=100, fov=60):
		self.target = target_pos
		self.distance = distance
		self.yaw, self.pitch, self.roll = yaw, pitch, roll
		self.width, self.height = width, height
		self.near, self.far = near, far
		self.fov = fov
		self.up_axis = 2

		p.resetDebugVisualizerCamera(
			cameraDistance=distance, 
			cameraTargetPosition=target_pos, 
			cameraYaw=yaw, cameraPitch=pitch
		)

		self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
			self.target, self.distance, self.yaw, self.pitch, self.roll, self.up_axis)
		self.proj_matrix = p.computeProjectionMatrixFOV(
			self.fov, self.width/self.height, self.near, self.far)

		self.V = np.array(self.view_matrix).reshape(4, 4, order='F')
		self.P = np.array(self.proj_matrix).reshape(4, 4, order='F')

	def capture_image(self):
		_, _, rgb, _, _ = p.getCameraImage(
			self.width, self.height,
			viewMatrix=self.view_matrix,
			projectionMatrix=self.proj_matrix,
			shadow=True,
			renderer=p.ER_BULLET_HARDWARE_OPENGL)
		rgba = np.reshape(rgb, (self.height, self.width, 4)).astype(np.uint8)
		return rgba[:, :, :3]

	def project_points(self, world_points):
		N = world_points.shape[0]
		clip_pts = self.P @ (self.V @ np.vstack([world_points.T, np.ones(N)]))
		ndc = clip_pts / clip_pts[3]
		u = (ndc[0]*0.5 + 0.5) * self.width
		v = (1 - (ndc[1]*0.5 + 0.5)) * self.height
		return u, v

	def compute_bounding_box(self, bodyName, bodyId, bodyType, numSamples=200):
		mesh_path = f"objects/{bodyName}/{bodyName}_{bodyType}/{bodyName}_{bodyType}.obj"
		mesh = trimesh.load_mesh(mesh_path)
		points, _ = trimesh.sample.sample_surface(mesh, numSamples)
		pos, orn = p.getBasePositionAndOrientation(bodyId)
		R = trimesh.transformations.quaternion_matrix([orn[3], *orn[:3]])[:3, :3]
		world_pts = (R @ points.T).T + np.array(pos)
		u, v = self.project_points(world_pts)
		u_min, u_max = np.floor(u.min()).astype(int), np.ceil(u.max()).astype(int)
		v_min, v_max = np.floor(v.min()).astype(int), np.ceil(v.max()).astype(int)
		return (u_min, u_max, v_min, v_max)

	def show_img(self, image, title=''):
		fig, ax = plt.subplots(figsize=(self.width / 100, self.height / 100), dpi=100)
		ax.imshow(image)
		plt.title(title)
		plt.axis('off')
		plt.show()

	def draw_bounding_boxes(self, image, boxes=[], labels=[], color='yellow', title=''):
		fig, axs = plt.subplots(1, 2, figsize=(2 * self.width / 100, self.height / 100), dpi=100)

		# Subplot 1: Raw image
		axs[0].imshow(image)
		axs[0].set_title("Raw Image")
		axs[0].axis('off')

		# Subplot 2: Image with bounding boxes
		axs[1].imshow(image)
		axs[1].set_title("Image with Bounding Boxes")
		for i, (u_min, u_max, v_min, v_max) in enumerate(boxes):
			rect = patches.Rectangle(
				(u_min, v_min), u_max - u_min, v_max - v_min,
				linewidth=2, edgecolor=color, facecolor='none'
			)
			axs[1].add_patch(rect)
			if labels and i < len(labels):
				axs[1].text(u_min, v_min-10, labels[i], color=color,
							fontsize=8, backgroundcolor='black')
		axs[1].axis('off')
		
		plt.suptitle(title)
		plt.tight_layout()
		plt.show()
