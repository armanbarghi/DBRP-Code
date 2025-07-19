import os
import cv2
import numpy as np
import pybullet as p
import trimesh
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def solve_pnp_for_view_matrix(world_points, image_points, K):
	"""Solves PnP to find the world-to-camera view matrix (V)."""
	success, rvec, tvec = cv2.solvePnP(
		np.array(world_points, dtype=np.float32),
		np.array(image_points, dtype=np.float32),
		K, distCoeffs=None, flags=cv2.SOLVEPNP_IPPE
	)
	if not success:
		return None
	R_pnp, _ = cv2.Rodrigues(rvec)
	V = np.eye(4)
	V[:3, :3] = R_pnp
	V[:3, 3] = tvec.flatten()
	return V

def draw_markers(camera, world_points, image_points):
	"""Visualizes 3D world points in the simulation and 2D points on an image."""
	print("Visualizing 3D markers (blue spheres) and 2D markers (red dots)...")
	
	# Draw bold dots (spheres) in the 3D simulation
	visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.02, rgbaColor=[0, 0, 1, 1])
	for pt in world_points:
		p.createMultiBody(baseVisualShapeIndex=visual_shape_id, basePosition=pt)

	# Draw filled circles on the 2D image
	img = camera.capture_image()
	img_contiguous = np.ascontiguousarray(img, dtype=np.uint8)
	for u, v in image_points:
		cv2.circle(img_contiguous, (int(u), int(v)), radius=6, color=(255, 0, 0), thickness=-1)
	
	# Display the image with markers
	plt.figure(figsize=(8, 6))
	plt.imshow(img_contiguous)
	plt.title("2D Marker Points")
	plt.axis('off')
	plt.show()

class CameraManager:
	"""A camera model for PyBullet simulations and computer vision tasks."""

	def __init__(self, target_pos=[0, 0, 0.5], 
			distance=1, yaw=90, pitch=-50, roll=0, 
			width=640, height=480, fov=60, cam_offset=0.75, cam_height=1):
		"""
		Initializes the camera and computes its matrices.
		"""
		self.width, self.height = width, height
		self.fov = fov

		self.cam_offset = cam_offset
		self.cam_height = cam_height
		self._viewpoints = self._precompute_viewpoints()

		# Set the initial view
		self.set_view(target_pos, distance, yaw, pitch, roll)

	def find_viewpoint_id(self, tolerance=1e-6):
		"""
		Finds the viewpoint ID that matches the current camera parameters.
		"""
		current_params = {
			'distance': self.distance,
			'yaw': self.yaw,
			'pitch': self.pitch
		}
		
		for i, viewpoint in enumerate(self._viewpoints):
			# Check if all parameters match within tolerance
			distance_match = abs(viewpoint['distance'] - current_params['distance']) < tolerance
			yaw_match = abs(viewpoint['yaw'] - current_params['yaw']) < tolerance
			pitch_match = abs(viewpoint['pitch'] - current_params['pitch']) < tolerance
			
			if distance_match and yaw_match and pitch_match:
				return i + 1  # Return 1-based index
		
		return None  # No matching viewpoint found

	def _precompute_viewpoints(self):
		"""Pre-calculates the distance, yaw, and pitch for 9 standard viewpoints."""
		poses = [
			[-self.cam_offset, -self.cam_offset], [-self.cam_offset, 0], [-self.cam_offset, self.cam_offset],
			[0, -self.cam_offset], [0, 0], [0, self.cam_offset],
			[self.cam_offset, -self.cam_offset], [self.cam_offset, 0], [self.cam_offset, self.cam_offset]
		]
		viewpoints = []
		for x, y in poses:
			distance = np.sqrt(x**2 + y**2 + self.cam_height**2)
			yaw = np.rad2deg(np.arctan2(x, -y))
			horizontal_dist = np.sqrt(x**2 + y**2)
			pitch = -90 if horizontal_dist == 0 else np.rad2deg(np.arctan2(-self.cam_height, horizontal_dist))
			viewpoints.append({'distance': distance, 'yaw': yaw, 'pitch': pitch})
		return viewpoints

	def set_viewpoint(self, target_pos, viewpoint_id=None):
		"""Sets the camera to a pre-defined viewpoint."""
		if viewpoint_id is None:
			# choose a random viewpoint if none is specified
			viewpoint_id = np.random.randint(1, len(self._viewpoints) + 1)

		if not (1 <= viewpoint_id <= len(self._viewpoints)):
			raise ValueError(f"Viewpoint ID must be between 1 and {len(self._viewpoints)}.")

		params = self._viewpoints[viewpoint_id - 1]
		self.set_view(target_pos, params['distance'], params['yaw'], params['pitch'])

	def set_view(self, target_pos, distance, yaw, pitch, roll=0):
		"""Sets the camera's view and updates its matrices."""
		self.target_pos = target_pos
		self.distance = distance
		self.yaw = yaw
		self.pitch = pitch
		self.roll = roll

		# Configure the GUI camera visuals
		p.resetDebugVisualizerCamera(
			cameraDistance=self.distance,
			cameraTargetPosition=self.target_pos,
			cameraYaw=self.yaw, cameraPitch=self.pitch
		)
		
		# Store the raw PyBullet matrices (GL convention)
		self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
			self.target_pos, self.distance, self.yaw, self.pitch, self.roll, upAxisIndex=2)
		self.proj_matrix = p.computeProjectionMatrixFOV(
			self.fov, self.width/self.height, 0.01, 100)
		
		# Set all internal matrix attributes (K, V, R, t)
		self.cal_parameters()

		self.viewpoint_id = self.find_viewpoint_id()

	def cal_parameters(self, world_points=None, image_points=None):
		"""Calculates and sets all camera matrices (K, V, R, t)."""
		# Intrinsic Matrix K (from FOV)
		o_x, o_y = self.width / 2, self.height / 2
		f_y = (self.height / 2) / np.tan(np.deg2rad(self.fov) / 2)
		self.K = np.array([[f_y, 0, o_x], [0, f_y, o_y], [0, 0, 1]])

		# View Matrix V (World -> GL Camera)
		self.V = np.array(self.view_matrix).reshape(4, 4, order='F')
		if world_points is not None and image_points is not None:
			# Estimate pose using PnP
			V_estimated = solve_pnp_for_view_matrix(world_points, image_points, self.K)
			if V_estimated is None:
				print("\n❌ Validation Failed: PnP solver returned None.")
				return

			# Convert the PnP result (CV frame) to the GL frame for a correct comparison
			cv_to_gl_flip = np.diag([1, -1, -1, 1])
			V_estimated = cv_to_gl_flip @ V_estimated

			# Compare the matrices
			difference = np.sum(np.abs(self.V - V_estimated))
			print(f"\nSum of absolute differences: {difference:.8f}")
	
			# self.V = np.diag([1, -1, -1, 1]) @ V_estimated

		# Inverse Transformation (CV Camera -> World) for unprojection
		V_inv_gl = np.linalg.inv(self.V)
		R_gl_to_world = V_inv_gl[:3, :3]
		self.t = V_inv_gl[:3, 3].reshape(3, 1)
		
		# Adapter to create a CV-to-World rotation matrix for unprojection math
		self.R = R_gl_to_world @ np.diag([1, -1, -1])

	def z_c_calculator(self, u, v, n, p0, h=0.0):
		"""Calculates a pixel's depth by intersecting its ray with a plane."""
		K_inv = np.linalg.inv(self.K)
		ray_cam_cv = K_inv @ [u, v, 1.0]
		numerator = h + (n @ p0) - (n @ self.t.flatten())
		denominator = n @ self.R @ ray_cam_cv
		if abs(denominator) < 1e-6: return None
		return numerator / denominator

	def project_world_to_pixel(self, world_points):
		"""Projects 3D world points to 2D image pixels."""
		world_points_h = np.vstack([np.asanyarray(world_points).T, np.ones(len(world_points))])
		
		# World -> GL Camera -> CV Camera
		camera_points_gl = (self.V @ world_points_h)[:3, :]
		camera_points_cv = np.diag([1, -1, -1]) @ camera_points_gl
		
		# Project to image plane
		image_coords_h = self.K @ camera_points_cv
		
		# Dehomogenize
		u = image_coords_h[0, :] / image_coords_h[2, :]
		v = image_coords_h[1, :] / image_coords_h[2, :]
		return u, v

	def project_pixel_to_world(self, u, v, n, p0, h=0.0):
		"""Unprojects a 2D pixel to a 3D world point on an arbitrary plane."""
		z_c = self.z_c_calculator(u, v, n, p0, h)
		if z_c is None: return None
		
		# Point in CV Camera Frame
		K_inv = np.linalg.inv(self.K)
		point_in_cam_cv = z_c * (K_inv @ [u, v, 1.0])
		
		# CV Camera Frame -> World Frame
		world_point = self.R @ point_in_cam_cv + self.t.flatten()
		return world_point

	def compute_2d_bounding_box(self, objects_dir, body_name, body_id, body_type, num_samples=500, padding=3):
		"""
		Sample points on the object's mesh, check for occlusion, project visible points,
		and return the 2D bounding box and visibility ratio.
		A padding can be added to dilate the box.
		"""
		object_body_path = os.path.join(objects_dir, body_name, f"{body_name}_{body_type}")
		mesh_path = os.path.join(object_body_path, f"{body_name}_{body_type}.obj")
		mesh = trimesh.load_mesh(mesh_path)
		points, _ = trimesh.sample.sample_surface(mesh, num_samples)
		pos, orn = p.getBasePositionAndOrientation(body_id)
		R = trimesh.transformations.quaternion_matrix([orn[3], *orn[:3]])[:3, :3]
		world_pts = (R @ points.T).T + np.array(pos)

		# Occlusion check
		camera_pos = self.t.flatten()
		num_pts = len(world_pts)
		if num_pts == 0:
			return None, 0

		ray_from_positions = [camera_pos] * num_pts
		ray_to_positions = world_pts

		batch_results = p.rayTestBatch(ray_from_positions, ray_to_positions)

		visible_world_pts = []
		for i, result in enumerate(batch_results):
			hit_object_id = result[0]
			# If rayTest hits nothing (-1) or the intended object, the point is visible.
			if hit_object_id == -1 or hit_object_id == body_id:
				visible_world_pts.append(world_pts[i])
		
		visibility_ratio = len(visible_world_pts) / len(world_pts) if len(world_pts) > 0 else 0

		if not visible_world_pts:
			return None, visibility_ratio

		u, v = self.project_world_to_pixel(np.array(visible_world_pts))

		# Create a tight bounding box and then add padding
		u_min, u_max = np.floor(u.min()).astype(int), np.ceil(u.max()).astype(int)
		v_min, v_max = np.floor(v.min()).astype(int), np.ceil(v.max()).astype(int)

		# Dilate the bounding box and clamp to image dimensions
		u_min = max(0, u_min - padding)
		v_min = max(0, v_min - padding)
		u_max = min(self.width, u_max + padding)
		v_max = min(self.height, v_max + padding)
		
		return (u_min, v_min, u_max, v_max), visibility_ratio

	def draw_2d_bounding_boxes(self, image, boxes=[], labels=[], color='yellow', title=''):
		"""Display image with bounding box overlays."""
		fig, axs = plt.subplots(1, 2, figsize=(2 * self.width / 100, self.height / 100), dpi=100)

		# Subplot 1: Raw image
		axs[0].imshow(image)
		axs[0].set_title("Raw Image")
		axs[0].axis('off')

		# Subplot 2: Image with bounding boxes
		axs[1].imshow(image)
		axs[1].set_title("Image with Bounding Boxes")
		for i, (u_min, v_min, u_max, v_max) in enumerate(boxes):
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

	def draw_3d_bounding_box(self, bbox, n, p0, color=[1, 0, 0], line_width=2):
		"""Projects a 2D pixel bbox onto an arbitrary 3D plane and draws it."""
		u_min, v_min, u_max, v_max = bbox
		pixel_corners = [
			(u_min, v_min), (u_max, v_min),
			(u_max, v_max), (u_min, v_max)
		]
		
		# Project each corner to the specified plane
		world_corners = [self.project_pixel_to_world(u, v, n, p0) for u, v in pixel_corners]
		if any(c is None for c in world_corners):
			print("Could not project one of the bbox corners. Aborting draw.")
			return []

		# Draw lines in the simulation
		debug_item_ids = []
		for i in range(4):
			start = world_corners[i]
			end = world_corners[(i + 1) % 4]
			line_id = p.addUserDebugLine(start, end, lineColorRGB=color, lineWidth=line_width)
			debug_item_ids.append(line_id)
		return debug_item_ids

	def capture_image(self):
		"""Captures and returns an RGB image from the camera's viewpoint."""
		_, _, rgb, _, _ = p.getCameraImage(
			self.width, self.height, 
			viewMatrix=self.view_matrix,
			projectionMatrix=self.proj_matrix, 
			renderer=p.ER_BULLET_HARDWARE_OPENGL,
			shadow=1,
		)

		# Convert to numpy array and ensure correct data type
		image = np.array(rgb, dtype=np.uint8)
		image = image.reshape((self.height, self.width, 4))[:, :, :3]  # Remove alpha channel

		return image

	def show_img(self, image, title=''):
		"""Display image."""
		fig, ax = plt.subplots(figsize=(self.width / 100, self.height / 100))
		ax.imshow(image)
		plt.title(title)
		plt.axis('off')
		plt.show()
