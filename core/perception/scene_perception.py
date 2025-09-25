import sys
import torch
import numpy as np
from PIL import Image
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional, Dict
from scipy.optimize import linear_sum_assignment
sys.path.append("./core/perception/yolov5") # TODO: clone if not exists
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes

BBox = Tuple[float, float, float, float]
Detection = Dict[str, any]

def cal_iou(box1: BBox, box2: BBox) -> float:
	"""Calculates IoU for boxes in [xmin, ymin, xmax, ymax] format."""
	# Determine the coordinates of the intersection rectangle
	inter_x1 = max(box1[0], box2[0])
	inter_y1 = max(box1[1], box2[1])
	inter_x2 = min(box1[2], box2[2])
	inter_y2 = min(box1[3], box2[3])

	# Compute the area of intersection
	inter_w = max(0, inter_x2 - inter_x1)
	inter_h = max(0, inter_y2 - inter_y1)
	inter_area = inter_w * inter_h

	# Compute the area of both bounding boxes
	area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
	area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

	# Compute the area of union
	union_area = area1 + area2 - inter_area
	return inter_area / union_area if union_area > 0 else 0.0

def plot_detections(image, detections, class_map=None, ax=None, title=None):
	"""
	Plots an image and overlays bounding boxes from detections on a given ax.
	"""
	if isinstance(image, str):
		try:
			image = Image.open(image)
		except FileNotFoundError:
			print(f"Error: Image file not found at {image}")
			return

	# If no axes is provided, create a new figure and axes
	show_plot = False
	if ax is None:
		fig, ax = plt.subplots(1, figsize=(10, 6))
		show_plot = True

	ax.imshow(image)

	for i, det in enumerate(detections):
		bbox = det['bbox']
		xmin, ymin, xmax, ymax = bbox
		width = xmax - xmin
		height = ymax - ymin

		# Create a Rectangle patch
		rect = patches.Rectangle(
			(xmin, ymin), width, height,
			linewidth=2, edgecolor='lime', facecolor='none'
		)
		ax.add_patch(rect)

		# Add a label
		label_parts = []
		if class_map:
			label_parts.append(f"({i})")
			label_parts.append(class_map.get(det['class'], f"ID {det['class']}"))
		if 'confidence' in det:
			label_parts.append(f"{det['confidence']:.2f}")

		if label_parts:
			ax.text(
				xmin, ymin - 5, " ".join(label_parts),
				color='black', fontsize=8,
				bbox=dict(facecolor='lime', alpha=0.7, pad=1, edgecolor='none')
			)

	ax.axis('off')
	if title:
		ax.set_title(title)

	# Only call plt.show() if we created the figure inside this function
	if show_plot:
		plt.tight_layout()
		plt.show()

def filter_duplicate_detections(
	detections,
	same_class_iou_thres=0.65,
	cross_class_iou_thres=0.90
):
	"""
	Filters out duplicate detections using Non-Maximum Suppression (NMS).

	This function handles both same-class and cross-class duplicates. It keeps
	the detection with the highest confidence and removes others that have a
	high overlap (IoU).

	Args:
		detections (list): A list of detected objects from YOLO.
						Each detection is a dict with 'bbox', 'confidence', and 'class'.
		same_class_iou_thres (float): The IoU threshold for detections of the same class.
		cross_class_iou_thres (float): The IoU threshold for detections of different classes.
									This should be high to only remove clear duplicates.

	Returns:
		list: A new list of filtered detections.
	"""
	# Sort detections by confidence in descending order
	dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)

	kept_detections = []
	while dets:
		# Keep the detection with the highest confidence
		best_det = dets.pop(0)
		kept_detections.append(best_det)

		# Remove other detections that have a high IoU
		remaining_dets = []
		for d in dets:
			iou = cal_iou(best_det['bbox'], d['bbox'])

			# Check for same-class duplicates
			if d['class'] == best_det['class']:
				if iou <= same_class_iou_thres:
					remaining_dets.append(d)
			# Check for cross-class duplicates (high overlap but different class)
			else:
				if iou <= cross_class_iou_thres:
					remaining_dets.append(d)
		dets = remaining_dets

	return kept_detections

# ===============================YOLO===============================

def load_yolo(weights_path: str, device: Optional[str] = None):
	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"

	yolo = torch.hub.load(
		"ultralytics/yolov5",
		"custom",
		path=weights_path,
		autoshape=False,
		force_reload=True, 
		device='cpu'
	).eval()

	return yolo

@torch.inference_mode()
def detect_objects(
		model,
		pil_image,
		device="cuda" if torch.cuda.is_available() else "cpu",
		imgsz=640,
		conf_thres=0.35,
		iou_thres=0.45,
):
	model.to(device).eval()

	img0 = np.array(pil_image)[:, :, ::-1]

	stride = model.stride
	if not isinstance(stride, (int, float)):
		stride = max(stride)
	stride = int(stride)

	img = letterbox(img0, imgsz, stride=stride, auto=True)[0]

	img = img[:, :, ::-1].transpose(2, 0, 1)
	img = np.ascontiguousarray(img)
	img = torch.from_numpy(img).to(device).float() / 255.0
	img = img.unsqueeze(0)

	pred = model(img)[0]

	det = non_max_suppression(pred, conf_thres, iou_thres, max_det=300)[0]

	if det is None or not len(det):
		return []

	det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

	results = [
		{
			"bbox":   [int(x1), int(y1), int(x2), int(y2)],
			"confidence": float(conf),
			"class":     int(cls),
		}
		for (x1, y1, x2, y2, conf, cls) in det.cpu().numpy()
	]
	return results

# ===============================Filter & Matching===============================

def _apply_same_class_nms(detections: List[Detection], thresh: float) -> List[Detection]:
	"""Applies Non-Maximum Suppression to a list of detections on a per-class basis."""
	grouped_dets = defaultdict(list)
	for det in detections:
		grouped_dets[det['class']].append(det)

	final_detections = []
	for class_id, dets in grouped_dets.items():
		dets = sorted(dets, key=lambda d: d['confidence'], reverse=True)
		kept_dets = []
		while dets:
			best_det = dets.pop(0)
			kept_dets.append(best_det)
			dets = [d for d in dets if cal_iou(best_det['bbox'], d['bbox']) <= thresh]
		final_detections.extend(kept_dets)
	return final_detections

def _apply_cross_class_nms_on_pairs(
	detections_A: List[Detection],
	detections_B: List[Detection],
	thresh: float
) -> (List[Detection], List[Detection]):
	"""Applies cross-class NMS on paired detections using their combined confidence."""
	if not detections_A:
		return [], []

	paired_detections = []
	for i, (det_A, det_B) in enumerate(zip(detections_A, detections_B)):
		combined_confidence = (det_A['confidence'] + det_B['confidence']) / 2.0
		paired_detections.append({'det_A': det_A, 'combined_confidence': combined_confidence, 'original_idx': i})

	sorted_pairs = sorted(paired_detections, key=lambda x: x['combined_confidence'], reverse=True)

	kept_indices = []
	while sorted_pairs:
		best_pair = sorted_pairs.pop(0)
		kept_indices.append(best_pair['original_idx'])
		sorted_pairs = [p for p in sorted_pairs if cal_iou(best_pair['det_A']['bbox'], p['det_A']['bbox']) <= thresh]

	final_A = [detections_A[i] for i in kept_indices]
	final_B = [detections_B[i] for i in kept_indices]
	return final_A, final_B

def consensus_filter(
	detections_A: List[Detection],
	detections_B: List[Detection],
	same_class_iou_thresh: float = 0.5,
	cross_class_iou_thresh: float = 0.6,
	confusable_groups: List[set] = None
) -> (List[Detection], List[Detection]):
	"""
	Establishes a consistent set of objects between two scenes using reciprocal validation.

	This module acts as a "Consensus Filter" by:
	1. Removing same-class duplicates in each scene via NMS.
	2. Finding conceptual matches based on mutual confidence.
	3. Removing cross-class spatial duplicates from the final matched pairs.
	"""
	# Step 1: Per-class NMS on each scene independently
	clean_A = _apply_same_class_nms(detections_A, same_class_iou_thresh)
	clean_B = _apply_same_class_nms(detections_B, same_class_iou_thresh)

	# Step 2: Reciprocal matching to get conceptually aligned pairs
	# (This internal logic finds exact matches first, then uses confusion groups for leftovers)
	matched_A, matched_B = [], []
	matched_indices_A, matched_indices_B = set(), set()
	
	# Exact Matching
	grouped_A = defaultdict(list)
	for i, det in enumerate(clean_A):
		grouped_A[det['class']].append({'det': det, 'idx': i})
	grouped_B = defaultdict(list)
	for i, det in enumerate(clean_B):
		grouped_B[det['class']].append({'det': det, 'idx': i})
	
	all_classes = set(grouped_A.keys()) | set(grouped_B.keys())
	for class_id in all_classes:
		dets_A, dets_B = grouped_A.get(class_id, []), grouped_B.get(class_id, [])
		if not dets_A or not dets_B:
			continue
		
		cost_matrix = np.zeros((len(dets_A), len(dets_B)))
		for i in range(len(dets_A)):
			for j in range(len(dets_B)):
				cost_matrix[i, j] = 1 - (dets_A[i]['det']['confidence'] * dets_B[j]['det']['confidence'])
		
		row_ind, col_ind = linear_sum_assignment(cost_matrix)
		for i, j in zip(row_ind, col_ind):
			matched_A.append(dets_A[i]['det'])
			matched_B.append(dets_B[j]['det'])
			matched_indices_A.add(dets_A[i]['idx'])
			matched_indices_B.add(dets_B[j]['idx'])

	# Confusion Matching
	if confusable_groups:
		unmatched_A = [d for i, d in enumerate(clean_A) if i not in matched_indices_A]
		unmatched_B = [d for i, d in enumerate(clean_B) if i not in matched_indices_B]
		for group in confusable_groups:
			group_dets_A = [d for d in unmatched_A if d['class'] in group]
			group_dets_B = [d for d in unmatched_B if d['class'] in group]
			if not group_dets_A or not group_dets_B:
				continue
			
			cost_matrix = np.zeros((len(group_dets_A), len(group_dets_B)))
			for i in range(len(group_dets_A)):
				for j in range(len(group_dets_B)):
					cost_matrix[i, j] = 1 - (group_dets_A[i]['confidence'] * group_dets_B[j]['confidence'])
			
			row_ind, col_ind = linear_sum_assignment(cost_matrix)
			for i, j in zip(row_ind, col_ind):
				det_A, det_B = group_dets_A[i], group_dets_B[j]
				if det_A['confidence'] >= det_B['confidence']:
					det_B['class'] = det_A['class']
				else:
					det_A['class'] = det_B['class']
				matched_A.append(det_A)
				matched_B.append(det_B)

	# Step 3: Cross-class NMS on the final pairs to resolve spatial conflicts
	final_A, final_B = _apply_cross_class_nms_on_pairs(matched_A, matched_B, cross_class_iou_thresh)

	print(f"Consensus Filter complete. Final object count: {len(final_A)}.")
	return final_A, final_B

def instance_matching(
	initial_dets: List[Detection],
	target_dets: List[Detection]
) -> List[Detection]:
	"""
	Matches target object instances to initial instances based on spatial distance.

	This module takes a consistent set of objects and establishes the final
	ordered correspondence, which is crucial for multi-instance classes.
	"""
	initial_classes = Counter(d['class'] for d in initial_dets)
	target_classes = Counter(d['class'] for d in target_dets)
	if initial_classes != target_classes:
		raise ValueError(f"Object mismatch after filtering. Initial: {initial_classes}, Target: {target_classes}")

	def get_bbox_center(det):
		box = det['bbox']
		return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

	sorted_target_list = [None] * len(initial_dets)
	
	initial_by_class = defaultdict(list)
	for i, det in enumerate(initial_dets):
		initial_by_class[det['class']].append({'idx': i, 'det': det})

	target_by_class = defaultdict(list)
	for det in target_dets:
		target_by_class[det['class']].append(det)

	for class_id, initial_dets_info in initial_by_class.items():
		target_dets_for_class = target_by_class[class_id]
		if len(initial_dets_info) == 1:
			sorted_target_list[initial_dets_info[0]['idx']] = target_dets_for_class[0]
		else:
			num_instances = len(initial_dets_info)
			cost_matrix = np.zeros((num_instances, num_instances))
			for i in range(num_instances):
				for j in range(num_instances):
					center_initial = get_bbox_center(initial_dets_info[i]['det'])
					center_target = get_bbox_center(target_dets_for_class[j])
					cost_matrix[i, j] = np.linalg.norm(center_initial - center_target)
			
			row_ind, col_ind = linear_sum_assignment(cost_matrix)
			for i, j in zip(row_ind, col_ind):
				initial_idx = initial_dets_info[i]['idx']
				sorted_target_list[initial_idx] = target_dets_for_class[j]

	return sorted_target_list

