import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from PIL import Image
import cv2
import itertools
from collections import deque, defaultdict
from typing import List, Tuple
from core.perception.scene_perception import BBox, cal_iou

# ===============================SRI===============================

def load_model(checkpoint_path: str, num_classes: int=3, device: str='cpu'):
	# Model (identical architecture to training) ------------------------------
	# resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
	resnet = models.resnet50(weights=None)

	old_conv = resnet.conv1
	new_conv = nn.Conv2d(
		5,
		old_conv.out_channels,
		kernel_size=old_conv.kernel_size,
		stride=old_conv.stride,
		padding=old_conv.padding,
		bias=False,
	)
	with torch.no_grad():
		new_conv.weight[:, :3] = old_conv.weight          # copy RGB weights
		mean_w = old_conv.weight.mean(dim=1, keepdim=True)
		new_conv.weight[:, 3:5] = mean_w.repeat(1, 2, 1, 1)
	resnet.conv1 = new_conv

	resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
	resnet = resnet.to(device)

	# Load checkpoint ----------------------------------------------------------
	ckpt = torch.load(checkpoint_path, map_location=device)
	resnet.load_state_dict(ckpt["model_state_dict"])
	# optimizer.load_state_dict(ckpt["optimizer_state_dict"])
	start_epoch = ckpt["epoch"] + 1
	best_val_acc = ckpt["best_val_acc"]
	global_step = ckpt.get("global_step", {"train": 0, "val": 0})
	print(f"\n🔄 Loaded checkpoint {checkpoint_path}")

	resnet.eval()
	return resnet

def union_box(box1: BBox, box2: BBox) -> BBox:
	"""Return the bounding rectangle that encloses *both* input boxes."""
	xmin = min(box1[0], box2[0])
	ymin = min(box1[1], box2[1])
	xmax = max(box1[2], box2[2])
	ymax = max(box1[3], box2[3])
	return xmin, ymin, xmax, ymax

def bbox_map(mask: np.ndarray) -> np.ndarray:
	"""Binary image with ones inside the bounding box of *mask*."""
	y, x = np.where(mask > 0)
	if len(x) == 0:
		return np.zeros_like(mask, dtype=np.uint8)
	x1, x2 = x.min(), x.max()
	y1, y2 = y.min(), y.max()
	out = np.zeros_like(mask, dtype=np.uint8)
	out[y1 : y2 + 1, x1 : x2 + 1] = 1
	return out

def classify_pair(vec1: np.ndarray, vec2: np.ndarray) -> str:
	"""Return “mask2_parent”, “mask1_parent” or “none”."""
	v1 = vec1.copy()
	v1[0], v1[1] = v1[1], v1[0]
	combined = v1 + vec2
	onehot = np.zeros_like(combined, dtype=int)
	onehot[np.argmax(combined)] = 1
	if onehot[0] == 1:
		return "mask2_parent"
	if onehot[1] == 1:
		return "mask1_parent"
	return "none"

def bbox_to_slice(bbox):
	x1, y1, x2, y2 = map(int, bbox)
	return slice(y1, y2), slice(x1, x2)

def square_bbox(bbox, img_w, img_h):
	x1, y1, x2, y2 = map(float, bbox)
	w, h = x2 - x1, y2 - y1
	cx, cy = x1 + w / 2, y1 + h / 2
	side = max(w, h)
	half = side / 2
	new_x1, new_y1 = cx - half, cy - half
	new_x2, new_y2 = cx + half, cy + half

	new_x1, new_y1 = max(0, new_x1), max(0, new_y1)
	new_x2, new_y2 = min(img_w, new_x2), min(img_h, new_y2)

	new_w, new_h = new_x2 - new_x1, new_y2 - new_y1
	if new_w != side:
		diff = side - new_w
		if new_x1 - diff >= 0:
			new_x1 -= diff
		else:
			new_x2 += diff
	if new_h != side:
		diff = side - new_h
		if new_y1 - diff >= 0:
			new_y1 -= diff
		else:
			new_y2 += diff
	return [int(round(v)) for v in (new_x1, new_y1, new_x2, new_y2)]

def _filter_transitive_pairs(edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
	adj: dict[int, set[int]] = defaultdict(set)
	for p, c in edges:
		adj[p].add(c)

	keep = []
	for parent, child in edges:
		queue = deque(adj[parent] - {child})
		visited = set(queue)
		redundant = False
		while queue and not redundant:
			node = queue.popleft()
			if node == child:
				redundant = True
				break
			queue.extend(n for n in adj[node] if n not in visited)
			visited.update(adj[node])
		if not redundant:
			keep.append((parent, child))
	return keep

def compute_scene_relation_graph(
    image_path: str,
    bboxes: list,
    model,
    iou_threshold: float = 0.04,
    device: str = 'cpu',
    target_size: tuple = (224, 224),
) -> list:
    """
    Computes parent-child relations for a scene given an image and a list of bounding boxes.

    Args:
        image_path (str): Path to the scene image.
        bboxes (list): List of bounding boxes (each as [xmin, ymin, xmax, ymax]).
        model: The SRI model.
        iou_threshold (float): IoU threshold for considering pairs.
        device (str): Device for model inference.
        target_size (tuple): Size to which crops are resized.

    Returns:
        List[Tuple[int, int]]: List of (parent_idx, child_idx) pairs.
    """
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    img_arr = np.asarray(img, dtype=np.float32) / 255.0

    ids = list(range(len(bboxes)))
    parent_pairs = []

    for id1, id2 in itertools.combinations(ids, 2):
        b1 = tuple(map(float, bboxes[id1]))
        b2 = tuple(map(float, bboxes[id2]))
        un_mask = union_box(b1, b2)

        if cal_iou(b1, b2) <= iou_threshold:
            continue

        mask1 = np.zeros((img_h, img_w), dtype=np.float32)
        mask2 = np.zeros((img_h, img_w), dtype=np.float32)
        mask1[bbox_to_slice(b1)] = 1.0
        mask2[bbox_to_slice(b2)] = 1.0

        sq_bbox = square_bbox(un_mask, img_w, img_h)
        s = bbox_to_slice(sq_bbox)
        rgb_crop = img_arr[s]
        m1_crop = mask1[s][..., None]
        m2_crop = mask2[s][..., None]

        rgb_resized = cv2.resize(rgb_crop, target_size, interpolation=cv2.INTER_LINEAR)
        m1_resized = cv2.resize(m1_crop, target_size, interpolation=cv2.INTER_NEAREST)
        m2_resized = cv2.resize(m2_crop, target_size, interpolation=cv2.INTER_NEAREST)

        stacked12 = np.concatenate([rgb_resized, m1_resized[..., None], m2_resized[..., None]], axis=-1)
        stacked21 = np.concatenate([rgb_resized, m2_resized[..., None], m1_resized[..., None]], axis=-1)

        tensor12 = torch.from_numpy(stacked12).permute(2, 0, 1).unsqueeze(0).to(device)
        tensor21 = torch.from_numpy(stacked21).permute(2, 0, 1).unsqueeze(0).to(device)

        v12 = model(tensor12).squeeze(0).cpu().detach().numpy()
        v21 = model(tensor21).squeeze(0).cpu().detach().numpy()
        relation = classify_pair(v12, v21)

        if relation == "mask2_parent":
            parent_pairs.append((id2, id1))
        elif relation == "mask1_parent":
            parent_pairs.append((id1, id2))

    parent_pairs = _filter_transitive_pairs(parent_pairs)
    return parent_pairs
