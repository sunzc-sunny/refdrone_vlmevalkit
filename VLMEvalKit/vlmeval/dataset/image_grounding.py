from .image_base import ImageBaseDataset
from ..smp import *
import numpy as np
import re

# Qwen3-VL internally resizes images to multiples of this patch size
QWEN3VL_PATCH_SIZE = 32


def _round_to_patch_size(dim, patch_size=QWEN3VL_PATCH_SIZE):
    """Round image dimension to nearest patch_size multiple (minimum patch_size)."""
    return max(patch_size, round(dim / patch_size) * patch_size)


def bbox_overlaps(bboxes1,
                  bboxes2,
                  mode='iou',
                  eps=1e-6,
                  use_legacy_coordinate=False):
    assert mode in ['iou', 'iof']
    extra_length = 1. if use_legacy_coordinate else 0.
    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + extra_length) * (
        bboxes1[:, 3] - bboxes1[:, 1] + extra_length)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + extra_length) * (
        bboxes2[:, 3] - bboxes2[:, 1] + extra_length)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + extra_length, 0) * np.maximum(
            y_end - y_start + extra_length, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        union = np.maximum(union, eps)
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious


def _nms(boxes, iou_threshold=0.3):
    boxes = np.array(boxes, dtype=np.float32)
    selected = []
    while len(boxes) > 0:
        selected.append(boxes[0].tolist())
        if len(boxes) == 1:
            break
        x1 = np.maximum(boxes[0, 0], boxes[1:, 0])
        y1 = np.maximum(boxes[0, 1], boxes[1:, 1])
        x2 = np.minimum(boxes[0, 2], boxes[1:, 2])
        y2 = np.minimum(boxes[0, 3], boxes[1:, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area0 = (boxes[0, 2] - boxes[0, 0]) * (boxes[0, 3] - boxes[0, 1])
        areas = (boxes[1:, 2] - boxes[1:, 0]) * (boxes[1:, 3] - boxes[1:, 1])
        iou = inter / (area0 + areas - inter + 1e-6)
        boxes = boxes[1:][iou <= iou_threshold]
    return selected


def _box_iou_torch(boxes1, boxes2):
    import torch
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union, union


def _generalized_box_iou(boxes1, boxes2):
    import torch
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        boxes1 = torch.zeros_like(boxes1)
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = _box_iou_torch(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area


def _parse_prediction_bboxes(prediction_str):
    """Parse all bbox_2d entries from model prediction string.

    Handles three formats:
    1. JSON: {"bbox_2d": [x1, y1, x2, y2], ...}  — raw model response (no visual_grounding)
    2. Multi-bbox list: [[x1, y1, x2, y2], ...]   — visual_grounding multi-box output
    3. Single list: [x1, y1, x2, y2]              — visual_grounding single-box output

    Returned coords are in [0,1] (model.py divides raw [0,1000] output by 1000).
    """
    import ast
    bboxes = []
    s = str(prediction_str) if prediction_str is not None else ''
    if not s or s == 'nan':
        return bboxes
    # Format 1: bbox_2d JSON key (raw response, coords in [0,1000])
    for match in re.findall(r'"bbox_2d":\s*\[([^\]]+)\]', s):
        try:
            coords = [float(x.strip()) for x in match.split(',')]
            if len(coords) == 4:
                # Normalize [0,1000] → [0,1] to match visual_grounding output
                bboxes.append([c / 1000.0 for c in coords])
        except ValueError:
            pass
    if bboxes:
        return bboxes
    # Format 2 & 3: ast.literal_eval for [[...], ...] or [x, y, x, y]
    try:
        parsed = ast.literal_eval(s.strip())
        if isinstance(parsed, list) and len(parsed) > 0:
            if isinstance(parsed[0], list):
                for item in parsed:
                    if len(item) == 4:
                        bboxes.append([float(v) for v in item])
            elif len(parsed) == 4 and all(isinstance(v, (int, float)) for v in parsed):
                bboxes.append([float(v) for v in parsed])
    except (ValueError, SyntaxError):
        pass
    return bboxes


def _parse_answer_bboxes(answer_str):
    """Parse semicolon-separated, space-delimited GT bboxes from answer string."""
    bboxes = []
    s = str(answer_str) if answer_str is not None else ''
    if s and s != 'nan':
        for part in s.split(';'):
            coords = part.strip().split()
            if len(coords) == 4:
                try:
                    bboxes.append([float(c) for c in coords])
                except ValueError:
                    pass
    return bboxes


def _gt_to_resized_pixel(gt_bboxes_px, orig_h, orig_w):
    """Scale GT bboxes from original pixel space to Qwen3-VL resized pixel space.

    Qwen3-VL resizes the input image to the nearest QWEN3VL_PATCH_SIZE multiple
    before processing. GT coords (original pixels) must be scaled accordingly so
    they align with the model's coordinate reference frame.
    """
    new_h = _round_to_patch_size(orig_h)
    new_w = _round_to_patch_size(orig_w)
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h
    return [
        [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
        for x1, y1, x2, y2 in gt_bboxes_px
    ], new_h, new_w


def _pred_to_resized_pixel(pred_bboxes_01, new_h, new_w):
    """Convert predictions from [0,1] normalized to resized pixel space.

    Predictions from model.py are already divided by 1000 ([0,1]),
    and the model's [0,1000] range spans the resized image dimensions.
    """
    return [
        [x1 * new_w, y1 * new_h, x2 * new_w, y2 * new_h]
        for x1, y1, x2, y2 in pred_bboxes_01
    ]


class Grounding_RefDrone(ImageBaseDataset):
    TYPE = 'VG'

    DATASET_URL = {
        'refdrone_test': '/mnt/tidal-alsh01/dataset/perceptionVLM/benchmark/refdrone_test_base64.tsv',
        'refdrone_test_sample': '/mnt/tidal-alsh01/dataset/perceptionVLM/benchmark/refdrone_test_sample_100.tsv',
    }

    def load_data(self, dataset):
        url = self.DATASET_URL.get(dataset)
        if url and url != dataset + '.tsv':
            from vlmeval.smp import load
            return load(url)
        return super().load_data(dataset)

    @classmethod
    def evaluate(cls, eval_file, **kwargs):
        """
        Multi-bbox evaluation using GIoU-based matching (IoU threshold 0.5).

        Coordinate handling:
          - GT bboxes (original pixel space) are scaled to the Qwen3-VL resized
            image dimensions (nearest QWEN3VL_PATCH_SIZE=32 multiple).
          - Predicted bboxes ([0,1] normalized, already /1000 in model.py) are
            converted to the same resized pixel space.
          - GIoU matching is then performed in a consistent coordinate frame.

        Metrics: instance-level and image-level F1 / accuracy.
        """
        import torch

        data = load(eval_file)
        lt = len(data)

        instance_nt = {'TP': 0, 'FN': 0, 'FP': 0, 'TN': 0}
        image_nt = {'TP': 0, 'FN': 0, 'FP': 0, 'TN': 0}
        error_num = 0

        for i in range(lt):
            line = data.iloc[i]
            gt_bboxes_px = _parse_answer_bboxes(line['answer'])
            pred_bboxes_01 = _parse_prediction_bboxes(line['prediction'])

            orig_h = float(line.get('height', 1) or 1)
            orig_w = float(line.get('width', 1) or 1)

            # Scale GT from original pixel → resized pixel space
            gt_bboxes, new_h, new_w = _gt_to_resized_pixel(gt_bboxes_px, orig_h, orig_w)

            # Convert predictions from [0,1] → resized pixel space
            pred_bboxes = _pred_to_resized_pixel(pred_bboxes_01, new_h, new_w)

            no_target_flag = (len(gt_bboxes) == 0) or (gt_bboxes == [[0.0, 0.0, 0.0, 0.0]])
            wrong_flag = len(pred_bboxes) == 0

            if not wrong_flag:
                try:
                    filtered = _nms(pred_bboxes, 0.3) if len(pred_bboxes) > 1 else pred_bboxes
                    filtered_boxes = torch.as_tensor(filtered, dtype=torch.float32).view(-1, 4)
                except Exception:
                    error_num += 1
                    wrong_flag = True

            if wrong_flag:
                if no_target_flag:
                    image_nt['TN'] += 1
                    instance_nt['TN'] += 1
                else:
                    image_nt['FN'] += 1
                    instance_nt['FN'] += len(gt_bboxes)
            else:
                zero_box = torch.zeros(4, dtype=torch.float32)
                predict_non = all(torch.all(fb == zero_box) for fb in filtered_boxes)

                if no_target_flag:
                    if predict_non:
                        image_nt['TN'] += 1
                        instance_nt['TN'] += 1
                    else:
                        image_nt['FP'] += 1
                        instance_nt['FP'] += filtered_boxes.shape[0]
                else:
                    if predict_non:
                        instance_nt['FN'] += len(gt_bboxes)
                        image_nt['FN'] += 1
                    else:
                        gt_tensor = torch.tensor(gt_bboxes, dtype=torch.float32)
                        giou = _generalized_box_iou(filtered_boxes, gt_tensor)
                        num_pred = filtered_boxes.shape[0]
                        num_gt = gt_tensor.shape[0]

                        TP = 0
                        for _ in range(min(num_pred, num_gt)):
                            top_value, top_index = torch.topk(giou.flatten(), 1)
                            if top_value.item() < 0.5:
                                break
                            top_x = top_index[0] // num_gt
                            top_y = top_index[0] % num_gt
                            TP += 1
                            giou[top_x, :] = 0.0
                            giou[:, top_y] = 0.0

                        FP = num_pred - TP
                        FN = num_gt - TP
                        instance_nt['TP'] += TP
                        instance_nt['FP'] += FP
                        instance_nt['FN'] += FN

                        f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
                        if f1 >= 1.0:
                            image_nt['TP'] += 1
                        else:
                            image_nt['FN'] += 1

        denom_inst_f1 = 2 * instance_nt['TP'] + instance_nt['FP'] + instance_nt['FN']
        denom_img_f1 = 2 * image_nt['TP'] + image_nt['FP'] + image_nt['FN']
        total_inst = sum(instance_nt.values())
        total_img = sum(image_nt.values())

        results = {
            'instance_F1': 2 * instance_nt['TP'] / denom_inst_f1 if denom_inst_f1 > 0 else 0,
            'instance_acc': (instance_nt['TP'] + instance_nt['TN']) / total_inst if total_inst > 0 else 0,
            'image_F1': 2 * image_nt['TP'] / denom_img_f1 if denom_img_f1 > 0 else 0,
            'image_acc': (image_nt['TP'] + image_nt['TN']) / total_img if total_img > 0 else 0,
            'instance_counts': instance_nt,
            'image_counts': image_nt,
            'error_count': error_num,
        }

        print("instance_counts:", instance_nt)
        print("image_counts:", image_nt)
        print("error_count:", error_num)
        print("Results:", {k: v for k, v in results.items() if not k.endswith('_counts')})

        score_pth = eval_file.replace('.xlsx', '_grounding_score.json')
        with open(score_pth, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return results
