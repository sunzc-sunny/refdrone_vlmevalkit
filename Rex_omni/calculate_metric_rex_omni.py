import json
import pandas as pd
import numpy as np
import re
import torch
from torchvision.ops.boxes import box_area

def nms(boxes, iou_threshold):
    # 确保 boxes 是 numpy 数组格式
    boxes = np.array(boxes)
    selected_boxes = []

    while len(boxes) > 0:
        current_box = boxes[0]
        selected_boxes.append(current_box)

        other_boxes = boxes[1:]
        ious = compute_iou(current_box, other_boxes)

        # 选择 IoU 小于阈值的框
        boxes = np.array([box for i, box in enumerate(other_boxes) if ious[i] <= iou_threshold])

    return selected_boxes

def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = area_box + area_boxes - intersection
    return intersection / union

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    if (boxes1[:, 2:] >= boxes1[:, :2]).all() == False:
        boxes1 = torch.zeros_like(boxes1)
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def resize_bbox(bbox, orig_h, orig_w, patch_size=28):
    """调整bbox坐标到patch_size的倍数尺寸"""

    # 调整bbox坐标
    adjusted_bbox = [
        bbox[0] * orig_w / 1000,  # x1
        bbox[1] * orig_h / 1000,  # y1
        bbox[2] * orig_w / 1000,  # x2
        bbox[3] * orig_h / 1000   # y2
    ]
    
    return adjusted_bbox


def adjust_bbox_to_patch_size(bbox, orig_h, orig_w, patch_size=28):
    """调整bbox坐标到patch_size的倍数尺寸"""
    # 计算新尺寸
    new_h = round(orig_h / patch_size) * patch_size
    new_w = round(orig_w / patch_size) * patch_size
    
    # 计算缩放比例
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h
    
    # 调整bbox坐标
    adjusted_bbox = [
        bbox[0] * scale_x,  # x1
        bbox[1] * scale_y,  # y1
        bbox[2] * scale_x,  # x2
        bbox[3] * scale_y   # y2
    ]
    
    return adjusted_bbox


def parse_ground_truth(ground_truth_list, orig_h, orig_w):
    """解析并调整ground truth bbox"""
    if not ground_truth_list:
        return []
    
    adjusted_bboxes = []
    for bbox in ground_truth_list:
        if len(bbox) == 4:
            # 调整到patch_size倍数尺寸
            adjusted_bbox = adjust_bbox_to_patch_size(bbox, orig_h, orig_w)
            adjusted_bboxes.append(adjusted_bbox)
    
    return adjusted_bboxes

def parse_prediction(answer_list, orig_h, orig_w):
    """解析并调整prediction bbox"""
    if not answer_list:
        return []
    
    adjusted_bboxes = []
    for bbox in answer_list:
        if len(bbox) == 4:
            # 调整到patch_size倍数尺寸
            adjusted_bbox = adjust_bbox_to_patch_size(bbox, orig_h, orig_w)
            adjusted_bboxes.append(adjusted_bbox)
    
    return adjusted_bboxes

def resize_prediction(answer_list, orig_h, orig_w):
    """解析并调整prediction bbox"""
    if not answer_list:
        return []
    
    adjusted_bboxes = []
    for bbox in answer_list:
        if len(bbox) == 4:
            # 调整到patch_size倍数尺寸
            adjusted_bbox = resize_bbox(bbox, orig_h, orig_w)
            adjusted_bboxes.append(adjusted_bbox)
    
    return adjusted_bboxes

def load_jsonl(file_path):
    """加载jsonl文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON line: {e}")
                    continue
    return data

def evaluate_jsonl_data(jsonl_file):
    """评测jsonl文件中的数据"""
    # 加载jsonl文件
    data = load_jsonl(jsonl_file)
    
    instance_nt = {'TP': 0, 'FN': 0, 'FP': 0, 'TN': 0}
    image_nt = {'TP': 0, 'FN': 0, 'FP': 0, 'TN': 0}
    
    error_num = 0
    
    for idx, item in enumerate(data):
        # 获取原始尺寸
        orig_h, orig_w = item['hw']
        
        # 解析ground truth和prediction

        pred_bboxes = item.get('answer', [])
        gt_bboxes = item.get('ground_truth', [])[0]

        # 处理预测结果
        wrong_flag = False
        if len(pred_bboxes) == 0:
            wrong_flag = True
        else:
            try:
                # 应用NMS
                if len(pred_bboxes) == 1:
                    filtered_boxes = pred_bboxes
                else:
                    filtered_boxes = nms(pred_bboxes, 0.3)
                
                filtered_boxes = torch.as_tensor(filtered_boxes).view(-1, 4).to(dtype=torch.float32)
                
            except Exception as e:
                print(f"Error processing predictions for item {idx}: {e}")
                error_num += 1
                wrong_flag = True
        
        # 处理ground truth
        target = gt_bboxes if gt_bboxes else [[0, 0, 0, 0]]
        no_target_flag = (len(gt_bboxes) == 0) or (gt_bboxes == [[0, 0, 0, 0]])
        
        # 评测逻辑
        if wrong_flag:
            if no_target_flag:
                image_nt['TN'] += 1
                instance_nt['TN'] += 1
            else:
                image_nt['FN'] += 1
                instance_nt['FN'] += len(gt_bboxes)
        else:
            if no_target_flag:
                # 检查是否有非零预测
                predict_non = True
                for filtered_box in filtered_boxes:
                    if torch.all(filtered_box != torch.tensor([0, 0, 0, 0])):
                        predict_non = False
                        break
                
                if predict_non:
                    image_nt['TN'] += 1
                    instance_nt['TN'] += 1
                else:
                    image_nt['FP'] += 1
                    instance_nt['FP'] += len(filtered_boxes)
            else:
                # 计算IoU并匹配
                TP = 0
                gt_bbox_tensor = torch.tensor(gt_bboxes, dtype=torch.float32)
                
                # 检查预测是否为空
                predict_non = True
                for filtered_box in filtered_boxes:
                    if torch.all(filtered_box != torch.tensor([0, 0, 0, 0])):
                        predict_non = False
                        break
                
                if predict_non:
                    # 预测为空但有GT
                    instance_nt['FN'] += len(gt_bboxes)
                    image_nt['FN'] += 1
                else:
                    print("filtered_boxes", filtered_boxes)
                    print("gt_bbox_tensor", gt_bbox_tensor)
                    # 计算GIoU
                    giou = generalized_box_iou(filtered_boxes, gt_bbox_tensor)
                    
                    num_prediction = filtered_boxes.shape[0]
                    num_gt = gt_bbox_tensor.shape[0]
                    
                    # 匹配预测和GT
                    for i in range(min(num_prediction, num_gt)):
                        top_value, top_index = torch.topk(giou.flatten(0, 1), 1)
                        if top_value < 0.5:  # IoU阈值
                            break
                        else:
                            top_index_x = top_index // num_gt
                            top_index_y = top_index % num_gt
                            TP += 1
                            giou[top_index_x[0], :] = 0.0
                            giou[:, top_index_y[0]] = 0.0
                    
                    FP = num_prediction - TP
                    FN = num_gt - TP
                    
                    instance_nt['TP'] += TP
                    instance_nt['FP'] += FP
                    instance_nt['FN'] += FN
                    
                    # 图像级别评测
                    f_1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
                    if f_1 >= 1.0:
                        image_nt['TP'] += 1
                    else:
                        image_nt['FN'] += 1
    
    print("instance_acc", instance_nt)
    print("image_acc", image_nt)
    print("Error count:", error_num)
    
    # 计算最终指标
    results = {
        'instance_F1_score': 2*instance_nt['TP'] / (2*instance_nt['TP'] + instance_nt['FP'] + instance_nt['FN']) if (2*instance_nt['TP'] + instance_nt['FP'] + instance_nt['FN']) > 0 else 0,
        'instance_acc': (instance_nt['TP'] + instance_nt['TN'])/ (instance_nt['TP'] + instance_nt['FN'] + instance_nt['FP'] + instance_nt['TN']) if (instance_nt['TP'] + instance_nt['FN'] + instance_nt['FP'] + instance_nt['TN']) > 0 else 0,
        'image_F1_score': 2*image_nt['TP'] / (2*image_nt['TP'] + image_nt['FP'] + image_nt['FN']) if (2*image_nt['TP'] + image_nt['FP'] + image_nt['FN']) > 0 else 0,
        'image_acc': (image_nt['TP'] + image_nt['TN'])/ (image_nt['TP'] + image_nt['FN'] + image_nt['FP'] + image_nt['TN']) if (image_nt['TP'] + image_nt['FN'] + image_nt['FP'] + image_nt['TN']) > 0 else 0,
    }
    
    print("Results:", results)
    return results

if __name__ == "__main__":
    # 指定jsonl文件路径

    jsonl_file = "rex_omni_refdrone_test.jsonl"


    evaluate_jsonl_data(jsonl_file)
