import pandas as pd
import json
import os
import re

def parse_prediction_bboxes(prediction_str):
    """解析prediction字段中的bbox"""
    bboxes = []
    if pd.notna(prediction_str):
        # 使用正则表达式提取bbox坐标
        bbox_pattern = r'"bbox_2d":\s*\[([^\]]+)\]'
        matches = re.findall(bbox_pattern, str(prediction_str))
        for match in matches:
            coords = [float(x.strip()) for x in match.split(',')]
            if len(coords) == 4:
                bboxes.append(coords)
    return bboxes

def parse_answer_bboxes(answer_str):
    """解析answer字段中的bbox"""
    bboxes = []
    if pd.notna(answer_str) and answer_str.strip():
        bbox_parts = answer_str.split(';')
        for bbox_part in bbox_parts:
            coords = bbox_part.strip().split()
            if len(coords) == 4:
                x1, y1, x2, y2 = map(float, coords)
                bboxes.append([x1, y1, x2, y2])
    return bboxes

def convert_xlsx_to_jsonl(xlsx_file, output_jsonl_file):
    df = pd.read_excel(xlsx_file)
    
    with open(output_jsonl_file, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            # 从prediction字段解析answer
            answer_bboxes = parse_prediction_bboxes(row['prediction'])
            
            # 从answer字段解析ground_truth
            ground_truth_bboxes = parse_answer_bboxes(row['answer'])
            
            
            jsonl_data = {
                "answer": answer_bboxes,  # 从prediction解析的bbox
                "hw": [int(row['height']), int(row['width'])],
                "ground_truth": ground_truth_bboxes,  # 从answer解析的bbox
                "question": row['question'],
            }
            
            f.write(json.dumps(jsonl_data, ensure_ascii=False) + '\n')


# 使用示例
if __name__ == "__main__":
    # 替换为你的文件路径
    xlsx_file = "/mnt/tidal-alsh01/dataset/perceptionVLM/code_sunzhichao/VLMEvalKit/outputs/InternVL3-8B_grounding/T20251030_Ge017fe23/InternVL3-8B_grounding_refdrone_test_base64.xlsx"  # 替换为您的xlsx文件路径
    xlsx_file = "/mnt/tidal-alsh01/dataset/perceptionVLM/code_sunzhichao/VLMEvalKit/outputs/Qwen3-VL-4B-Instruct_Grounding/T20251103_Ge017fe23/Qwen3-VL-4B-Instruct_Grounding_refdrone_test_base64.xlsx"

    xlsx_file = "/mnt/tidal-alsh01/dataset/perceptionVLM/code_sunzhichao/VLMEvalKit/outputs/Qwen3-VL-8B-Instruct_Grounding/T20251104_Ge017fe23/Qwen3-VL-8B-Instruct_Grounding_refdrone_test_base64.xlsx"

    xlsx_file = "/mnt/tidal-alsh01/dataset/perceptionVLM/code_sunzhichao/VLMEvalKit/outputs/Qwen3-VL-8B-Instruct_Grounding_vllm/T20251104_Ge017fe23/Qwen3-VL-8B-Instruct_Grounding_vllm_refdrone_test_base64.xlsx"

    xlsx_file = "outputs/Qwen3-VL-235B-Instruct_Grounding_vllm/T20251104_Ge017fe23/Qwen3-VL-235B-Instruct_Grounding_vllm_refdrone_test_base64.xlsx"


    output_jsonl_file = "/mnt/tidal-alsh01/dataset/perceptionVLM/code_sunzhichao/VLMEvalKit/try/InternVL3-8B_grounding_refdrone_test.jsonl"

    output_jsonl_file = "/mnt/tidal-alsh01/dataset/perceptionVLM/code_sunzhichao/VLMEvalKit/try/qwen3vl-48_grounding_refdrone_test.jsonl"

    output_jsonl_file = "/mnt/tidal-alsh01/dataset/perceptionVLM/code_sunzhichao/VLMEvalKit/try/qwen3vl-8B_grounding_refdrone_test.jsonl"
    # xlsx_file = "/mnt/public/usr/sunzhichao/VLMEvalKit/outputs/deepseek_vl2_small/T202507261807_G/deepseek_vl2_small_refdrone_test.xlsx"
    # output_jsonl_file = "/mnt/public/usr/sunzhichao/mmdetection/qwen_vl/deepseekvl2_small_refdrone_test.jsonl"

    output_jsonl_file = "/mnt/tidal-alsh01/dataset/perceptionVLM/code_sunzhichao/VLMEvalKit/try/qwen3vl-8B_grounding_refdrone_test_new.jsonl"

    output_jsonl_file = "/mnt/tidal-alsh01/dataset/perceptionVLM/code_sunzhichao/VLMEvalKit/try/qwen3vl-235B_grounding_refdrone_test_new.jsonl"


    xlsx_file = "/mnt/tidal-alsh01/dataset/perceptionVLM/code_sunzhichao/VLMEvalKit/try/Qwen2.5-VL-3B-Instruct_grounding_refdrone_test.xlsx"
    output_jsonl_file = "/mnt/tidal-alsh01/dataset/perceptionVLM/code_sunzhichao/VLMEvalKit/try/qwen25vl-3B_grounding_refdrone_test_new.jsonl"


    xlsx_file = "/mnt/tidal-alsh01/dataset/perceptionVLM/code_sunzhichao/VLMEvalKit/outputs/Qwen3-VL-30B-A3B-Instruct/T20251105_Ge017fe23/Qwen3-VL-30B-A3B-Instruct_refdrone_test_base64.xlsx"
    output_jsonl_file = "/mnt/tidal-alsh01/dataset/perceptionVLM/code_sunzhichao/VLMEvalKit/try/Qwen3-VL-30B-A3B_grounding_refdrone_test_new.jsonl"

    # xlsx_file = "/mnt/tidal-alsh01/dataset/perceptionVLM/code_sunzhichao/VLMEvalKit/outputs/Qwen3-VL-8B-Instruct_Grounding_vllm/T20251106_Ge017fe23/Qwen3-VL-8B-Instruct_Grounding_vllm_RefCOCO_testA.xlsx"
    xlsx_file = "/mnt/tidal-alsh01/dataset/perceptionVLM/code_sunzhichao/VLMEvalKit/outputs/Qwen3-VL-8B-Instruct_Grounding_vllm_v2/T20251107_Ge017fe23/Qwen3-VL-8B-Instruct_Grounding_vllm_v2_RefCOCO_testA.xlsx"
    output_jsonl_file = "/mnt/tidal-alsh01/dataset/perceptionVLM/code_sunzhichao/VLMEvalKit/try/Qwen3-VL-8B-Instruct_Grounding_vllm_RefCOCO_testA.jsonl"


    xlsx_file = "/mnt/tidal-alsh01/dataset/perceptionVLM/code_sunzhichao/VLMEvalKit/outputs/Qwen3-VL-8B-Instruct_Grounding/T20251104_Ge017fe23/Qwen3-VL-8B-Instruct_Grounding_refdrone_test_base64.xlsx"

    output_jsonl_file = "/mnt/tidal-alsh01/dataset/perceptionVLM/code_sunzhichao/refdrone/try/Qwen3-VL-8B-Instruct_Grounding_refdrone.jsonl"


    xlsx_file = "/mnt/tidal-alsh01/dataset/perceptionVLM/code_sunzhichao/refdrone/VLMEvalKit/outputs/Qwen3-VL-8B-Instruct_Grounding/Qwen3-VL-8B-Instruct_Grounding_refdrone_test.xlsx"
    output_jsonl_file = "/mnt/tidal-alsh01/dataset/perceptionVLM/code_sunzhichao/refdrone/try/new_Qwen3-VL-8B-Instruct_Grounding_refdrone.jsonl"


    convert_xlsx_to_jsonl(xlsx_file, output_jsonl_file)
    print(f"转换完成！输出文件: {output_jsonl_file}")
