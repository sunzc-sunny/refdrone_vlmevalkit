#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用 Rex Omni 模型进行 Grounding 任务的批量处理
参考 OpenAI API 调用代码的文件处理逻辑
"""

import os
import json
import base64
from io import BytesIO
import pandas as pd
from PIL import Image
import torch

from rex_omni import RexOmniVisualize, RexOmniWrapper


class RexOmniGroundingProcessor:
    def __init__(self, model_path, backend="transformers"):
        """
        初始化 Rex Omni 模型
        
        Args:
            model_path: 模型路径
            backend: 推理后端，"transformers" 或 "vllm"
        """
        print("Loading Rex Omni model...")
        self.rex_model = RexOmniWrapper(
            model_path=model_path,
            backend=backend,
            max_tokens=4096,
            temperature=0.0,
            top_p=0.05,
            top_k=1,
            repetition_penalty=1.05,
        )
        print("Model loaded successfully!")
    
    def base64_to_pil(self, base64_str):
        """
        将 base64 字符串转换为 PIL Image
        
        Args:
            base64_str: base64 编码的图像字符串
            
        Returns:
            PIL Image 对象
        """
        try:
            image_data = base64.b64decode(base64_str)
            image = Image.open(BytesIO(image_data)).convert("RGB")
            return image
        except Exception as e:
            print(f"Error converting base64 to image: {e}")
            return None
    
    def parse_ground_truth_bbox(self, answer_str):
        """
        解析 answer 字段中的 bbox
        
        Args:
            answer_str: 格式如 "x1 y1 x2 y2" 或 "x1 y1 x2 y2; x1 y1 x2 y2"
            
        Returns:
            list of bboxes: [[x1, y1, x2, y2], ...]
        """
        bboxes = []
        if pd.notna(answer_str) and answer_str.strip():
            bbox_parts = answer_str.split(';')
            for bbox_part in bbox_parts:
                coords = bbox_part.strip().split()
                if len(coords) == 4:
                    x1, y1, x2, y2 = map(float, coords)
                    bboxes.append([x1, y1, x2, y2])
        return bboxes
    
    def load_processed_records(self, output_jsonl_path):
        """
        加载已处理的记录（用于断点续传）
        
        Returns:
            tuple: (已处理的问题集合, 已处理数量)
        """
        processed_questions = set()
        processed_count = 0
        
        if os.path.exists(output_jsonl_path):
            try:
                with open(output_jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                record = json.loads(line)
                                # 保存原始问题用于去重
                                processed_questions.add(record['question'])
                                processed_count += 1
                            except json.JSONDecodeError:
                                continue
                print(f"Found {processed_count} already processed records")
            except Exception as e:
                print(f"Error loading existing records: {e}")
        
        return processed_questions, processed_count
    
    def inference_grounding(self, image, question):
        """
        使用 Rex Omni 进行 Grounding 推理
        
        Args:
            image: PIL Image 对象
            question: 问题/描述，如 "the person on the left"
            
        Returns:
            list of bboxes: [[x1, y1, x2, y2], ...]
        """
        try:
            # 将问题作为检测类别
            # 支持多个类别，用逗号或分号分隔
            if ';' in question or ',' in question:
                categories = [cat.strip() for cat in question.replace(';', ',').split(',')]
            else:
                categories = [question.strip()]
            
            print(f"  Categories: {categories}")
            
            # 调用 Rex Omni 推理
            results = self.rex_model.inference(
                images=image,
                task="detection",
                categories=categories
            )
            
            # 解析结果
            result = results[0]
            if result["success"]:
                predictions = result["extracted_predictions"]
                print("  Predictions:", predictions)
                
                # 提取 bboxes
                # predictions 格式: 
                # {
                #     'category_name': [
                #         {'type': 'box', 'coords': [x1, y1, x2, y2]},
                #         {'type': 'box', 'coords': [x1, y1, x2, y2]},
                #         ...
                #     ]
                # }
                bboxes = []
                
                for category, detections in predictions.items():
                    print(f"    Category '{category}': {len(detections) if isinstance(detections, list) else 0} detections")
                    
                    if isinstance(detections, list):
                        for detection in detections:
                            if isinstance(detection, dict):
                                # 检查是否有 coords 字段
                                if 'coords' in detection:
                                    coords = detection['coords']
                                    if len(coords) == 4:
                                        bbox = [float(coord) for coord in coords]
                                        bboxes.append(bbox)
                                        print(f"      - Box: {bbox}")
                                # 有些格式可能直接是坐标
                                elif 'type' in detection and detection['type'] == 'box':
                                    # 可能还有其他键名
                                    for key in ['coordinates', 'bbox', 'box', 'box_2d']:
                                        if key in detection:
                                            coords = detection[key]
                                            if len(coords) == 4:
                                                bbox = [float(coord) for coord in coords]
                                                bboxes.append(bbox)
                                                print(f"      - Box: {bbox}")
                                                break
                
                print(f"  ✓ Found {len(bboxes)} bboxes in total")
                return bboxes
            else:
                print(f"  ✗ Inference failed: {result.get('error', 'Unknown error')}")
                return []
                
        except Exception as e:
            print(f"  ✗ Error during inference: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def process_tsv_file(self, tsv_file_path, output_jsonl_path, resume=True, 
                        save_visualizations=False, vis_output_dir=None):
        """
        批量处理 TSV 文件
        
        Args:
            tsv_file_path: 输入 TSV 文件路径
            output_jsonl_path: 输出 JSONL 文件路径
            resume: 是否从断点继续
            save_visualizations: 是否保存可视化结果
            vis_output_dir: 可视化结果保存目录
        """
        # 读取 TSV 文件
        print(f"Loading TSV file: {tsv_file_path}")
        df = pd.read_csv(tsv_file_path, sep='\t')
        print(f"Total records: {len(df)}")
        
        # 加载已处理的记录
        processed_questions = set()
        processed_count = 0
        
        if resume:
            processed_questions, processed_count = self.load_processed_records(output_jsonl_path)
        else:
            # 如果不续传，删除已有文件
            if os.path.exists(output_jsonl_path):
                os.remove(output_jsonl_path)
                print("Removed existing output file, starting fresh")
        
        # 创建可视化输出目录
        if save_visualizations and vis_output_dir:
            os.makedirs(vis_output_dir, exist_ok=True)
            print(f"Visualizations will be saved to: {vis_output_dir}")
        
        results = []
        skipped = 0
        processed_new = 0
        failed = 0
        
        for idx, row in df.iterrows():
            question = row['question']
            
            # 检查是否已处理
            if resume and question in processed_questions:
                skipped += 1
                if skipped % 10 == 0:
                    print(f"Skipped {skipped} already processed records...")
                continue
            
            print(f"\n[{idx + 1}/{len(df)}] Processing (New: {processed_new + 1}, Skipped: {skipped}, Failed: {failed})...")
            print(f"  Question: {question}")
            
            try:
                # 提取数据
                height = int(row['height'])
                width = int(row['width'])
                ground_truth_str = row['answer']
                base64_image = row['image']
                
                # 解析 ground truth bbox
                gt_coords = self.parse_ground_truth_bbox(ground_truth_str)
                ground_truth = [gt_coords]  # 包装成列表以保持格式一致
                
                print(f"  Ground truth: {len(gt_coords)} bboxes")
                
                # 转换 base64 到 PIL Image
                image = self.base64_to_pil(base64_image)
                if image is None:
                    print(f"  ✗ Failed to decode image, skipping...")
                    failed += 1
                    continue
                
                # 使用 Rex Omni 进行推理
                predicted_bboxes = self.inference_grounding(image, question)
                
                if not predicted_bboxes:
                    print(f"  ⚠ Warning: No bboxes predicted")
                
                # 构建结果
                result = {
                    "answer": predicted_bboxes,
                    "hw": [height, width],
                    "ground_truth": ground_truth,
                    "question": question
                }
                
                results.append(result)
                
                # 实时保存到 JSONL
                with open(output_jsonl_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
                processed_new += 1
                print(f"  ✓ Saved result")
                
                # 保存可视化（可选）
                if save_visualizations and vis_output_dir and predicted_bboxes:
                    try:
                        # 构建可视化所需的 predictions 字典
                        predictions_dict = {
                            'boxes': predicted_bboxes,
                            'labels': [question] * len(predicted_bboxes),
                            'scores': [1.0] * len(predicted_bboxes)
                        }
                        
                        vis_image = RexOmniVisualize(
                            image=image,
                            predictions=predictions_dict,
                            font_size=20,
                            draw_width=3,
                            show_labels=True,
                        )
                        
                        vis_path = os.path.join(vis_output_dir, f"{idx:05d}.jpg")
                        vis_image.save(vis_path)
                        print(f"  ✓ Visualization saved to: {vis_path}")
                    except Exception as e:
                        print(f"  ⚠ Visualization failed: {e}")
                
            except Exception as e:
                print(f"  ✗ Error processing row {idx}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
                continue
        
        # 打印统计信息
        print(f"\n{'='*60}")
        print(f"Processing Complete!")
        print(f"{'='*60}")
        print(f"Total records in TSV:       {len(df)}")
        print(f"Already processed (skipped): {skipped}")
        print(f"Newly processed:            {processed_new}")
        print(f"Failed:                     {failed}")
        print(f"Total in output file:       {processed_count + processed_new}")
        print(f"Output saved to:            {output_jsonl_path}")
        print(f"{'='*60}")
        
        # 打印详细统计
        if results:
            avg_predictions = sum(len(r['answer']) for r in results) / len(results)
            print(f"\n=== Prediction Statistics ===")
            print(f"Average predictions per image: {avg_predictions:.2f}")
            
            # 统计预测数量分布
            pred_counts = [len(r['answer']) for r in results]
            print(f"Min predictions: {min(pred_counts)}")
            print(f"Max predictions: {max(pred_counts)}")
            
            # 统计空预测
            empty_preds = sum(1 for r in results if len(r['answer']) == 0)
            print(f"Empty predictions: {empty_preds} ({empty_preds/len(results)*100:.1f}%)")
        
        return results


def main():
    """主函数"""
    
    # ============ 配置参数 ============
    
    # 模型路径
    MODEL_PATH = "/mnt/public/usr/sunzhichao/hf_hub/models--IDEA-Research--Rex-Omni"
    
    # 数据路径
    TSV_FILE_PATH = "/mnt/public/usr/sunzhichao/benchmark/refdrone_test_base64.tsv"
    OUTPUT_JSONL_PATH = "/mnt/public/usr/sunzhichao/Rex-Omni/rex_omni_refdrone_test.jsonl"
    
    # 可视化设置（可选）
    SAVE_VISUALIZATIONS = False  # 改为 True 启用可视化
    VIS_OUTPUT_DIR = "/mnt/public/usr/sunzhichao/Rex-Omni/try/rex_omni_visualizations"
    
    # 推理设置
    BACKEND = "transformers"  # 或 "vllm" 以获得更快速度
    RESUME = True  # True: 断点续传, False: 重新开始
    
    # ==================================
    
    # 初始化处理器
    processor = RexOmniGroundingProcessor(
        model_path=MODEL_PATH,
        backend=BACKEND
    )
    
    # 处理数据
    results = processor.process_tsv_file(
        tsv_file_path=TSV_FILE_PATH,
        output_jsonl_path=OUTPUT_JSONL_PATH,
        resume=RESUME,
        save_visualizations=SAVE_VISUALIZATIONS,
        vis_output_dir=VIS_OUTPUT_DIR
    )
    
    print("\n✨ All done!")


if __name__ == "__main__":
    main()
