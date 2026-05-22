 #!/bin/bash
export LMUData=""

MODELS=(
    "Qwen3-VL-8B-Instruct_Grounding"
    
)

datasets=(
    'refdrone_test'
)

for MODEL in "${MODELS[@]}"; do
    for dataset in "${datasets[@]}"; do

        echo "====================================================="
        echo "开始评估模型: $MODEL 数据集: $dataset"
        echo "====================================================="
        
        torchrun --master_port=29501 --nproc-per-node=4 run.py --data "$dataset" --model "$MODEL" --verbose


        echo "评估完成: $MODEL - $dataset"
        echo "====================================================="
        echo ""
    done
done

echo "所有数据集评估完成!"
