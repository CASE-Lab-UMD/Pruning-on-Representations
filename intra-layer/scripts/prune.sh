#!/bin/bash

# model_name="Qwen/Qwen2.5-7B-Instruct"
# model_name="meta-llama/Meta-Llama-3-8B-Instruct"
model_name=mistralai/Mistral-7B-Instruct-v0.3
root="your_model_root_path"
save_root="your_model_output_path"
model=$root/$model_name
sparsity_ratio=0.5

# Define function to run python command
run_python_command () {

    echo $4

    python main.py \
    --model $model \
    --prune_method $1 \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type $2 \
    --save $3 \
    --save_model $4
}


# wanda pruning method
echo "Running with wanda pruning method"
run_python_command "wanda" "unstructured" "$save_root/${model_name}/unstructured/wanda/"  "$save_root/${model_name}/unstructured/wanda/"
run_python_command "wanda" "2:4" "$save_root/${model_name}/2-4/wanda/" "$save_root/${model_name}/2-4/wanda/"
run_python_command "wanda" "4:8" "$save_root/${model_name}/4-8/wanda/" "$save_root/${model_name}/4-8/wanda/"
echo "Finished wanda pruning method"

# sparsegpt pruning method
echo "Running with sparsegpt pruning method"
run_python_command "sparsegpt" "unstructured" "$save_root/${model_name}/unstructured/sparsegpt/" "$save_root/${model_name}/unstructured/sparsegpt/"
run_python_command "sparsegpt" "2:4" "$save_root/${model_name}/2-4/sparsegpt/" "$save_root/${model_name}/2-4/sparsegpt/"
run_python_command "sparsegpt" "4:8" "$save_root/${model_name}/4-8/sparsegpt/" "$save_root/${model_name}/4-8/sparsegpt/"
# echo "Finished sparsegpt pruning method"

# magnitude pruning method
echo "Running with magnitude pruning method"
run_python_command "magnitude" "unstructured" "$save_root/${model_name}/unstructured/magnitude/" "$save_root/${model_name}/unstructured/magnitude/"
run_python_command "magnitude" "2:4" "$save_root/${model_name}/2-4/magnitude/" "$save_root/${model_name}/2-4/magnitude/"
run_python_command "magnitude" "4:8" "$save_root/${model_name}/4-8/magnitude/" "$save_root/${model_name}/4-8/magnitude/"
echo "Finished magnitude pruning method"