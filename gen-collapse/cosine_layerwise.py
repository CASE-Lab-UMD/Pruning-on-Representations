import os

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch.nn.functional as F
import numpy as np


def generate_text_with_probabilities(prompts=None, input_ids=None, max_length=512, temperature=0.0, top_k=50, top_p=0.9, use_cache=False):
    # Encode input prompt
    if input_ids is None:
        input_ids = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

    do_sample = True
    if temperature == 0.0:
        do_sample = False

    # Generate text and track probabilities
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            output_scores=True,  # Ensure that we output the token probabilities (logits)
            return_dict_in_generate=True,  # Return a dictionary containing output information
            use_cache=use_cache, 
        )

    # Get logits and calculate probabilities for each token generated
    logits = outputs.scores  # List of logits for each generated token
    probabilities = [torch.nn.functional.softmax(logit, dim=-1) for logit in logits]  # Convert logits to probabilities

    # Decode generated token IDs back to text
    generated_texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    
    return generated_texts, probabilities, input_ids


def get_cos_sim(PA, PB):
    """
    Calculate cosine similarity between two probability distributions (PA and PB).
    """    
    if isinstance(PA, list):
        PA = torch.tensor(PA, device='cuda')
    if isinstance(PB, list):
        PB = torch.tensor(PB, device='cuda')
    return F.cosine_similarity(PA.unsqueeze(0), PB.unsqueeze(0), dim=-1)
            

def calculate_stepwise_cosine_similarity(probabilities_model_1, probabilities_model_2, decimal_places=4):
    """
    计算原模型和剪枝模型每个时间步的余弦相似度
    输入：`probabilities_model_1` 和 `probabilities_model_2` 是两个模型生成的概率分布
    输出：每个时间步的余弦相似度
    """
    num_steps = len(probabilities_model_1)
    stepwise_similarities = []

    for i in range(num_steps):
        similarity = get_cos_sim(probabilities_model_1[i], probabilities_model_2[i])
        stepwise_similarities.append(similarity.item())  # 转换为标量并保存
        
    stepwise_similarities = [round(p, decimal_places) for p in stepwise_similarities]
    return stepwise_similarities


def get_kl_divergence(PA, PB):
    """
    Calculate KL divergence between two probability distributions (PA and PB).
    PA is the target distribution, and PB is the predicted distribution.
    """
    if isinstance(PA, list):
        PA = torch.tensor(PA, device='cuda', dtype=torch.float32)
    if isinstance(PB, list):
        PB = torch.tensor(PB, device='cuda', dtype=torch.float32)
    
    # Ensure the distributions are normalized (i.e., they sum to 1)
    PA = PA / PA.sum(dim=-1, keepdim=True)
    PB = PB / PB.sum(dim=-1, keepdim=True)

    # Use PyTorch's KL Divergence function (it expects log input for the first argument)
    return F.kl_div(PB.log(), PA, reduction='batchmean')  # For batch-wise average

def calculate_stepwise_kl_divergence(probabilities_model_1, probabilities_model_2, decimal_places=4):
    """
    Calculate stepwise KL divergence between two models' probability distributions.
    """
    num_steps = len(probabilities_model_1)
    stepwise_kl_divergences = []

    for i in range(num_steps):
        divergence = get_kl_divergence(probabilities_model_1[i], probabilities_model_2[i])
        stepwise_kl_divergences.append(divergence.item())  # Convert to scalar and append

    stepwise_kl_divergences = [round(p, decimal_places) for p in stepwise_kl_divergences]
    return stepwise_kl_divergences

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run text generation with different hyperparameters.")
    parser.add_argument("--model_root_path", type=str, default="your_model_root_path")
    parser.add_argument("--model_postfix", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dropped_root_path", type=str, default="you_dropped_root_path")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--drop_n", type=int, default=8)
    parser.add_argument("--target_layer", type=str, default="attn")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    log_dir = f"./cosine_logs/{args.target_layer}/temp{args.temperature}"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"log-cosine.txt")

    def write_log(s):
        print(f"log_path: {log_path}")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(s + "\n")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== Drop Experiment Log ===\n")

    model_root_path = args.model_root_path
    model_postfix = args.model_postfix
    model_name = f"{model_root_path}/{model_postfix}"
    dropped_root_path = args.dropped_root_path

    max_length = args.max_length
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    drop_n = args.drop_n
    target_layer = args.target_layer
    use_cache = True

    drop_attn_list = None
    drop_mlp_list = None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)

    prompts = [
        "John has twice as many books as Mary. Together they have 18 books. How many books does John have?"
    ]

    # attach lm_head to each layer
    for j, layer in enumerate(model.model.layers):
        layer.lm_head = model.lm_head
        layer.drop_n = args.drop_n

    for i, prompt in enumerate(prompts):
        print(f"prompt: {prompt}")
        write_log(f"\n=== Prompt {i+1} ===\n{prompt}\n")

        # Dense model output
        output, probabilities, input_ids = generate_text_with_probabilities(
            prompts=[prompt],
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_cache=use_cache
        )

        dense_text = output[0].replace(prompt, "")
        print("\nGenerated Output from Dense Model:")
        print(dense_text)

        write_log("--- Dense Model Output ---")
        write_log(dense_text + "\n")

        # Drop experiments
        if args.drop_n > 0:
            BIAS = np.array(range(0, model.config.num_hidden_layers))

            for bias in BIAS:
                drop_attn_list = range(model.config.num_hidden_layers - args.drop_n - bias,
                                    model.config.num_hidden_layers - bias) if "attn" in target_layer else []

                drop_mlp_list = range(model.config.num_hidden_layers - args.drop_n - bias,
                                    model.config.num_hidden_layers - bias) if "mlp" in target_layer else []

                if len(drop_attn_list) > 0 and min(drop_attn_list) < 0:
                    continue
                if len(drop_mlp_list) > 0 and min(drop_mlp_list) < 0:
                    continue

                print(f"===drop_attn_list: {drop_attn_list} | drop_mlp_list: {drop_mlp_list}")
                write_log(f"==============================================================================")
                write_log(f"drop_attn_list: {list(drop_attn_list)} | drop_mlp_list: {list(drop_mlp_list)}")

                # Apply drop mask to model layers
                for j, layer in enumerate(model.model.layers):
                    if "attn" in target_layer:
                        layer.drop_attn = True if j in drop_attn_list else False
                    elif "mlp" in target_layer:
                        layer.drop_mlp = True if j in drop_mlp_list else False
                    layer.drop_n = args.drop_n

                torch.manual_seed(42)
                output_dropped, probabilities_dropped, _ = generate_text_with_probabilities(
                    prompts=[prompt],
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    use_cache=use_cache
                )

                dropped_text = output_dropped[0].replace(prompt, "")
                print("\nGenerated Output from Dropped Model:")
                print(dropped_text)

                write_log("--- Dropped Model Output ---")
                write_log(dropped_text + "\n")

