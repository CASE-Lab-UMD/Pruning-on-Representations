import os
import torch
import argparse
import json
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_text_with_probabilities(prompts=None, input_ids=None, max_length=512, temperature=0.0, top_k=50, top_p=0.9, use_cache=False):
    if input_ids is None:
        input_ids = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

    do_sample = temperature != 0.0

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            output_hidden_states=True,
            output_scores=True,
            return_dict_in_generate=True,
            use_cache=use_cache,
        )

    hidden_states = outputs.hidden_states

    last_layer_per_step = [step_hidden[-1] for step_hidden in hidden_states]
    last_layer_per_step = [
        (step_hidden[-1] if step_hidden.size(1) == 1 else step_hidden[:, -1, :])
        for step_hidden in last_layer_per_step
    ]

    logits = outputs.scores
    probabilities = [torch.nn.functional.softmax(logit, dim=-1) for logit in logits]

    generated_texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    
    return generated_texts, probabilities, input_ids, last_layer_per_step, logits


def get_cos_sim(PA, PB):
    if isinstance(PA, list):
        PA = torch.tensor(PA, device='cuda')
    if isinstance(PB, list):
        PB = torch.tensor(PB, device='cuda')
    return F.cosine_similarity(PA.unsqueeze(0), PB.unsqueeze(0), dim=-1)


def calculate_stepwise_cosine_similarity(probabilities_model_1, probabilities_model_2, decimal_places=4):
    num_steps = len(probabilities_model_1)
    stepwise_similarities = []

    for i in range(num_steps):
        similarity = get_cos_sim(probabilities_model_1[i], probabilities_model_2[i])
        stepwise_similarities.append(similarity.item())
        
    stepwise_similarities = [round(p, decimal_places) for p in stepwise_similarities]
    return stepwise_similarities


def get_kl_divergence(PA, PB):
    if isinstance(PA, list):
        PA = torch.tensor(PA, device='cuda', dtype=torch.float32)
    if isinstance(PB, list):
        PB = torch.tensor(PB, device='cuda', dtype=torch.float32)
    
    PA = PA / PA.sum(dim=-1, keepdim=True)
    PB = PB / PB.sum(dim=-1, keepdim=True)

    return F.kl_div(PB.log(), PA, reduction='batchmean')


def calculate_stepwise_kl_divergence(probabilities_model_1, probabilities_model_2, decimal_places=4):
    num_steps = len(probabilities_model_1)
    stepwise_kl_divergences = []

    for i in range(num_steps):
        divergence = get_kl_divergence(probabilities_model_1[i], probabilities_model_2[i])
        stepwise_kl_divergences.append(divergence.item())

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
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    log_dir = f"./cosine_logs/{args.target_layer}/temp{args.temperature}"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"log_drop{args.drop_n}-final.txt")

    def write_log(s):
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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)

    prompts = [
        "John has twice as many books as Mary. Together they have 18 books. How many books does John have?"
    ]

    for j, layer in enumerate(model.model.layers):
        layer.lm_head = model.lm_head
        layer.drop_n = args.drop_n

    for i, prompt in enumerate(prompts):
        write_log(f"\n=== Prompt {i+1} ===\n{prompt}\n")

        torch.manual_seed(42)
        output, probabilities, _, hidden_states, logits = generate_text_with_probabilities(
            prompts=[prompt],
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_cache=use_cache
        )

        dense_text = output[0].replace(prompt, "")
        write_log("--- Dense Model Output ---")
        write_log(dense_text + "\n")

        if args.drop_n > 0:
            if target_layer in ["attn", "mlp"]:
                dropped_model_path = f"{dropped_root_path}/{model_postfix}-layer_drop_{target_layer}-discrete-drop{args.drop_n}/checkpoint"
            else: 
                dropped_model_path = f"{dropped_root_path}/block_drop/{model_postfix}-block_drop-{target_layer}-discrete-drop{args.drop_n}/checkpoint"

            config_path = os.path.join(dropped_model_path, "config.json")

            with open(config_path, "r") as f:
                config = json.load(f)

            drop_attn_list = config.get("drop_attn_list", [])
            drop_mlp_list = config.get("drop_mlp_list", [])

            write_log(f"drop_attn_list: {list(drop_attn_list)} | drop_mlp_list: {list(drop_mlp_list)}")

            for j, layer in enumerate(model.model.layers):
                if "attn" in target_layer:
                    layer.drop_attn = True if j in drop_attn_list else False
                elif "mlp" in target_layer:
                    layer.drop_mlp = True if j in drop_mlp_list else False
                layer.drop_n = args.drop_n

            torch.manual_seed(42)
            output_dropped, probabilities_dropped, _, hidden_states_dropped, logits_dropped = generate_text_with_probabilities(
                prompts=[prompt],
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                use_cache=use_cache
            )

            dropped_text = output_dropped[0].replace(prompt, "")
            write_log("--- Dropped Model Output ---")
            write_log(dropped_text + "\n")


            def cosine_sim(a, b, dim=-1, eps=1e-8):
                a_norm = a / (a.norm(dim=dim, keepdim=True) + eps)
                b_norm = b / (b.norm(dim=dim, keepdim=True) + eps)
                return (a_norm * b_norm).sum(dim=dim)
            
            def cosine_hidden_states(h1, h2):
                sims = []
                for step1, step2 in zip(h1, h2):
                    layer_sims = []
                    for l1, l2 in zip(step1, step2):
                        sim = cosine_sim(l1, l2, dim=-1).mean()
                        layer_sims.append(sim)
                    sims.append(torch.stack(layer_sims))
                return sims


            emb_cos_sim = cosine_hidden_states(hidden_states, hidden_states_dropped)
            logits_cos_sim = cosine_hidden_states(logits, logits_dropped)
            prob_cos_sim = cosine_hidden_states(probabilities, probabilities_dropped)

            kl_list = []
            for p, q in zip(probabilities, probabilities_dropped):
                kl = F.kl_div(q.log(), p, reduction='batchmean')
                kl_list.append(kl)

            kl_divergence = torch.stack(kl_list)

            def flatten_tensor_list(x):
                if isinstance(x, list):
                    x = torch.cat([t.flatten() for t in x], dim=0)
                return x.detach().cpu().flatten()

            # print(f"emb_cos_sim: {emb_cos_sim.size()}")
            emb_flat = flatten_tensor_list(emb_cos_sim)
            print(f"emb_flat: {emb_flat.size()}")

            logits_flat = flatten_tensor_list(logits_cos_sim)

            print(f"prob_cos_sim: {prob_cos_sim.size()}")
            prob_flat = flatten_tensor_list(prob_cos_sim)
            print(f"prob_flat: {prob_flat.size()}")

            kl_flat = flatten_tensor_list(kl_divergence)

            write_log("=== Cosine Similarity (Flattened) ===")
            write_log(f"emb_cos_sim_flat: {emb_flat.tolist()}")
            write_log(f"logits_cos_sim_flat: {logits_flat.tolist()}")
            write_log(f"prob_cos_sim_flat: {prob_flat.tolist()}")

            write_log("=== KL Divergence (Flattened) ===")
            write_log(f"kl_divergence_flat: {kl_flat.tolist()}")
            write_log("=" * 80)


    def weighted_variance(delta, weights):
        """
        delta: (vocab,)
        weights: (vocab,) normalized
        """
        mean = (weights * delta).sum()
        var = (weights * (delta - mean)**2).sum()
        return var


    kl_estimates = []
    cos_estimates = []

    T = 1.0

    for step, (z, z2, p) in enumerate(zip(logits, logits_dropped, probabilities)):

        # logits shape: (batch=1, vocab)
        z = z.squeeze(0)
        z2 = z2.squeeze(0)
        p = p.squeeze(0)

        delta = z2 - z

        # ---- KL estimate ----
        var_p = weighted_variance(delta, p)
        kl_hat = var_p / (2 * T * T)
        kl_estimates.append(kl_hat.item())

        # ---- 1 - cos estimate ----
        r = p**2
        r = r / r.sum()
        var_r = weighted_variance(delta, r)
        cos_hat = var_r / (2 * T * T)
        cos_estimates.append(cos_hat.item())

    print(f"kl_estimates: {kl_estimates}")
    print(f"cos_estimates: {cos_estimates}")