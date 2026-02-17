import os

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import json

def log_subword_probs(prob_orig, prob_drop, sub_token_ids, tokenizer):
    # last step
    p1 = prob_orig[-1].squeeze(0)   # [V]
    p2 = prob_drop[-1].squeeze(0)   # [V]

    write_log("\n===== SUBWORD PROBABILITIES (Original vs Dropped) =====")
    print("\n===== SUBWORD PROBABILITIES (Original vs Dropped) =====")

    for tid in sub_token_ids:
        token = tokenizer.decode([tid])
        po = p1[tid].log().item()
        pd = p2[tid].log().item()

        s = f"{token:<10} | orig={po:.6f} | drop={pd:.6f}"
        print(s)
        write_log(s)


def generate_text_with_probabilities(prompts=None, input_ids=None,
                                     max_length=512, temperature=0.0,
                                     top_k=50, top_p=0.9, use_cache=False):
    if input_ids is None:
        input_ids = tokenizer(prompts, return_tensors="pt",
                              padding=True, truncation=True).input_ids.to(device)

    do_sample = temperature != 0.0

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=use_cache,
        )

    # ========= logits =========
    logits = outputs.logits                # [B, L, V]
    last_logits = logits[:, -1, :]         # last token logits  [B, V]

    # ========= probability =========
    probabilities = torch.softmax(last_logits, dim=-1)

    # ========= hidden states =========
    # outputs.hidden_states is a tuple(len = num_layers+1)
    # each: [B, L, D]
    hidden_states = [h[:, -1, :] for h in outputs.hidden_states]

    generated_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    return generated_texts, probabilities, input_ids, hidden_states, logits


def log_top_tokens_with_dropped(prob_orig, prob_drop, tokenizer, top_k=10):
    # last step probabilities
    p1 = prob_orig[-1].squeeze(0)      # original probs [V]
    p2 = prob_drop[-1].squeeze(0)      # dropped probs  [V]

    values, indices = torch.topk(p1, k=top_k, dim=-1)
    tokens = []

    write_log("\n===== TOP TOKENS (Original vs Dropped) =====")
    for v, idx in zip(values, indices):
        idx = idx.item()
        token = repr(tokenizer.decode([idx]))
        p_orig = v.item()
        p_drop = p2[idx].item()

        s = f"{token:<15} | orig={p_orig:.6f} | drop={p_drop:.6f}"
        print(s)
        write_log(s)
        tokens.append(token)

    write_log(f"\n===== TOP TOKENS {tokens} =====")



def get_top_tokens(prob_tensor, tokenizer, top_k=10):
    probs = prob_tensor[-1].squeeze(0)   # last step, shape: [Vocab]
    values, indices = torch.topk(probs, k=top_k, dim=-1)

    tokens = [tokenizer.decode([i]) for i in indices.tolist()]
    probs = values.tolist()
    return tokens, probs


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
    parser.add_argument("--mode", type=str, default="default", help="# FIX: mode tag used in log filename")

    args = parser.parse_args()

    log_dir = f"./cosine_logs/{args.target_layer}_temp{args.temperature}"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"log_drop{args.drop_n}-{args.mode}-subspace.txt")  # FIX: args.mode is now explicitly defined

    def write_log(s):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(s + "\n")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== Drop Experiment Log ===\n")

    model_root_path = args.model_root_path
    model_postfix = args.model_postfix
    model_name = f"{model_root_path}/{model_postfix}"
    dropped_root_path = args.dropped_root_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)


    prompts = [
        """Which animal is a mammal?
        Choose the correct answer:
        a) Snake
        b) Frog
        c) Dog
        d) Lizard""",
    ]

    for j, layer in enumerate(model.model.layers):
        layer.lm_head = model.lm_head
        layer.drop_n = args.drop_n

    for prompt in prompts:

        write_log(prompt)

        for j, layer in enumerate(model.model.layers):
            layer.drop_attn = False
            layer.drop_mlp = False

        torch.manual_seed(42)
        output, probabilities, _, hidden_states, logits = generate_text_with_probabilities(
            prompts=[prompt],
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            use_cache=True, 
        )

        dense_text = output[0].replace(prompt, "")
        write_log(dense_text)

        if args.drop_n > 0:
            if args.target_layer in ["attn", "mlp"]:
                dropped_model_path = f"{dropped_root_path}/{model_postfix}-layer_drop_{args.target_layer}-discrete-drop{args.drop_n}/checkpoint"
            else:
                dropped_model_path = f"{dropped_root_path}/block_drop/{model_postfix}-block_drop-{args.target_layer}-discrete-drop{args.drop_n}/checkpoint"

            with open(os.path.join(dropped_model_path, "config.json"), "r") as f:
                config = json.load(f)

            drop_attn_list = config.get("drop_attn_list", [])
            drop_mlp_list = config.get("drop_mlp_list", [])

            write_log(f"drop_attn_list: {list(drop_attn_list)} | drop_mlp_list: {list(drop_mlp_list)}")

            for j, layer in enumerate(model.model.layers):
                if "attn" in args.target_layer:
                    layer.drop_attn = True if j in drop_attn_list else False
                elif "mlp" in args.target_layer:
                    layer.drop_mlp = True if j in drop_mlp_list else False
                layer.drop_n = args.drop_n

            torch.manual_seed(42)
            output_dropped, probabilities_dropped, _, hidden_states_dropped, logits_dropped = generate_text_with_probabilities(
                prompts=[prompt],
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                use_cache=True
            )

            option_tokens = [" a", " b", " c", " d"]
            sub_token_ids = []

            for t in option_tokens:
                ids = tokenizer.encode(t, add_special_tokens=False)
                if len(ids) == 0:
                    raise ValueError(f"Tokenizer failed for {t}")
                sub_token_ids.append(ids[-1])

            print("Subspace token ids:", sub_token_ids)
            write_log(f"Subspace token ids: {sub_token_ids}")

            sub_cos_logits = []
            sub_cos_prob = []
            sub_kl = []

            global_cos_logits = []
            global_cos_prob = []
            global_kl = []

            z  = logits[:, -1, :].squeeze(0)          # [V]
            z2 = logits_dropped[:, -1, :].squeeze(0)  # [V]

            p  = probabilities.squeeze(0)             # [V]
            p2 = probabilities_dropped.squeeze(0)     # [V]

            global_cos_logits = [
                F.cosine_similarity(z.unsqueeze(0), z2.unsqueeze(0), dim=-1).item()
            ]

            global_cos_prob = [
                F.cosine_similarity(p.unsqueeze(0), p2.unsqueeze(0), dim=-1).item()
            ]

            global_kl = [
                F.kl_div(p2.log(), p, reduction="batchmean").item()
            ]

            # ---------------- SUBSPACE ----------------
            sub_z  = z[sub_token_ids]
            sub_z2 = z2[sub_token_ids]

            sub_p  = p[sub_token_ids]
            sub_p2 = p2[sub_token_ids]

            # normalize inside subspace
            sub_p  = sub_p  / sub_p.sum()
            sub_p2 = sub_p2 / sub_p2.sum()

            sub_cos_logits = [
                F.cosine_similarity(sub_z.unsqueeze(0), sub_z2.unsqueeze(0), dim=-1).item()
            ]

            sub_cos_prob = [
                F.cosine_similarity(sub_p.unsqueeze(0), sub_p2.unsqueeze(0), dim=-1).item()
            ]

            sub_kl = [
                F.kl_div(sub_p2.log(), sub_p, reduction="batchmean").item()
            ]


            print("\n===== GLOBAL =====")
            print("Global logits cosine:", global_cos_logits)
            print("Global prob cosine:", global_cos_prob)
            print("Global KL:", global_kl)

            print("\n===== SUBSPACE (a/b/c/d) =====")
            print("Subspace logits cosine:", sub_cos_logits)
            print("Subspace prob cosine:", sub_cos_prob)
            print("Subspace KL:", sub_kl)

            write_log("\n===== GLOBAL =====")
            write_log(f"Global logits cosine: {global_cos_logits}")
            write_log(f"Global prob cosine: {global_cos_prob}")
            write_log(f"Global KL: {global_kl}")

            log_top_tokens_with_dropped(probabilities, probabilities_dropped, tokenizer, top_k=30)

            write_log("\n===== SUBSPACE (a/b/c/d) =====")
            write_log(f"Subspace logits cosine: {sub_cos_logits}")
            write_log(f"Subspace prob cosine: {sub_cos_prob}")
            write_log(f"Subspace KL: {sub_kl}")
            write_log("=" * 80)

            log_subword_probs(probabilities, probabilities_dropped, sub_token_ids, tokenizer)
