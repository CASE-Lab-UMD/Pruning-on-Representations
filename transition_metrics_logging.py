import torch
import torch.nn.functional as F


def compute_and_log_transition_metrics(
    residual,
    hidden_states,
    lm_head,
    layer_idx,
    drop_n,
    temperature=1.0,
    prompt_idx=0,
    log_path=None,
    write_fn=None,
    label="attn",
    topk=5,
    tokenizer=None,
    decode_topk=False,
    topd=16,
    log_hidden_topd=True,
):
    del drop_n, prompt_idx  # kept for call-site compatibility
    assert residual.shape == hidden_states.shape, "residual & hidden_states shape mismatch!"
    if hidden_states.size()[-2] != 1:
        return

    def default_write(path, msg):
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    writer = write_fn if write_fn is not None else default_write
    assert log_path is not None, "log_path must be provided"

    def _list_float(t):
        return t.detach().float().cpu().tolist()

    def _list_int(t):
        return t.detach().cpu().tolist()

    emb_sim = F.cosine_similarity(residual, hidden_states, dim=-1, eps=1e-12)

    residual_head = lm_head(residual)
    output_head = lm_head(hidden_states)
    head_sim = F.cosine_similarity(residual_head, output_head, dim=-1, eps=1e-12)

    z = residual_head.detach()
    z2 = output_head.detach()
    dz = z2 - z

    cos_z = F.cosine_similarity(z, z2, dim=-1, eps=1e-12)
    z_real_1_minus_cos = (1 - cos_z).mean().item()

    z_norm2 = (z.norm(dim=-1, keepdim=True) ** 2 + 1e-12)
    alpha_z = (dz * z).sum(dim=-1, keepdim=True) / z_norm2
    dz_para = alpha_z * z
    dz_perp = dz - dz_para
    alpha_z_mean = alpha_z.mean().item()

    z_para_norm = dz_para.norm(dim=-1)
    z_perp_norm = dz_perp.norm(dim=-1)
    z_base_norm = z.norm(dim=-1)

    z_para_rel_t = z_para_norm / (z_base_norm + 1e-12)
    z_orth_rel_t = z_perp_norm / (z_base_norm + 1e-12)
    z_para_perp_t = z_para_norm / (z_perp_norm + 1e-12)

    z_para_rel = z_para_rel_t.mean().item()
    z_orth_rel = z_orth_rel_t.mean().item()
    z_para_perp_ratio_mean = z_para_perp_t.mean().item()
    z_para_perp_ratio_median = z_para_perp_t.median().item()
    z_ratio_of_means = (z_para_norm.mean() / (z_perp_norm.mean() + 1e-12)).item()

    h = residual.detach()
    h2 = hidden_states.detach()
    dh = h2 - h

    cos_h = F.cosine_similarity(h, h2, dim=-1, eps=1e-12)
    h_real_1_minus_cos = (1 - cos_h).mean().item()

    h_norm2 = (h.norm(dim=-1, keepdim=True) ** 2 + 1e-12)
    alpha_h = (dh * h).sum(dim=-1, keepdim=True) / h_norm2
    dh_para = alpha_h * h
    dh_perp = dh - dh_para
    alpha_h_mean = alpha_h.mean().item()

    h_para_norm = dh_para.norm(dim=-1)
    h_perp_norm = dh_perp.norm(dim=-1)
    h_base_norm = h.norm(dim=-1)

    h_para_rel_t = h_para_norm / (h_base_norm + 1e-12)
    h_orth_rel_t = h_perp_norm / (h_base_norm + 1e-12)
    h_para_perp_t = h_para_norm / (h_perp_norm + 1e-12)

    h_para_rel = h_para_rel_t.mean().item()
    h_orth_rel = h_orth_rel_t.mean().item()
    h_para_perp_ratio_mean = h_para_perp_t.mean().item()
    h_para_perp_ratio_median = h_para_perp_t.median().item()
    h_ratio_of_means = (h_para_norm.mean() / (h_perp_norm.mean() + 1e-12)).item()

    q = F.softmax(residual_head / temperature, dim=-1)
    p = F.softmax(output_head / temperature, dim=-1)
    vocab_sim = F.cosine_similarity(p, q, dim=-1, eps=1e-12)

    log_q = F.log_softmax(residual_head / temperature, dim=-1)
    real_kl = F.kl_div(log_q, p, reduction="batchmean").item()
    real_1_minus_cos = (1 - vocab_sim.mean()).item()

    k = int(topk)
    topk_p, topk_idx_p = torch.topk(p, k=k, dim=-1)
    topk_q, topk_idx_q = torch.topk(q, k=k, dim=-1)

    z_orig_topk = residual_head.gather(-1, topk_idx_p)
    z_post_topk = output_head.gather(-1, topk_idx_p)
    dz_perp_topk = dz_perp.gather(-1, topk_idx_p)
    dz_para_topk = dz_para.gather(-1, topk_idx_p)

    z_orig_median = residual_head.median().item()
    z_post_median = output_head.median().item()
    dz_median = dz.abs().median().item()
    dz_perp_median = dz_perp.abs().median().item()
    dz_para_median = dz_para.abs().median().item()

    delta = (residual_head - output_head).detach()
    delta_norm = delta.norm(dim=-1)

    def weighted_variance(x, weights):
        mean = (weights * x).sum(dim=-1, keepdim=True)
        return (weights * (x - mean) ** 2).sum(dim=-1)

    var_p = weighted_variance(delta, p)
    kl_estimate = (var_p / (2 * temperature * temperature)).mean().item()

    vocab_size = delta.size(-1)
    uniform_w = torch.full_like(delta, 1.0 / vocab_size)
    var_uni = weighted_variance(delta, uniform_w).mean().item()

    r = p ** 2
    r = r / (r.sum(dim=-1, keepdim=True) + 1e-12)
    var_r = weighted_variance(delta, r)
    cos_estimate = (var_r / (2 * temperature * temperature)).mean().item()

    if log_hidden_topd:
        d = int(topd)
        dh_abs = dh.abs().squeeze(1)
        torch.topk(dh_abs, k=min(d, dh_abs.size(-1)), dim=-1)

    batch_size = p.size(0)
    for b in range(batch_size):
        p_probs = topk_p[b, 0]
        p_idx = topk_idx_p[b, 0]
        q_probs = topk_q[b, 0]
        q_idx = topk_idx_q[b, 0]

        writer(log_path, f"{layer_idx} {label} top{k} p_probs[{b}]: {_list_float(p_probs)}")
        writer(log_path, f"{layer_idx} {label} top{k} p_idx  [{b}]: {_list_int(p_idx)}")
        writer(log_path, f"{layer_idx} {label} top{k} q_probs[{b}]: {_list_float(q_probs)}")
        writer(log_path, f"{layer_idx} {label} top{k} q_idx  [{b}]: {_list_int(q_idx)}")
        writer(log_path, "================================================================================")
        writer(log_path, f"{layer_idx} {label} top{k} z_orig_topk    [{b}]: {_list_float(z_orig_topk[b,0])}")
        writer(log_path, f"{layer_idx} {label} top{k} z_post_topk    [{b}]: {_list_float(z_post_topk[b,0])}")
        writer(log_path, f"{layer_idx} {label} top{k} z_orig_median  [{b}]: {z_orig_median}")
        writer(log_path, f"{layer_idx} {label} top{k} z_post_median  [{b}]: {z_post_median}")
        writer(log_path, f"{layer_idx} {label} top{k} dz_perp_topk   [{b}]: {_list_float(dz_perp_topk[b,0])}")
        writer(log_path, f"{layer_idx} {label} top{k} dz_para_topk   [{b}]: {_list_float(dz_para_topk[b,0])}")
        writer(log_path, f"{layer_idx} {label} top{k} dz_median      [{b}]: {dz_median}")
        writer(log_path, f"{layer_idx} {label} top{k} dz_perp_median [{b}]: {dz_perp_median}")
        writer(log_path, f"{layer_idx} {label} top{k} dz_para_median [{b}]: {dz_para_median}")
        writer(log_path, f"{layer_idx} {label} top{k} delta_norm     [{b}]: {delta_norm}")
        writer(log_path, "================================================================================")

        if decode_topk:
            assert tokenizer is not None
            p_tok = [tokenizer.decode(int(i), skip_special_tokens=False) for i in p_idx]
            q_tok = [tokenizer.decode(int(i), skip_special_tokens=False) for i in q_idx]
            writer(log_path, f"{layer_idx} {label} top{k} p_tok [{b}]: {p_tok}")
            writer(log_path, f"{layer_idx} {label} top{k} q_tok [{b}]: {q_tok}")

    writer(log_path, f"{layer_idx} {label} emb sim:         {emb_sim.mean().item():.4f}")
    writer(log_path, f"{layer_idx} {label} head sim:        {head_sim.mean().item():.4f}")
    writer(log_path, f"{layer_idx} {label} vocab sim:       {vocab_sim.mean().item():.4f}")
    writer(log_path, f"{layer_idx} {label} REAL_KL:         {real_kl:.6f}")
    writer(log_path, f"{layer_idx} {label} KL_estimate:     {kl_estimate:.6f}")
    writer(log_path, f"{layer_idx} {label} REAL(1-cos):     {real_1_minus_cos:.6f}")
    writer(log_path, f"{layer_idx} {label} 1-cos_estimate:  {cos_estimate:.6f}")
    writer(log_path, f"{layer_idx} {label} Uniform_Var:     {var_uni:.6f}")
    writer(log_path, f"{layer_idx} {label} Z_real(1-cos):   {z_real_1_minus_cos:.6f}")
    writer(log_path, f"{layer_idx} {label} Z_para_rel:      {z_para_rel:.6f}")
    writer(log_path, f"{layer_idx} {label} Z_orth_rel:      {z_orth_rel:.6f}")
    writer(log_path, f"{layer_idx} {label} Z_para_perp(mean):   {z_para_perp_ratio_mean:.6f}")
    writer(log_path, f"{layer_idx} {label} Z_para_perp(median): {z_para_perp_ratio_median:.6f}")
    writer(log_path, f"{layer_idx} {label} Z_ratio_of_means:    {z_ratio_of_means:.6f}")
    writer(log_path, f"{layer_idx} {label} alpha_z_mean:    {alpha_z_mean:.6f}")
    writer(log_path, f"{layer_idx} {label} H_real(1-cos):   {h_real_1_minus_cos:.6f}")
    writer(log_path, f"{layer_idx} {label} H_para_rel:      {h_para_rel:.6f}")
    writer(log_path, f"{layer_idx} {label} H_orth_rel:      {h_orth_rel:.6f}")
    writer(log_path, f"{layer_idx} {label} H_para_perp(mean):   {h_para_perp_ratio_mean:.6f}")
    writer(log_path, f"{layer_idx} {label} H_para_perp(median): {h_para_perp_ratio_median:.6f}")
    writer(log_path, f"{layer_idx} {label} H_ratio_of_means:    {h_ratio_of_means:.6f}")
    writer(log_path, f"{layer_idx} {label} alpha_h_mean:    {alpha_h_mean:.6f}")
    writer(log_path, "")
    writer(log_path, "")


def compute_and_log_similarity(*args, **kwargs):
    """Backward-compatible alias."""
    return compute_and_log_transition_metrics(*args, **kwargs)


def compute_and_log_attn_similarity(*args, **kwargs):
    """Backward-compatible alias."""
    return compute_and_log_transition_metrics(*args, **kwargs)
