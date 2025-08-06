import torch

# bs x seqlen
num_tokens = 4096
hidden_size = 32
global_num_experts = 256
top_k = 1

hidden_states = torch.rand((num_tokens, hidden_size), dtype=torch.float32)
router_weight = torch.rand((hidden_size, global_num_experts), dtype=torch.float32)
router_logits = torch.matmul(hidden_states, router_weight)
print(f"[+] router_logits: {router_logits.shape}")
topk_weights = router_logits.sigmoid()
topk_weights, topk_ids = torch.topk(topk_weights.to(torch.float32), k=top_k, dim=-1, sorted=False)
print(f"[+] topk_weights: {topk_weights.shape}")
print(f"[+] topk_ids: {topk_ids.shape}")
print(topk_ids)
exit()
row_idx_len = num_tokens * top_k

row_idx = torch.arange(0, row_idx_len, dtype=torch.int32).view(top_k, -1).permute(1, 0).contiguous() % num_tokens
print(f"[+] row_idx: {row_idx.shape}")
print(row_idx)

def fake_npu_moe_init_routing(x, row_idx, expert_idx, active_num):
    num_rows, k = expert_idx.shape
    flat_expert_idx = expert_idx.flatten()
    flat_row_idx = row_idx.flatten()

    sorted_expert_idx, sorted_indices = torch.sort(flat_expert_idx) # 对 topk_ids 进行排序
    sorted_row_idx = flat_row_idx[sorted_indices]

    expanded_x = x[sorted_row_idx]
    expanded_row_idx = torch.empty_like(sorted_indices)
    expanded_row_idx[sorted_indices] = torch.arange(len(sorted_indices), dtype=torch.int64)

    return expanded_x[:active_num], expanded_row_idx[:active_num], sorted_expert_idx[:active_num]

hidden_states, expanded_row_idx, expanded_expert_idx = fake_npu_moe_init_routing(
            hidden_states,
            row_idx=row_idx,
            expert_idx=topk_ids,
            active_num=num_tokens*top_k)
print(f"[+] hidden_states: {hidden_states.shape}")
global_expert_tokens = torch.bincount(expanded_expert_idx, minlength=global_num_experts)

print(global_expert_tokens.shape)
print(global_expert_tokens)

world_size = 32
scatter_sizes = global_expert_tokens.view(world_size, -1).sum(-1)
print(scatter_sizes)
print(scatter_sizes.shape)
