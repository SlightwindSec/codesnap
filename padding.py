import torch
from pad_utils import process_topk_ids, recover_topk_ids, pad_topk_ids

expert_num = 16
ep_size = 4
max_row_per_ep_rank = 8
num_tokens = 6
top_k = 4

topk_ids = torch.tensor([
    0, 0, 1, 1, 1, 2, 3, 3, 3,      # 9 elements for ep_rank 0 (IDs 0-3)
    4, 4, 5, 5, 6,                  # 5 elements for ep_rank 1 (IDs 4-7)
    8, 9, 10, 10, 10, 11,           # 6 elements for ep_rank 2 (IDs 8-11)
    12, 14, 15, 15                  # 4 elements for ep_rank 3 (IDs 12-15)
], dtype=torch.int32)

topk_ids = torch.tensor([
    0, 0, 1, 1, 1, 2, 3, 3, 3, 3, 3,      # 9 elements for ep_rank 0 (IDs 0-3)
    4, 4, 5,                  # 5 elements for ep_rank 1 (IDs 4-7)
    8, 8, 8, 9, 10, 10, 10, 11, 11,           # 6 elements for ep_rank 2 (IDs 8-11)
    12                  # 4 elements for ep_rank 3 (IDs 12-15)
], dtype=torch.int32)
assert len(topk_ids) == num_tokens * top_k, len(topk_ids)
topk_ids_pad, unpad_indices = process_topk_ids(
    topk_ids,
    expert_num,
    ep_size,
    max_row_per_ep_rank,
    num_tokens,
    top_k
)
result = pad_topk_ids(topk_ids, expert_num, ep_size, max_row_per_ep_rank)

print(result.tolist() == topk_ids_pad.tolist())

print("Input topk_ids:\n", topk_ids)
print("Processed topk_ids_pad:\n", topk_ids_pad.view(ep_size, max_row_per_ep_rank))
print("Unpad indices:\n", unpad_indices)

recovered_ids = recover_topk_ids(
    topk_ids_pad,
    unpad_indices,
    expert_num,
    original_dtype=topk_ids.dtype
)
print("Recovered topk_ids:\n", recovered_ids)

for _ in range(100):
    topk_ids = torch.randint(0, 16, (num_tokens * top_k,), dtype=torch.int32)
    topk_ids, _ = torch.sort(topk_ids)
    result = pad_topk_ids(topk_ids, expert_num, ep_size, max_row_per_ep_rank)
    topk_ids_pad, unpad_indices = process_topk_ids(topk_ids, expert_num, ep_size, max_row_per_ep_rank, num_tokens, top_k)
    assert result.tolist() == topk_ids_pad.tolist()
