import torch

def pad_topk_ids(topk_ids, expert_num, ep_size, max_row_per_ep_rank):
    topk_ids = topk_ids.to(torch.int32)
    num_expert_per_ep = expert_num // ep_size
    invalid_value = expert_num

    expert_to_ep = torch.arange(expert_num, device=topk_ids.device) // num_expert_per_ep
    ep_ranks = expert_to_ep[topk_ids]

    total_slots = ep_size * max_row_per_ep_rank
    output = torch.full((total_slots,), invalid_value, dtype=torch.int32, device=topk_ids.device)

    for i in range(ep_size):
        mask = (ep_ranks == i)
        num = mask.sum()
        if num == 0:
            continue
        pos = torch.arange(num, device=topk_ids.device)
        if num > max_row_per_ep_rank:
            pos = pos[:max_row_per_ep_rank]
            mask = mask.nonzero(as_tuple=True)[0][:max_row_per_ep_rank]
        else:
            mask = mask.nonzero(as_tuple=True)[0]
        offset = i * max_row_per_ep_rank
        output[offset:offset+pos.numel()] = topk_ids[mask]

    return output

expert_num = 16
ep_size = 4
max_row_per_ep_rank = 8
num_tokens = 6
top_k = 4
topk_ids = torch.tensor([0,0,1,1,1,2,3,3,3,4,4,5,5,6,8,9,10,10,10,11,15], dtype=torch.int32)
result = pad_topk_ids(topk_ids, expert_num, ep_size, max_row_per_ep_rank)
print(result.tolist())

