import torch

def process_topk_ids(
    topk_ids: torch.Tensor,
    expert_num: int,
    ep_size: int,
    max_row_per_ep_rank: int,
    num_tokens: int,
    top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    original_total_elements = num_tokens * top_k
    device = topk_ids.device
    original_dtype = topk_ids.dtype

    if original_total_elements == 0:
        output_len = ep_size * max_row_per_ep_rank
        topk_ids_pad = torch.full((output_len,), expert_num, dtype=original_dtype, device=device)
        unpad_indices = torch.full((original_total_elements,), -1, dtype=torch.long, device=device)
        return topk_ids_pad, unpad_indices

    experts_per_ep_rank_val = expert_num // ep_size
    if experts_per_ep_rank_val == 0:
        raise ValueError("expert_num // ep_size is 0, which leads to division by zero in ep_rank calculation. "
                         "Ensure expert_num >= ep_size.")

    assigned_ep_rank = topk_ids // experts_per_ep_rank_val
    indices_arange = torch.arange(len(topk_ids), device=device)

    is_new_segment = torch.cat((
        torch.tensor([True], device=device),
        assigned_ep_rank[1:] != assigned_ep_rank[:-1]
    ))

    segment_start_indices_in_topk = is_new_segment.nonzero(as_tuple=True)[0]
    lookup_idx_for_segment_starts = torch.searchsorted(
        segment_start_indices_in_topk, indices_arange, right=True
    ) - 1
    start_offset_for_each_token = segment_start_indices_in_topk[lookup_idx_for_segment_starts]

    token_intra_ep_rank_idx = indices_arange - start_offset_for_each_token
    is_kept_mask = token_intra_ep_rank_idx < max_row_per_ep_rank
    original_indices_all = indices_arange
    kept_original_indices = original_indices_all[is_kept_mask]
    kept_topk_ids = topk_ids[is_kept_mask]
    kept_assigned_ep_rank = assigned_ep_rank[is_kept_mask]
    kept_token_intra_ep_rank_idx = token_intra_ep_rank_idx[is_kept_mask]
    output_len = ep_size * max_row_per_ep_rank
    topk_ids_pad = torch.full((output_len,), expert_num, dtype=original_dtype, device=device)

    if len(kept_topk_ids) > 0:
        destination_indices_in_pad = kept_assigned_ep_rank * max_row_per_ep_rank + kept_token_intra_ep_rank_idx
        topk_ids_pad[destination_indices_in_pad] = kept_topk_ids

    unpad_indices = torch.full((original_total_elements,), -1, dtype=torch.long, device=device)

    if len(kept_original_indices) > 0:
        indices_in_recovered_condensed_list = torch.arange(len(kept_original_indices), device=device, dtype=torch.long)
        unpad_indices[kept_original_indices] = indices_in_recovered_condensed_list

    return topk_ids_pad, unpad_indices


def process_topk_ids_optimized(
    topk_ids: torch.Tensor,
    expert_num: int,
    ep_size: int,
    max_row_per_ep_rank: int,
    num_tokens: int,
    top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    original_total_elements = num_tokens * top_k
    device = topk_ids.device
    original_dtype = topk_ids.dtype

    if original_total_elements == 0:
        output_len = ep_size * max_row_per_ep_rank
        topk_ids_pad = torch.full((output_len,), expert_num, dtype=original_dtype, device=device)
        unpad_indices = torch.full((original_total_elements,), -1, dtype=torch.long, device=device)
        return topk_ids_pad, unpad_indices

    experts_per_ep_rank_val = expert_num // ep_size
    if experts_per_ep_rank_val == 0:
        raise ValueError("expert_num // ep_size is 0, which leads to division by zero in ep_rank calculation. "
                         "Ensure expert_num >= ep_size.")
    assigned_ep_rank = topk_ids // experts_per_ep_rank_val
    indices_arange = torch.arange(len(topk_ids), device=device)
    is_new_segment = torch.cat((
        torch.tensor([True], device=device),
        assigned_ep_rank[1:] != assigned_ep_rank[:-1]
    ))
    temp_start_markers = torch.full_like(indices_arange, -1, dtype=indices_arange.dtype)
    temp_start_markers[is_new_segment] = indices_arange[is_new_segment]
    start_offset_for_each_token = torch.cummax(temp_start_markers, dim=0)[0]
    token_intra_ep_rank_idx = indices_arange - start_offset_for_each_token
    is_kept_mask = token_intra_ep_rank_idx < max_row_per_ep_rank
    cumsum_kept = torch.cumsum(is_kept_mask.to(torch.long), dim=0)
    indices_in_rec_cond_list_for_all = cumsum_kept - 1 
    unpad_indices = torch.where(
        is_kept_mask,
        indices_in_rec_cond_list_for_all,
        torch.tensor(-1, device=device, dtype=torch.long)
    )
    output_len = ep_size * max_row_per_ep_rank
    topk_ids_pad = torch.full((output_len,), expert_num, dtype=original_dtype, device=device)
    if len(topk_ids) > 0:
        all_destination_indices = assigned_ep_rank * max_row_per_ep_rank + token_intra_ep_rank_idx
        temp_pad_buffer = torch.full((output_len + 1,), expert_num, dtype=original_dtype, device=device)
        output_len_tensor = torch.tensor(output_len, dtype=torch.long, device=device)
        scatter_indices = torch.where(is_kept_mask, all_destination_indices, output_len_tensor)
        temp_pad_buffer.scatter_(0, scatter_indices, topk_ids)
        topk_ids_pad = temp_pad_buffer[:output_len]
    return topk_ids_pad, unpad_indices

def recover_topk_ids(
    topk_ids_pad: torch.Tensor,
    unpad_indices: torch.Tensor,
    expert_num: int,
    original_dtype: torch.dtype = torch.int32
) -> torch.Tensor:
    original_total_elements = len(unpad_indices)
    device = topk_ids_pad.device

    if original_total_elements == 0:
        return torch.empty((0,), dtype=original_dtype, device=device)

    recovered_condensed = topk_ids_pad[topk_ids_pad != expert_num]

    output_tensor = torch.full((original_total_elements,), -1, dtype=original_dtype, device=device)
    original_pos_kept_mask = unpad_indices != -1
    
    if torch.any(original_pos_kept_mask):
        original_pos_kept_indices = original_pos_kept_mask.nonzero(as_tuple=True)[0]
        indices_to_gather_from_condensed = unpad_indices[original_pos_kept_indices]
        output_tensor[original_pos_kept_indices] = recovered_condensed[indices_to_gather_from_condensed]
    return output_tensor


def pad_topk_ids(topk_ids, expert_num, ep_size, max_row_per_ep_rank):
    dtype = topk_ids.dtype
    num_expert_per_ep = expert_num // ep_size
    invalid_value = expert_num

    expert_to_ep = torch.arange(expert_num, device=topk_ids.device) // num_expert_per_ep
    ep_ranks = expert_to_ep[topk_ids]

    total_slots = ep_size * max_row_per_ep_rank
    output = torch.full((total_slots,), invalid_value, dtype=dtype, device=topk_ids.device)

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
