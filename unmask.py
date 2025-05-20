import torch


num_tokens = 4096
hidden_size = 7168

dtype = torch.int32
expert_idx = torch.randint(0, 257, (num_tokens,), dtype=dtype)

mask = expert_idx != 256
hidden_stats = torch.randn((num_tokens, hidden_size), dtype=torch.float32)

hidden_stats = hidden_stats[mask]
print(hidden_stats.shape)
