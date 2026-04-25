"""
基础模型评估辅助函数。
核心指标：BPB（Bits Per Byte），分词无关的评估指标。
"""
import math
import torch
import torch.distributed as dist

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """计算 Bits Per Byte（BPB）— 分词无关的评估指标。
    与简单的"平均 loss"不同，BPB 将 loss 按目标 token 表示的字节数归一化。
    这样即使改变词汇表大小，也能公平比较。

    BPB = total_nats / (ln(2) × total_bytes)
    - 普通 token 按其字节长度归一化
    - 特殊 token（如 <|bos|>）被排除（token_bytes=0）
    - 主动掩码的 token（ignore_index=-1）被排除

    token_bytes: 形状为 (vocab_size,) 的 1D 张量，每个 token ID 的字节数，
    对于不参与计数的 token 为 0。"""
    # record the losses
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        loss2d = model(x, y, loss_reduction='none') # (B, T)
        loss2d = loss2d.view(-1) # flatten
        y = y.view(-1) # flatten
        if (y.int() < 0).any(): # mps does not currently have kernel for < 0 for int64, only int32
            # slightly more complex code path if some target tokens are ignore_index (e.g. -1)
            # any target token < 0 is to be ignored: do NOT index token_bytes with negatives
            valid = y >= 0
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            # map valid targets to their byte length; ignored targets contribute 0 bytes
            num_bytes2d = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y, dtype=token_bytes.dtype)
            )
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
        else:
            # fast path: no ignored targets, safe to index directly
            num_bytes2d = token_bytes[y]
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
    # sum reduce across all ranks
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    # move both to cpu, calculate bpb and return
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    if total_bytes == 0:
        return float('inf')
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb
