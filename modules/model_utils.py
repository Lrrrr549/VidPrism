import torch
import torch.nn.functional as F
from typing import Callable, Tuple
import math

def diversity_loss_experts(expert_outputs) -> torch.Tensor:
    """
    计算所有专家之间的多样性损失
    """
    # 计算每对专家输出的余弦相似性
    num_experts = len(expert_outputs)
    if num_experts <= 1:
        # 只有一个专家或更少时，多样性无意义，返回0
        return torch.tensor(0.0, device=expert_outputs[0].device if num_experts == 1 else None)

    div_loss = 0.0
    num_pairs = 0
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            sim = F.cosine_similarity(expert_outputs[i], expert_outputs[j], dim=-1)  # [B, L_i, D]
            div_loss += (1.0 - sim.mean())  # 计算所有 token 输出之间的平均相似度
            num_pairs += 1
    if num_pairs == 0:
        return torch.tensor(0.0, device=expert_outputs[0].device)
    div_loss /= num_pairs  # 平均损失
    return div_loss


def diversity_loss(E_bnd):
    """
    E_bnd: [B, N, D]
    输出：专家多样性损失
    """
    B, N, D = E_bnd.shape
    if N <= 1:
        return torch.tensor(0.0, device=E_bnd.device)
    E_norm = F.normalize(E_bnd, p=2, dim=-1)  # [B,N,D]
    sim_matrix = torch.bmm(E_norm, E_norm.transpose(1, 2))  # [B,N,N]
    mask = 1 - torch.eye(N, device=E_bnd.device).unsqueeze(0)  # [1,N,N]
    # 只看非对角线
    loss = ((sim_matrix * mask)**2).sum() / (B * N * (N - 1))
    return loss

def expert_similarity_stats(E_bnd: torch.Tensor):
    """
    计算专家相似度矩阵的非对角线均值
    E_bnd: [B, N, D]
    return: mean_off_diag_sim (float)
    """
    B, N, D = E_bnd.shape
    # L2 归一化
    E_norm = F.normalize(E_bnd, p=2, dim=-1)    # [B, N, D]

    # 相似度矩阵
    sim_matrix = torch.bmm(E_norm, E_norm.transpose(1, 2))  # [B, N, N]

    # 构造 mask，去掉对角线
    mask = 1 - torch.eye(N, device=E_bnd.device).unsqueeze(0)  # [1, N, N]

    # 计算非对角线均值
    if N <= 1:
        return 0.0
    off_diag_mean = (sim_matrix * mask).sum() / (B * N * (N - 1))
    return off_diag_mean.item()  # 返回 Python float，便于打印

# --- 硬配对合并 ---
def bipartite_soft_matching(
        metric: torch.Tensor,
        r: int,
        class_token: bool = False
    ) -> Tuple[Callable, Callable]:
    protected = 1 if class_token else 0
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)
    if r <= 0:
        return lambda x, mode=None: x, lambda x: x

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)
        if class_token:
            scores[..., 0, :] = -math.inf
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        return x  # 这里不实现反操作

    return merge, unmerge