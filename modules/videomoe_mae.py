import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from modules.model_utils import diversity_loss_experts

# === Temporal Reduction ===

# -----------------------------------------------------------------------------
# Temporal Pooling
# -----------------------------------------------------------------------------
class RgSTA(nn.Module):
    def __init__(self, d_model: int, rate: int = 4, keep_k: int = 1, tau: float = 0.1,
                 alpha_mix: float = 0.7, use_rank_loss: bool = True,
                 score_temp: float = 0.2, loss_type: str = "listwise"):
        super().__init__()
        assert keep_k > 0 and keep_k <= rate, "keep_k must be between 1 and rate"
        self.rate = rate
        self.keep_k = keep_k
        self.tau = tau
        self.alpha_mix = alpha_mix

        self.metric_proj = nn.Linear(d_model, d_model)
        self.score_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1, bias=False)
        )

        self.use_rank_loss = use_rank_loss
        self.score_temp = score_temp
        self.loss_type = loss_type
        self.last_aux_loss: torch.Tensor = torch.tensor(0.0)

    def _compute_sim(self, chunk: torch.Tensor):
        B, D, w = chunk.shape
        metric = F.normalize(self.metric_proj(chunk.permute(0, 2, 1)), dim=-1)
        sim = torch.bmm(metric, metric.transpose(1, 2))
        mask = torch.eye(w, device=chunk.device, dtype=torch.bool)
        sim.masked_fill_(mask.unsqueeze(0), 0)
        return metric, sim

    def _token_scores(self, chunk: torch.Tensor):
        feat = self.metric_proj(chunk.permute(0, 2, 1))
        return self.score_head(feat).squeeze(-1)

    def _rank_loss(self, s_pred: torch.Tensor, s_tgt: torch.Tensor):
        p_pred = F.softmax(s_pred / self.score_temp, dim=-1)
        p_tgt  = F.softmax(s_tgt  / self.score_temp, dim=-1)
        kl = (p_tgt * (p_tgt.clamp_min(1e-9).log() - p_pred.clamp_min(1e-9).log())).sum(dim=-1)
        return kl.mean()

    @torch.no_grad()
    def _rest_indices(self, wlen: int, top_idx: torch.Tensor, device):
        mask = torch.ones(wlen, dtype=torch.bool, device=device)
        mask.scatter_(0, top_idx, False)
        return mask.nonzero(as_tuple=False).squeeze(-1)

    def forward(self, x: torch.Tensor, importance_prior: torch.Tensor = None,
                return_aux: bool = True):
        B, D, T = x.shape
        # print("x.shape entering pooler:", x.shape)
        out_tokens = []
        aux_losses = []

        for start in range(0, T, self.rate):
            end = min(start + self.rate, T)
            chunk = x[:, :, start:end]
            wlen = chunk.size(-1)

            if wlen <= self.keep_k:
                out_tokens.append(chunk)
                continue

            metric, sim = self._compute_sim(chunk)

            s_pred = self._token_scores(chunk)

            if self.use_rank_loss and self.training:
                if importance_prior is not None:
                    s_tgt = importance_prior[:, start:end]
                else:
                    s_tgt = (chunk.norm(dim=1) + sim.mean(dim=1)).detach()
                
                rank_loss = self._rank_loss(s_pred, s_tgt)
                aux_losses.append(rank_loss)
            
            imp_norm = chunk.norm(dim=1)
            s_mix = self.alpha_mix * s_pred + (1 - self.alpha_mix) * imp_norm

            attn = F.softmax(sim / self.tau, dim=-1)
            merged_window = []
            for b in range(B):
                k_to_keep = min(self.keep_k, wlen)
                top_idx = s_mix[b].topk(k_to_keep).indices
                kept = chunk[b, :, top_idx]
                
                if k_to_keep < wlen:
                    rest_idx = self._rest_indices(wlen, top_idx, device=x.device)
                    rest_tokens = chunk[b, :, rest_idx]
                    
                    rest_imp = attn[b][rest_idx][:, top_idx]
                    norm_w = rest_imp / (rest_imp.sum(dim=1, keepdim=True) + 1e-6)
                    merged_add = torch.mm(norm_w.t(), rest_tokens.t()).t()
                    kept = kept + merged_add / 2.0
                
                merged_window.append(kept.unsqueeze(0))
            out_tokens.append(torch.cat(merged_window, dim=0))

        out = torch.cat(out_tokens, dim=-1)

        extra_loss = torch.stack(aux_losses).mean() if len(aux_losses) > 0 else x.new_tensor(0.0)
        self.last_aux_loss = extra_loss.detach()

        if return_aux:
            return out, extra_loss
        return out


# === Cross-Scale Interaction (DBI) ===
# -----------------------------------------------------------------------------
# Cross-Expert Interaction
# -----------------------------------------------------------------------------
class Slow2FastGate(nn.Module):
    """Slow -> Fast: interpolation upsampling, channel projection, and score-weighted fusion."""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.proj = nn.Conv1d(d_in, d_out, kernel_size=1)

    def forward(self, slow_feat, fast_feat, score):
        slow_up = F.interpolate(slow_feat, size=fast_feat.size(2), mode='linear', align_corners=False)
        slow_up = self.proj(slow_up)        # -> [B,D_fast,T_fast]
        score = score.view(-1, 1, 1)        # -> [B,1,1]
        return fast_feat + score * slow_up


class Fast2SlowTConv(nn.Module):
    """Fast -> Slow: temporal convolution downsampling with score-weighted fusion."""
    def __init__(self, d_in, d_out, stride, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.tconv = nn.Conv1d(d_in, d_out, kernel_size=kernel_size,
                               stride=stride, padding=pad, bias=False)

    def forward(self, fast_feat, slow_feat, score):
        # fast_feat: (B,D_fast,T_fast)
        fast_down = self.tconv(fast_feat)  # -> [B,D_slow,T_slow]
        score = score.view(-1, 1, 1)       # -> [B,1,1]
        return slow_feat + score * fast_down


class DBI(nn.Module):
    def __init__(self, d_model, num_experts, is_slow, sampling_rates, threshold=0.3):
        super().__init__()
        self.num_experts = num_experts
        self.is_slow = is_slow
        self.sampling_rates = sampling_rates
        self.threshold = threshold

        # Score MLP for each expert pair (i, j)
        self.score_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2*d_model, d_model//4),
                nn.ReLU(inplace=True),
                nn.Linear(d_model//4, 1),
                nn.Sigmoid()
            ) for _ in range(num_experts * num_experts)
        ])

        self.slow2fast = Slow2FastGate(d_in=d_model, d_out=d_model)

        # Derive the fast-to-slow stride from the sampling-rate ratio
        self.fast2slow_ops = nn.ModuleDict()
        for i in range(num_experts):
            for j in range(num_experts):
                if not is_slow[i] and is_slow[j]:
                    stride = sampling_rates[j] // sampling_rates[i]
                    key = f"{i}->{j}"
                    self.fast2slow_ops[key] = Fast2SlowTConv(
                        d_in=d_model, d_out=d_model, stride=stride
                    )

    def forward(self, features):
        """
        features: list of [T_i,B,D]
        Returns: updated list of [T_i, B, D]
        """
        N = self.num_experts
        outputs = features.copy()
        B = features[0].size(1)

        # Convert to (B, D, T) for convolution/interpolation
        feats_BDT = [feat.permute(1,2,0) for feat in features]  # list: (B,D,T)

        # Global summaries
        globals_ = [feat.mean(dim=2) for feat in feats_BDT]  # list of (B,D)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                pair_vec = torch.cat([globals_[i], globals_[j]], dim=-1)  # (B,2D)
                score = self.score_mlps[i*N + j](pair_vec).squeeze(-1)    # (B,)
                if score.mean().item() < self.threshold:
                    continue

                if self.is_slow[i] and not self.is_slow[j]:
                    # slow->fast
                    outputs[j] = self.slow2fast(feats_BDT[i], feats_BDT[j], score).permute(2,0,1)
                elif not self.is_slow[i] and self.is_slow[j]:
                    # fast->slow
                    key = f"{i}->{j}"
                    outputs[j] = self.fast2slow_ops[key](feats_BDT[i], feats_BDT[j], score).permute(2,0,1)

        return outputs


# === Expert Backbone and Readout ===
# -----------------------------------------------------------------------------
# Expert Readout
# -----------------------------------------------------------------------------
Tensor = torch.Tensor
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class TemporalExpert(nn.Module):
    """
    Single temporal expert.
    """
    def __init__(self, d_model: int, n_head: int, d_ffn: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            QuickGELU(),
            nn.Linear(d_ffn, d_model)
        )
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        # x: (Sequence_Length, Batch_Size, d_model)
        x = self.ln_1(x + self.attn(x, x, x)[0])
        x = self.ln_2(x + self.ffn(x))
        return x

class Combination(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4):
        super().__init__()
        self.d_model = d_model
        
        self.global_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=False)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, expert_seqs: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Args:
            expert_seqs (List[Tensor]): List of temporal outputs from N experts.
                                        Each tensor has shape [L_i, B, D].
        Returns:
            Tuple[Tensor, Tensor]:
            - fused_output (Tensor): Fused output vector with shape [B, D].
            - expert_weights (Tensor): Total attention paid to each expert, with shape [B, N], for analysis.
        """
        expert_lengths = [seq.size(0) for seq in expert_seqs]
        if not expert_seqs:
            raise ValueError("expert_seqs cannot be empty.")
        B = expert_seqs[0].size(1)
        num_experts = len(expert_seqs)

        kv_sequence = torch.cat(expert_seqs, dim=0)

        q = self.global_query.expand(1, B, self.d_model)

        pooled_output, attn_weights = self.attn(q, kv_sequence, kv_sequence)
        
        fused_output = self.ln(pooled_output.squeeze(0))

        attn_weights = attn_weights.squeeze(1) # -> [B, sum(L_i)]
        expert_weights_list = []
        current_idx = 0
        for length in expert_lengths:
            weights_slice = attn_weights[:, current_idx : current_idx + length]
            total_expert_weight = weights_slice.sum(dim=1)
            expert_weights_list.append(total_expert_weight)
            current_idx += length
        
        # [B, N]
        expert_weights = torch.stack(expert_weights_list, dim=1)

        return fused_output, expert_weights


# === Mixture and Aggregation ===
class MixtureOfTemporalExperts(nn.Module):
    def __init__(self, d_model: int, num_experts: int,
                 expert_n_head: int = 4, expert_d_ffn: int = 1024,
                 pooling_type: str = "attn",
                 gate_type: str = "attn",
                 gate_n_head: int = 4,
                 gate_queries: int = 1):   # gate_n_head is kept for signature compatibility (unused)
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        assert pooling_type in {"attn", "cls", "mean"}
        assert gate_type in {"attn", "mlp"}
        self.pooling_type = pooling_type
        self.gate_type = gate_type

        self.experts = nn.ModuleList([
            TemporalExpert(d_model, expert_n_head, expert_d_ffn)
            for _ in range(num_experts)
        ])

        self.readout_gate = Combination(
            d_model=d_model,
            nhead=gate_n_head,
        )

    def forward(self, expert_inputs: List[Tensor]):
        """
        expert_inputs: List[[L_i, B, D]] with length = num_experts
        return:
          fused_output: [B, D]
          gating_weights: [B, N]
        """
        assert len(expert_inputs) == self.num_experts, "Number of inputs must match the number of experts"

        expert_vecs = []
        for i, x in enumerate(expert_inputs):
            seq = self.experts[i](x)                     # [L_i,B,D]
            # print(f"Expert {i} output shape: {seq.shape}")
            expert_vecs.append(seq)
        fused_output, gating_weights = self.readout_gate(expert_vecs)
        expert_vecs_for_div = [seq.mean(dim=0) for seq in expert_vecs]
        E_bnd = torch.stack(expert_vecs_for_div, dim=0).permute(1, 0, 2).contiguous()

        return fused_output, gating_weights, E_bnd

# === VidPrism Head ===
# -----------------------------------------------------------------------------
# Top-Level VidPrism
# -----------------------------------------------------------------------------
class VidPrism(nn.Module):
    def __init__(self,
                #  clip_model, 
                 num_experts: int = 4,
                 sampling_rates: List[int] = [2, 4, 8, 16], # Sampling rates
                 num_classes: int = 101, # Classification task
                 d_model: int = None,
                 dtem_tau: float = 0.1,
                 loss_type: str = "listwise",
                 use_rank_loss: bool = True,
                 alpha_mix: float = 0.7,
                 score_temp: float = 0.2,): 
        super().__init__()
        
        assert len(sampling_rates) == num_experts, "The number of sampling rates must match the number of experts"
        self.sampling_rates = sampling_rates

        self.pooling_layers = nn.ModuleList([
            RgSTA(d_model=d_model, rate=rate, keep_k=1, tau=dtem_tau,
                             alpha_mix=alpha_mix,
                             use_rank_loss=use_rank_loss,
                             score_temp=score_temp,
                             loss_type=loss_type,)
            for rate in sampling_rates
        ])

        is_slow_flags = [rate >= 8 for rate in sampling_rates]

        self.dynamic_interaction = DBI(
            d_model=d_model,
            num_experts=num_experts,
            is_slow=is_slow_flags,
            sampling_rates=sampling_rates,
            threshold=0.3  # Tunable
        )

        self.mote = MixtureOfTemporalExperts(
            d_model=d_model,
            num_experts=num_experts,
            expert_n_head=4, # Attention heads inside each expert
            expert_d_ffn=d_model * 2 # FFN width inside each expert
        )

        # Output head
        self.head = nn.Linear(d_model, num_classes)
        # Projection head
        # self.projection_head = nn.Sequential(
        #     nn.Linear(d_model, d_model),
        #     nn.ReLU(),
        #     nn.Linear(d_model, d_model) # Output dimension matches text_embedding
        # )

    def feature_level_pooling(self, features: torch.Tensor, importance_prior: torch.Tensor = None):
        """
        Args:
            features (torch.Tensor): Raw feature sequence with shape (T, B, D)
        Returns:
            List[torch.Tensor]: Feature list processed by different pooling layers, each with shape (T_new, B, D)
        """
        features_for_pooling = features.permute(1, 2, 0) # (T, B, D) -> (B, D, T)
        if importance_prior is not None:
            importance_prior = importance_prior.detach()        
        expert_inputs = []
        extra_losses = []
        for pooler in self.pooling_layers:
            pooled_features, extra_loss = pooler(features_for_pooling)
            pooled_features = pooled_features.permute(2, 0, 1) # (B, D, T_new) -> (T_new, B, D)
            
            expert_inputs.append(pooled_features)
            extra_losses.append(extra_loss)
        if len(extra_losses) > 0:
            extra_loss = torch.stack(extra_losses).mean()
        else:
            extra_loss = features_for_pooling.new_tensor(0.0)            
        return expert_inputs, extra_loss
    
    def forward(self, video_frames: torch.Tensor, batch_size: int = None, num_frames: int = None, importance_prior: torch.Tensor = None):
        """
        Args:
            video_frames (torch.Tensor): (B, T, C, H, W)
        """
        features = video_frames.view(batch_size, num_frames, -1) # (B, T, D)
        features = features.permute(1, 0, 2) # (T, B, D)

        expert_inputs, extra_loss = self.feature_level_pooling(features, importance_prior)
        expert_inputs = self.dynamic_interaction(expert_inputs)
        fused_output, gating_weights, E_bnd = self.mote(expert_inputs)
        div_loss = diversity_loss_experts(E_bnd)
        logits = self.head(fused_output)

        return logits, gating_weights, div_loss, extra_loss


class VideoCLIP(nn.Module):
    def __init__(self, clip_model, n_seg) :
        super(VideoCLIP, self).__init__()
        self.visual = clip_model.visual
        self.n_seg = n_seg
        self.logit_scale = clip_model.logit_scale

    def forward(self, image):
        # CLIP encode images
        image_emb = self.encode_image(image) # [BS, T, C]
        return image_emb

    def encode_image(self, image):
        bt = image.size(0) # [BS*T, C, H, W]
        b = bt // self.n_seg
        image_emb = self.visual(image) # [BS*T, C]
        image_emb = image_emb.view(b, self.n_seg, -1) # [BS, T, C]
        return image_emb
    

class VideoMAEFeatureExtractor(nn.Module):
    def __init__(self, videomae_model, num_segments):
        super().__init__()
        self.videomae = videomae_model
        self.num_segments = num_segments
        proj = self.videomae.model.patch_embed.proj
        self.t_patch, self.h_patch, self.w_patch = proj.kernel_size

    def forward(self, video_tensor):
        b_times_t, c, h, w = video_tensor.shape
        t = self.num_segments
        b = b_times_t // t
        frames = video_tensor.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, H, W]
        patch_tokens = self.videomae(frames)  # Output can be [B, N, D] or [B, T_patches, D]

        # Infer the output shape automatically
        if patch_tokens.ndim == 3:
            B, N, D = patch_tokens.shape
        else:
            raise RuntimeError(f"Unexpected VideoMAEv2 output shape: {patch_tokens.shape}")

        # Assume N == T_patches * H_patches * W_patches
        T_patches = frames.shape[2] // self.t_patch
        H_patches = frames.shape[3] // self.h_patch
        W_patches = frames.shape[4] // self.w_patch
        expected_tokens = T_patches * H_patches * W_patches

        if N == expected_tokens:
            # Standard ViT output: preserve the true temporal dimension
            patch_tokens = patch_tokens.view(B, T_patches, H_patches * W_patches, D)
            time_tokens = patch_tokens.mean(2)
        elif N == T_patches:
            # The model already averaged spatially, so only temporal tokens remain
            time_tokens = patch_tokens
        else:
            # fail-safe fallback
            print(f"[WARN] Unrecognized patch layout: N={N}, expected {expected_tokens} or {T_patches}")
            time_tokens = patch_tokens.mean(1, keepdim=True)

        return time_tokens  # [B, T, D]

    def encode_image(self, video_tensor):
        return self.forward(video_tensor)
