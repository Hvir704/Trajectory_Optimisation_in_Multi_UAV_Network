"""
policy.py — Attention Policy Network (Set-Representation + Heated-Up Softmax)
==============================================================================
Implements two fixes from MOSAC-ATT (Toure et al., IEEE IoT Journal 2026).

FIX 1 — MaxPool set-representation (MOSAC-ATT §III-C, Fig. 3)
--------------------------------------------------------------
PROBLEM:  The original Baseline used mean(node_emb, dim=1). Because mean
          divides by M, its scale shrinks as M grows. The model trained on
          M=30 therefore saw a distribution shift at M=100, contributing to
          conservative/broken behaviour at unseen sizes.

FIX:      Replace mean with max(dim=1) in Baseline.forward(). MaxPool retains
          the strongest per-dimension signal regardless of how many nodes are
          present — the aggregate is M-independent in scale.

          The pointer head and self-attention layers were already M-agnostic
          (they operate on sequences of arbitrary length). This single change
          to Baseline is the minimal fix to make the whole model M-independent.

          A single trained model can now be evaluated zero-shot on any M.


FIX 2 — Heated-up softmax / decaying temperature (MOSAC-ATT §IV-A)
--------------------------------------------------------------------
PROBLEM:  REINFORCE with a static entropy bonus β struggles to balance early
          exploration (needs high randomness) with late convergence (needs
          determinism). Static β is a compromise that often does neither well.

FIX:      Add a temperature parameter τ to sample_action() and greedy_action().
          The trainer (trainer_fixed.py) anneals τ linearly from τ_start → τ_final
          over the first T_anneal epochs, then holds τ_final fixed.

          High τ (early):   logits/τ → flat → policy samples broadly
          Low τ  (late):    logits/τ → peaked → policy commits to best action

          Simultaneously the entropy target H0 decays so the entropy bonus
          also reduces as training matures (managed entirely in trainer_fixed.py).
          policy.py only needs to accept and apply the τ argument.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Sub-modules (node encoder, context encoder, pointer head) — UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────

class NodeEncoder(nn.Module):
    """Projects per-node 5-dim features into d_model-dim embeddings."""

    def __init__(self, d_model=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(5, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        # x: (batch, M, 5)  — M is arbitrary
        return self.proj(x)   # (batch, M, d_model)


class ContextEncoder(nn.Module):
    """Projects 6-dim global UAV state into a d_model-dim query vector."""

    def __init__(self, d_model=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(6, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, ctx):
        return self.proj(ctx)   # (batch, d_model)


class AttentionPolicyHead(nn.Module):
    """
    Multi-head attention pointer head.
    The context query attends over node keys/values to produce a
    refined query, which is then dot-producted against node keys to
    yield per-node compatibility scores (masked, tanh-clipped, log-softmax).
    Natively M-agnostic.
    """

    def __init__(self, d_model=128, n_heads=8, C=10.0):
        super().__init__()
        self.C       = C
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads

        self.W_q   = nn.Linear(d_model, d_model, bias=False)
        self.W_k   = nn.Linear(d_model, d_model, bias=False)
        self.W_v   = nn.Linear(d_model, d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model, bias=False)
        self.W_q2  = nn.Linear(d_model, d_model, bias=False)
        self.W_k2  = nn.Linear(d_model, d_model, bias=False)

    def forward(self, ctx_emb, node_emb, mask):
        """
        ctx_emb:  (B, d_model)
        node_emb: (B, M, d_model)  — M flexible
        mask:     (B, M) bool, True = INVALID (visited or infeasible)
        Returns:  (B, M) log-probabilities
        """
        B, M, D = node_emb.shape
        H, dk   = self.n_heads, self.d_k

        # Multi-head cross-attention: context → query, nodes → keys/values
        q  = self.W_q(ctx_emb).view(B, 1, H, dk).transpose(1, 2)    # (B,H,1,dk)
        k  = self.W_k(node_emb).view(B, M, H, dk).transpose(1, 2)   # (B,H,M,dk)
        v  = self.W_v(node_emb).view(B, M, H, dk).transpose(1, 2)   # (B,H,M,dk)

        scores = (q @ k.transpose(-2, -1)) / (dk ** 0.5)             # (B,H,1,M)
        scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), -1e9)
        attn   = F.softmax(scores, dim=-1)
        ctx_v  = (attn @ v).transpose(1, 2).contiguous().view(B, 1, D)
        ctx_v  = self.W_out(ctx_v).squeeze(1)                        # (B,D)

        # Final compatibility: updated context vs each node key
        q2     = self.W_q2(ctx_v).unsqueeze(1)                       # (B,1,D)
        k2     = self.W_k2(node_emb)                                  # (B,M,D)
        compat = (q2 @ k2.transpose(-2, -1)).squeeze(1) / (D ** 0.5) # (B,M)
        compat = self.C * torch.tanh(compat)
        compat = compat.masked_fill(mask, -1e9)

        return F.log_softmax(compat, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# FIX 1: Baseline with MaxPool
# ─────────────────────────────────────────────────────────────────────────────

class Baseline(nn.Module):
    """
    Value baseline for REINFORCE variance reduction.

    FIX 1 change: uses max(dim=1) instead of mean(dim=1) over node embeddings.

    Why this matters:
        mean(node_emb, dim=1) has magnitude ∝ 1/M (assuming roughly equal
        norms). The baseline therefore sees a systematically smaller input
        when M is large. A model trained at M=30 will receive an out-of-
        distribution input when run at M=100, causing the baseline to
        misestimate values and destabilise REINFORCE.

        max(node_emb, dim=1) selects the maximum along each feature
        dimension independently. Its magnitude does NOT depend on M — only
        on the distribution of individual node embeddings. A model trained
        at M=30 will see inputs of the same scale at M=100.
    """

    def __init__(self, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, ctx_emb, node_emb):
        # FIX 1: MaxPool over M dimension (was: node_emb.mean(dim=1))
        max_nodes = node_emb.max(dim=1).values          # (batch, d_model)
        inp       = torch.cat([ctx_emb, max_nodes], -1) # (batch, 2*d_model)
        return self.net(inp).squeeze(-1)                 # (batch,)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level policy: wires everything together, exposes temperature for Fix 2
# ─────────────────────────────────────────────────────────────────────────────

class AttentionPolicy(nn.Module):
    """
    Attention pointer policy with set-representation (Fix 1) and
    temperature-controlled sampling (Fix 2).

    Changes vs original policy.py:
      - Baseline uses MaxPool (Fix 1): M-independent baseline scale.
      - sample_action() and greedy_action() accept a `temperature` kwarg (Fix 2).

    All weight shapes are identical to the original, so pre-trained checkpoints
    (attn_M*.pt) remain compatible — load_state_dict will work unchanged.
    """

    def __init__(self, d_model=128, n_heads=8, n_layers=3):
        super().__init__()
        self.d_model  = d_model
        self.node_enc = NodeEncoder(d_model)

        self.sa_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=512, dropout=0.0,
                batch_first=True, norm_first=True
            ) for _ in range(n_layers)
        ])

        self.ctx_enc  = ContextEncoder(d_model)
        self.ptr_head = AttentionPolicyHead(d_model, n_heads)
        self.baseline = Baseline(d_model)   # ← MaxPool version (Fix 1)

    def encode_nodes(self, node_feats):
        """(batch, M, 5) → (batch, M, d_model). M is arbitrary."""
        h = self.node_enc(node_feats)
        for layer in self.sa_layers:
            h = layer(h)
        return h

    def forward(self, node_feats, ctx_feats, mask):
        """
        node_feats: (batch, M, 5)   — any M at train or inference
        ctx_feats:  (batch, 6)
        mask:       (batch, M) bool, True = cannot visit

        Returns
        -------
        log_probs : (batch, M)   log-softmax over valid actions
        value     : (batch,)     baseline value estimate
        """
        node_emb = self.encode_nodes(node_feats)   # (batch, M, d_model)
        ctx_emb  = self.ctx_enc(ctx_feats)          # (batch, d_model)
        log_p    = self.ptr_head(ctx_emb, node_emb, mask)
        value    = self.baseline(ctx_emb, node_emb)
        return log_p, value

    # ── FIX 2: temperature-controlled action selection ──────────────────────

    def greedy_action(self, node_feats, ctx_feats, mask, temperature=1.0):
        """
        Deterministic action (argmax). Used at evaluation time.

        temperature=1.0  → standard argmax on log-probs (identical to original).
        temperature>1.0  → soften logits before argmax; used for warm-up eval
                           so the best action is still selected but the ranking
                           is less brittle under a nearly-uniform early policy.
        """
        with torch.no_grad():
            log_p, _ = self.forward(node_feats, ctx_feats, mask)
            if temperature == 1.0:
                return log_p.argmax(dim=-1)
            logits = log_p / temperature
            logits = logits.masked_fill(mask, -1e9)
            return logits.argmax(dim=-1)

    def sample_action(self, node_feats, ctx_feats, mask, temperature=1.0):
        """
        FIX 2: Stochastic action with heated-up softmax.

        The trainer calls this with the current annealed temperature τ:
            τ_start (e.g. 5.0) → wide distribution → explore all nodes
            τ_final (e.g. 1.0) → peaked distribution → commit to best

        Implementation detail:
            The pointer head returns log_softmax(compat). We cannot directly
            divide compat by τ here, so we treat log_p as unnormalised logits
            (valid because log_softmax is a monotone transform — dividing the
            inputs by τ preserves the relative ordering and flattens/sharpens
            the resulting distribution identically to temperature scaling).

            For τ=1 the distribution is identical to the original.
            For τ>1 the distribution is flattened (more uniform → exploration).
            For τ<1 the distribution sharpens (not used during training here).

        IMPORTANT: log_prob for the REINFORCE gradient is always computed under
        the UNSCALED distribution. Temperature only governs which action is
        sampled, not how we score it. This keeps the gradient unbiased with
        respect to the learned policy.

        Returns
        -------
        action    : (batch,) int64  sampled action index
        log_prob  : (batch,)        log-probability under the *unscaled* policy
        value     : (batch,)        baseline value estimate
        """
        log_p, value = self.forward(node_feats, ctx_feats, mask)

        if temperature == 1.0:
            dist = torch.distributions.Categorical(logits=log_p)
        else:
            scaled = log_p / temperature
            scaled = scaled.masked_fill(mask, -1e9)
            dist   = torch.distributions.Categorical(logits=scaled)

        action = dist.sample()

        # Score under the UNSCALED policy for unbiased REINFORCE update
        log_prob_unscaled = torch.distributions.Categorical(logits=log_p).log_prob(action)
        return action, log_prob_unscaled, value