import torch
import torch.nn as nn
from torch.nn import functional as F
import triton
import triton.language as tl


# ==========================================
# 1. TRITON KERNELS (TRAINING & BACKWARD)
# ==========================================

@triton.jit
def wkv_forward_training_kernel(
        k_ptr, v_ptr, w_ptr, u_ptr,
        out_ptr,
        state_num_ptr, state_den_ptr, state_max_ptr,  # Saved states for backward
        B, T, C,
        stride_b, stride_t, stride_c
):
    # Parallelize over Batch (0) and Channel (1)
    batch_idx = tl.program_id(0)
    block_c_idx = tl.program_id(1)

    # We process BLOCK_C elements per thread block
    BLOCK_C: tl.constexpr = 64
    c_offsets = block_c_idx * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = c_offsets < C

    # Pointers
    k_base = k_ptr + batch_idx * stride_b + c_offsets * stride_c
    v_base = v_ptr + batch_idx * stride_b + c_offsets * stride_c
    out_base = out_ptr + batch_idx * stride_b + c_offsets * stride_c

    # State pointers (B, T, C) - We save state for every timestep for backward
    s_num_base = state_num_ptr + batch_idx * (T * C) + c_offsets
    s_den_base = state_den_ptr + batch_idx * (T * C) + c_offsets
    s_max_base = state_max_ptr + batch_idx * (T * C) + c_offsets

    # Load Parameters w and u
    w = tl.load(w_ptr + c_offsets, mask=mask_c, other=0.0)
    u = tl.load(u_ptr + c_offsets, mask=mask_c, other=0.0)

    # Initialize State
    num = tl.zeros([BLOCK_C], dtype=tl.float32)
    den = tl.zeros([BLOCK_C], dtype=tl.float32)
    max_state = tl.full([BLOCK_C], -1e38, dtype=tl.float32)

    for t in range(T):
        t_offset = t * stride_t

        # Load Current Input
        k = tl.load(k_base + t_offset, mask=mask_c, other=0.0)
        v = tl.load(v_base + t_offset, mask=mask_c, other=0.0)

        # 1. Calculate Output for this step
        # y_t = (e^u * state + e^k * v) / ...
        k_u = k + u
        max_for_out = tl.maximum(max_state, k_u)
        e1 = tl.exp(max_state - max_for_out)
        e2 = tl.exp(k_u - max_for_out)

        num_out = e1 * num + e2 * v
        den_out = e1 * den + e2
        y = num_out / den_out

        tl.store(out_base + t_offset, y, mask=mask_c)

        # 2. Update State for next step
        # state = state * e^w + k * v
        max_w = max_state + w
        max_new = tl.maximum(max_w, k)
        e1_s = tl.exp(max_w - max_new)
        e2_s = tl.exp(k - max_new)

        num = e1_s * num + e2_s * v
        den = e1_s * den + e2_s
        max_state = max_new

        # 3. Save State (Required for Backward)
        # Stride for state is simply C (contiguous in T*C block per batch)
        state_offset = t * C
        tl.store(s_num_base + state_offset, num, mask=mask_c)
        tl.store(s_den_base + state_offset, den, mask=mask_c)
        tl.store(s_max_base + state_offset, max_state, mask=mask_c)


@triton.jit
def wkv_backward_kernel(
        w_ptr, u_ptr, k_ptr, v_ptr,
        gy_ptr,
        state_num_ptr, state_den_ptr, state_max_ptr,
        gw_ptr, gu_ptr, gk_ptr, gv_ptr,
        B, T, C,
        stride_b, stride_t, stride_c
):
    batch_idx = tl.program_id(0)
    block_c_idx = tl.program_id(1)
    BLOCK_C: tl.constexpr = 64

    c_offsets = block_c_idx * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = c_offsets < C

    # Pointers
    off_b = batch_idx * stride_b + c_offsets * stride_c
    k_base = k_ptr + off_b
    v_base = v_ptr + off_b
    gy_base = gy_ptr + off_b

    gw_base = gw_ptr + off_b
    gu_base = gu_ptr + off_b
    gk_base = gk_ptr + off_b
    gv_base = gv_ptr + off_b

    # State Pointers (Saved from Forward)
    # Note: We need state[t-1] to compute grad at t.
    s_base = batch_idx * (T * C) + c_offsets
    s_num_ptr_base = state_num_ptr + s_base
    s_den_ptr_base = state_den_ptr + s_base
    s_max_ptr_base = state_max_ptr + s_base

    # Load Parameters
    w = tl.load(w_ptr + c_offsets, mask=mask_c, other=0.0)
    u = tl.load(u_ptr + c_offsets, mask=mask_c, other=0.0)

    # Gradients Accumulators for w and u (local to this thread block)
    gw_acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    gu_acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    # Gradient flowing back from future (initially 0)
    g_num = tl.zeros([BLOCK_C], dtype=tl.float32)
    g_den = tl.zeros([BLOCK_C], dtype=tl.float32)

    # Iterate Backwards
    for t in range(T - 1, -1, -1):
        t_offset = t * stride_t
        prev_state_idx = (t - 1) * C

        # Load inputs and gradient at t
        k = tl.load(k_base + t_offset, mask=mask_c, other=0.0)
        v = tl.load(v_base + t_offset, mask=mask_c, other=0.0)
        gy = tl.load(gy_base + t_offset, mask=mask_c, other=0.0)

        # Load State at t-1 (Input to step t)
        if t > 0:
            num_prev = tl.load(s_num_ptr_base + prev_state_idx, mask=mask_c, other=0.0)
            den_prev = tl.load(s_den_ptr_base + prev_state_idx, mask=mask_c, other=0.0)
            max_prev = tl.load(s_max_ptr_base + prev_state_idx, mask=mask_c, other=-1e38)
        else:
            num_prev = tl.zeros([BLOCK_C], dtype=tl.float32)
            den_prev = tl.zeros([BLOCK_C], dtype=tl.float32)
            max_prev = tl.full([BLOCK_C], -1e38, dtype=tl.float32)

        # --------------------------------------------
        # 1. Recompute Forward Calculations (Output Part)
        # --------------------------------------------
        # We need e1, e2, num_out, den_out to compute derivatives for Output
        k_u = k + u
        max_for_out = tl.maximum(max_prev, k_u)
        e1 = tl.exp(max_prev - max_for_out)
        e2 = tl.exp(k_u - max_for_out)

        num_out = e1 * num_prev + e2 * v
        den_out = e1 * den_prev + e2

        inv_den_out = 1.0 / (den_out + 1e-9)

        # --------------------------------------------
        # 2. Gradients for Output Part
        # --------------------------------------------
        # dLoss/d(num_out) = gy * (1/den)
        # dLoss/d(den_out) = gy * (-num/den^2)
        g_num_out = gy * inv_den_out
        g_den_out = gy * (-num_out * inv_den_out * inv_den_out)

        # Gradients flowing to e1 and e2 from Output equation
        g_e1 = g_num_out * num_prev + g_den_out * den_prev
        g_e2 = g_num_out * v + g_den_out

        # Accumulate gu (u affects e2)
        # d(e2)/du = e2
        gu_acc += g_e2 * e2

        # Initial gradients for k, v from Output equation
        gk_t = g_e2 * e2
        gv_t = g_num_out * e2

        # --------------------------------------------
        # 3. Recompute Forward Calculations (State Part)
        # --------------------------------------------
        # THIS WAS MISSING: We must re-calculate the transition factors
        # max_w = max_prev + w
        # max_new = max(max_w, k)
        # e1_s = exp(max_w - max_new)
        # e2_s = exp(k - max_new)

        max_w = max_prev + w
        max_new_state = tl.maximum(max_w, k)
        e1_s = tl.exp(max_w - max_new_state)
        e2_s = tl.exp(k - max_new_state)

        # --------------------------------------------
        # 4. Backpropagate through State Recurrence
        # --------------------------------------------
        # Gradients coming FROM future steps (g_num, g_den) flowing into current step
        g_e1_s = g_num * num_prev + g_den * den_prev
        g_e2_s = g_num * v + g_den

        # Accumulate gw (w affects e1_s via max_w)
        # d(e1_s)/dw = e1_s * 1 (approx, ignoring max grad discontinuity)
        gw_acc += g_e1_s * e1_s

        # Accumulate into k, v
        gk_t += g_e2_s * e2_s
        gv_t += g_num * e2_s

        # Gradients flowing to prev state (num_prev, den_prev)
        # Used for next iteration (t-1)
        g_num = g_num_out * e1 + g_num * e1_s
        g_den = g_den_out * e1 + g_den * e1_s

        tl.store(gk_base + t_offset, gk_t, mask=mask_c)
        tl.store(gv_base + t_offset, gv_t, mask=mask_c)

    # Store accumulated gradients for w and u
    tl.store(gw_base, gw_acc, mask=mask_c)
    tl.store(gu_base, gu_acc, mask=mask_c)


# ==========================================
# 2. AUTOGRAD FUNCTION
# ==========================================

class WKVFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, u, k, v):
        # Ensure contiguous for Triton pointers
        w = w.contiguous()
        u = u.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        B, T, C = k.shape

        # Alloc Output
        y = torch.empty_like(k)

        # Alloc State History (Required for Backward)
        # We need these to recompute gradients correctly
        state_num = torch.empty((B, T, C), device=k.device, dtype=torch.float32)
        state_den = torch.empty((B, T, C), device=k.device, dtype=torch.float32)
        state_max = torch.empty((B, T, C), device=k.device, dtype=torch.float32)

        BLOCK_C = 64
        grid = (B, triton.cdiv(C, BLOCK_C))

        wkv_forward_training_kernel[grid](
            k, v, w, u, y,
            state_num, state_den, state_max,
            B, T, C,
            k.stride(0), k.stride(1), k.stride(2)
        )

        ctx.save_for_backward(w, u, k, v, state_num, state_den, state_max)
        return y

    @staticmethod
    def backward(ctx, gy):
        w, u, k, v, state_num, state_den, state_max = ctx.saved_tensors
        gy = gy.contiguous()

        B, T, C = k.shape

        gw = torch.zeros_like(k)  # Shape B, T, C (conceptually) -> but we store B,C result at offset 0
        gu = torch.zeros_like(k)
        gk = torch.empty_like(k)
        gv = torch.empty_like(k)

        BLOCK_C = 64
        grid = (B, triton.cdiv(C, BLOCK_C))

        wkv_backward_kernel[grid](
            w, u, k, v,
            gy,
            state_num, state_den, state_max,
            gw, gu, gk, gv,
            B, T, C,
            k.stride(0), k.stride(1), k.stride(2)
        )

        # gw and gu are accumulated over T inside the kernel,
        # but we ran B programs. Need to sum over B.
        # The kernel stored the B-wise accumulation in the first timestep slot of the output tensor
        # (or strictly speaking, we mapped `gw_base` to the batch offset).
        # But `gw` tensor has shape B, T, C.
        # The kernel wrote result to `gw_ptr + off_b`, which is effectively gw[:, 0, :].
        # However, to keep it clean, we sum over B:

        gw_sum = gw[:, 0, :].sum(0)
        gu_sum = gu[:, 0, :].sum(0)

        return gw_sum, gu_sum, gk, gv


def wkv_forward_runner(k, v, w, u):
    """
    Drop-in replacement for your runner.
    Uses the Autograd Function which handles Triton Kernel launch.
    """
    return WKVFunc.apply(w, u, k, v)


# ==========================================
# 3. MODULES (Unchanged except logic separation)
# ==========================================

class BiRWKVLayer(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd
        self.rkv_proj = nn.Linear(n_embd, n_embd * 6, bias=False)
        self.output_proj = nn.Linear(n_embd * 2, n_embd, bias=False)

        self.time_decay = nn.Parameter(torch.empty(n_embd))
        self.time_first = nn.Parameter(torch.empty(n_embd))
        self.time_decay_rev = nn.Parameter(torch.empty(n_embd))
        self.time_first_rev = nn.Parameter(torch.empty(n_embd))

        self._init_params()

    def _init_params(self):
        with torch.no_grad():
            self.time_decay.uniform_(-6.0, -5.0)
            self.time_first.uniform_(-3.0, 0.0)
            self.time_decay_rev.uniform_(-6.0, -5.0)
            self.time_first_rev.uniform_(-3.0, 0.0)

    def forward(self, x):
        B, T, C = x.shape
        rkv = self.rkv_proj(x).view(B, T, 2, 3, C)

        # Fwd
        r_f = rkv[:, :, 0, 0]
        k_f = rkv[:, :, 0, 1]
        v_f = rkv[:, :, 0, 2]
        w_f = -torch.exp(self.time_decay)
        u_f = self.time_first

        out_f = wkv_forward_runner(k_f, v_f, w_f, u_f)
        out_f = torch.sigmoid(r_f) * out_f

        # Bwd
        r_b = rkv[:, :, 1, 0]
        k_b = rkv[:, :, 1, 1]
        v_b = rkv[:, :, 1, 2]
        w_b = -torch.exp(self.time_decay_rev)
        u_b = self.time_first_rev

        k_b_rev = k_b.flip(dims=[1])
        v_b_rev = v_b.flip(dims=[1])

        # The runner handles .contiguous() internally via the Function,
        # but flip creates a stride step of -1 which Triton dislikes.
        # We ensure contiguous here to be safe, though WKVFunc also checks it.
        out_b_rev = wkv_forward_runner(k_b_rev.contiguous(), v_b_rev.contiguous(), w_b, u_b)

        out_b = out_b_rev.flip(dims=[1])
        out_b = torch.sigmoid(r_b) * out_b

        out_cat = torch.cat([out_f, out_b], dim=-1)
        return self.output_proj(out_cat)


class BiRWKVBlock(nn.Module):
    def __init__(self, n_embd, mlp_ratio=4):
        super().__init__()
        self.attn = BiRWKVLayer(n_embd)
        self.ffn_proj = nn.Linear(n_embd, n_embd * mlp_ratio, bias=False)
        self.ffn_out = nn.Linear(n_embd * mlp_ratio, n_embd, bias=False)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn_out(F.gelu(self.ffn_proj(x)))
        return x


class BiRWKV(nn.Module):
    def __init__(self, input_size, num_layers=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            BiRWKVBlock(input_size) for _ in range(num_layers)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# ==========================================
# 4. GRADIENT CHECK
# ==========================================
if __name__ == "__main__":
    B, T, C = 2, 32, 64
    device = "cuda"

    # Create Model
    model = BiRWKV(input_size=C, num_layers=1).to(device)

    # Input
    x = torch.randn(B, T, C, device=device, requires_grad=True)

    # Forward
    y = model(x)
    loss = y.sum()

    # Backward
    print(f"Starting Backward...")
    loss.backward()
    print("Backward complete.")

    # Check if gradients exist
    print(f"x.grad norm: {x.grad.norm().item()}")
    print(f"time_decay grad norm: {model.blocks[0].attn.time_decay.grad.norm().item()}")
