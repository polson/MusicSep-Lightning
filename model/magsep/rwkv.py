import torch
import torch.nn as nn
from torch.nn import functional as F
import triton
import triton.language as tl


# ==========================================
# 1. TRITON KERNELS
# ==========================================

@triton.jit
def wkv_forward_training_kernel(
        k_ptr, v_ptr, w_ptr, u_ptr,
        out_ptr,
        state_num_ptr, state_den_ptr, state_max_ptr,
        B, T, C,
        stride_b, stride_t, stride_c,
        REVERSE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    block_c_idx = tl.program_id(1)
    BLOCK_C: tl.constexpr = 64

    c_offsets = block_c_idx * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = c_offsets < C

    # Base pointers for this batch/channel block
    # Note: We do NOT include time offset here yet
    off_bc = batch_idx * stride_b + c_offsets * stride_c
    k_base = k_ptr + off_bc
    v_base = v_ptr + off_bc
    out_base = out_ptr + off_bc

    # State bases (Contiguous in T*C per batch)
    s_off_bc = batch_idx * (T * C) + c_offsets
    s_num_base = state_num_ptr + s_off_bc
    s_den_base = state_den_ptr + s_off_bc
    s_max_base = state_max_ptr + s_off_bc

    # Load Parameters
    w = tl.load(w_ptr + c_offsets, mask=mask_c, other=0.0)
    u = tl.load(u_ptr + c_offsets, mask=mask_c, other=0.0)

    # Initialize State
    num = tl.zeros([BLOCK_C], dtype=tl.float32)
    den = tl.zeros([BLOCK_C], dtype=tl.float32)
    max_state = tl.full([BLOCK_C], -1e38, dtype=tl.float32)

    # Iterate over time
    for step in range(T):
        # Calculate actual time index 't' based on direction
        if REVERSE:
            t = T - 1 - step
        else:
            t = step

        t_offset = t * stride_t
        state_idx = t * C

        # Load Inputs
        k = tl.load(k_base + t_offset, mask=mask_c, other=0.0)
        v = tl.load(v_base + t_offset, mask=mask_c, other=0.0)

        # 1. Output Calculation (Input is 'state', 'k', 'v')
        k_u = k + u
        max_for_out = tl.maximum(max_state, k_u)
        e1 = tl.exp(max_state - max_for_out)
        e2 = tl.exp(k_u - max_for_out)

        num_out = e1 * num + e2 * v
        den_out = e1 * den + e2
        y = num_out / den_out

        tl.store(out_base + t_offset, y, mask=mask_c)

        # 2. State Update (Prepare 'state' for next step)
        max_w = max_state + w
        max_new = tl.maximum(max_w, k)
        e1_s = tl.exp(max_w - max_new)
        e2_s = tl.exp(k - max_new)

        num = e1_s * num + e2_s * v
        den = e1_s * den + e2_s
        max_state = max_new

        # 3. Save State (needed for backward)
        # We save the state RESULTING from step t
        tl.store(s_num_base + state_idx, num, mask=mask_c)
        tl.store(s_den_base + state_idx, den, mask=mask_c)
        tl.store(s_max_base + state_idx, max_state, mask=mask_c)


@triton.jit
def wkv_backward_kernel(
        w_ptr, u_ptr, k_ptr, v_ptr,
        gy_ptr,
        state_num_ptr, state_den_ptr, state_max_ptr,
        gw_ptr, gu_ptr, gk_ptr, gv_ptr,
        B, T, C,
        stride_b, stride_t, stride_c,
        REVERSE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    block_c_idx = tl.program_id(1)
    BLOCK_C: tl.constexpr = 64

    c_offsets = block_c_idx * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = c_offsets < C

    off_bc = batch_idx * stride_b + c_offsets * stride_c

    # State Base
    s_off_bc = batch_idx * (T * C) + c_offsets
    s_num_base = state_num_ptr + s_off_bc
    s_den_base = state_den_ptr + s_off_bc
    s_max_base = state_max_ptr + s_off_bc

    # Params
    w = tl.load(w_ptr + c_offsets, mask=mask_c, other=0.0)
    u = tl.load(u_ptr + c_offsets, mask=mask_c, other=0.0)

    # Accumulators
    gw_acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    gu_acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    # Gradients flowing back from "future" (recursion)
    g_num = tl.zeros([BLOCK_C], dtype=tl.float32)
    g_den = tl.zeros([BLOCK_C], dtype=tl.float32)

    # Iterate backwards through the sequence processing
    for step in range(T):
        # 1. Determine which timestep 't' we are calculating gradient for.
        # If Forward was 0->T, Backward is (T-1)->0
        # If Forward was (T-1)->0, Backward is 0->T
        if REVERSE:
            t = step
            prev_t = t + 1  # The "previous" state in the recursion was at t+1
            is_first_step = (t == T - 1)
        else:
            t = T - 1 - step
            prev_t = t - 1  # The "previous" state in the recursion was at t-1
            is_first_step = (t == 0)

        t_offset = t * stride_t

        # Load Inputs at t
        k = tl.load(k_ptr + off_bc + t_offset, mask=mask_c, other=0.0)
        v = tl.load(v_ptr + off_bc + t_offset, mask=mask_c, other=0.0)
        gy = tl.load(gy_ptr + off_bc + t_offset, mask=mask_c, other=0.0)

        # Load State (The input state to the forward calc at step t)
        if not is_first_step:
            prev_state_idx = prev_t * C
            num_prev = tl.load(s_num_base + prev_state_idx, mask=mask_c, other=0.0)
            den_prev = tl.load(s_den_base + prev_state_idx, mask=mask_c, other=0.0)
            max_prev = tl.load(s_max_base + prev_state_idx, mask=mask_c, other=-1e38)
        else:
            num_prev = tl.zeros([BLOCK_C], dtype=tl.float32)
            den_prev = tl.zeros([BLOCK_C], dtype=tl.float32)
            max_prev = tl.full([BLOCK_C], -1e38, dtype=tl.float32)

        # --- Recompute Forward Math ---

        # Output part
        k_u = k + u
        max_for_out = tl.maximum(max_prev, k_u)
        e1 = tl.exp(max_prev - max_for_out)
        e2 = tl.exp(k_u - max_for_out)

        num_out = e1 * num_prev + e2 * v
        den_out = e1 * den_prev + e2
        inv_den_out = 1.0 / (den_out + 1e-9)

        # State part (Transition)
        max_w = max_prev + w
        max_new_state = tl.maximum(max_w, k)
        e1_s = tl.exp(max_w - max_new_state)
        e2_s = tl.exp(k - max_new_state)

        # --- Gradient Math ---

        # 1. Gradients from Output (y)
        g_num_out = gy * inv_den_out
        g_den_out = gy * (-num_out * inv_den_out * inv_den_out)

        g_e1 = g_num_out * num_prev + g_den_out * den_prev
        g_e2 = g_num_out * v + g_den_out

        # Accumulate u
        gu_acc += g_e2 * e2

        gk_t = g_e2 * e2
        gv_t = g_num_out * e2

        # 2. Gradients from State Transition (Recursion)
        # g_num, g_den are gradients flowing from the "next" step
        g_e1_s = g_num * num_prev + g_den * den_prev
        g_e2_s = g_num * v + g_den

        # Accumulate w (via max_w -> e1_s)
        gw_acc += g_e1_s * e1_s

        # Accumulate k, v
        gk_t += g_e2_s * e2_s
        gv_t += g_num * e2_s

        # 3. Propagate to Prev State (num_prev, den_prev)
        # These become g_num/g_den for the next iteration of loop
        g_num = g_num_out * e1 + g_num * e1_s
        g_den = g_den_out * e1 + g_den * e1_s

        # Store Gradients
        tl.store(gk_ptr + off_bc + t_offset, gk_t, mask=mask_c)
        tl.store(gv_ptr + off_bc + t_offset, gv_t, mask=mask_c)

    # Store accumulated param grads (broadcasted to batch via stride logic in caller)
    # The caller expects shape [B, T, C] but we only write 1 value per batch/channel
    # We write to the first timestep slot to simplify, caller sums later
    tl.store(gw_ptr + off_bc, gw_acc, mask=mask_c)
    tl.store(gu_ptr + off_bc, gu_acc, mask=mask_c)


# ==========================================
# 2. AUTOGRAD FUNCTION
# ==========================================

class WKVFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, u, k, v, reverse=False):
        # No flip needed!
        w = w.contiguous()
        u = u.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        ctx.reverse = reverse  # Save flag

        B, T, C = k.shape
        y = torch.empty_like(k)

        # Save states for backward
        state_num = torch.empty((B, T, C), device=k.device, dtype=torch.float32)
        state_den = torch.empty((B, T, C), device=k.device, dtype=torch.float32)
        state_max = torch.empty((B, T, C), device=k.device, dtype=torch.float32)

        BLOCK_C = 64
        grid = (B, triton.cdiv(C, BLOCK_C))

        wkv_forward_training_kernel[grid](
            k, v, w, u, y,
            state_num, state_den, state_max,
            B, T, C,
            k.stride(0), k.stride(1), k.stride(2),
            REVERSE=reverse
        )

        ctx.save_for_backward(w, u, k, v, state_num, state_den, state_max)
        return y

    @staticmethod
    def backward(ctx, gy):
        w, u, k, v, state_num, state_den, state_max = ctx.saved_tensors
        reverse = ctx.reverse

        gy = gy.contiguous()
        B, T, C = k.shape

        gw = torch.zeros_like(k)
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
            k.stride(0), k.stride(1), k.stride(2),
            REVERSE=reverse
        )

        # Sum accumulated gradients over batch
        gw_sum = gw[:, 0, :].sum(0)
        gu_sum = gu[:, 0, :].sum(0)

        return gw_sum, gu_sum, gk, gv, None  # None for 'reverse' arg


def wkv_runner(k, v, w, u, reverse=False):
    return WKVFunc.apply(w, u, k, v, reverse)


# ==========================================
# 3. MODULES
# ==========================================

class BiRWKVLayer(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd
        self.rkv_proj = nn.Linear(n_embd, n_embd * 6, bias=False)
        self.output_proj = nn.Linear(n_embd * 2, n_embd, bias=False)

        # Forward Params
        self.time_decay = nn.Parameter(torch.empty(n_embd))
        self.time_first = nn.Parameter(torch.empty(n_embd))

        # Backward Params
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
        # Shape: [B, T, 2, 3, C] -> 2 directions, 3 (r, k, v)
        rkv = self.rkv_proj(x).view(B, T, 2, 3, C)

        # --- Forward Direction ---
        r_f = rkv[:, :, 0, 0]
        k_f = rkv[:, :, 0, 1]
        v_f = rkv[:, :, 0, 2]
        w_f = -torch.exp(self.time_decay)
        u_f = self.time_first

        # reverse=False
        out_f = wkv_runner(k_f, v_f, w_f, u_f, reverse=False)
        out_f = torch.sigmoid(r_f) * out_f

        # --- Backward Direction ---
        r_b = rkv[:, :, 1, 0]
        k_b = rkv[:, :, 1, 1]
        v_b = rkv[:, :, 1, 2]
        w_b = -torch.exp(self.time_decay_rev)
        u_b = self.time_first_rev

        # reverse=True
        # NO FLIP, NO MEMORY COPY
        out_b = wkv_runner(k_b, v_b, w_b, u_b, reverse=True)
        out_b = torch.sigmoid(r_b) * out_b

        # Concat
        out_cat = torch.cat([out_f, out_b], dim=-1)
        return self.output_proj(out_cat)


class BiRWKVBlock(nn.Module):
    def __init__(self, n_embd, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.attn = BiRWKVLayer(n_embd)
        self.ffn_proj = nn.Linear(n_embd, n_embd * mlp_ratio, bias=False)
        self.ffn_out = nn.Linear(n_embd * mlp_ratio, n_embd, bias=False)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn_out(F.gelu(self.ffn_proj(self.ln2(x))))
        return x


class BiRWKV(nn.Module):
    def __init__(self, input_size, num_layers=1):
        super().__init__()
        self.blocks = nn.ModuleList([
            BiRWKVBlock(input_size) for _ in range(num_layers)
        ])
        # 3. Add a Final Norm at the end of the backbone
        self.ln_f = nn.LayerNorm(input_size)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return x


# ==========================================
# 4. TESTING
# ==========================================

if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, C = 2, 128, 64
    device = "cuda"

    print(f"Testing BiRWKV with Triton (Optimized Memory)...")
    model = BiRWKV(input_size=C, num_layers=1).to(device)

    # Input
    x = torch.randn(B, T, C, device=device, requires_grad=True)

    # Forward
    y = model(x)
    loss = y.mean()

    print(f"Forward Output Mean: {y.mean().item():.6f}")

    # Backward
    loss.backward()

    print("Gradient Check:")
    print(f"x.grad norm: {x.grad.norm().item():.6f}")

    # Verify backward direction params are getting grads
    layer = model.blocks[0].attn
    print(f"Time Decay (Fwd) Grad: {layer.time_decay.grad.norm().item():.6f}")
    print(f"Time Decay (Rev) Grad: {layer.time_decay_rev.grad.norm().item():.6f}")

    if layer.time_decay_rev.grad.norm().item() == 0:
        print("ERROR: Backward sequence gradient is zero! Logic error in REVERSE kernel.")
    else:
        print("SUCCESS: Both directions learning.")
