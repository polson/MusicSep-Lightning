import torch
import torch.nn as nn
from torch.nn import functional as F
import triton
import triton.language as tl


# ==========================================
# 1. TRITON KERNEL (CUDA Fused Operation)
# ==========================================

@triton.jit
def wkv_kernel(
        k_ptr, v_ptr, w_ptr, u_ptr, out_ptr,
        state_num_ptr, state_den_ptr, state_max_ptr,
        batch_size, seq_len, n_embd,
        B_stride, T_stride, C_stride,  # Strides for K, V, Out
        SO_stride, SI_stride,  # Strides for State (Outer: Batch, Inner: Channel)
        has_initial_state: tl.constexpr
):
    # Parallelize over Batch (program_id(0)) and Channel (program_id(1))
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)

    # Load parameters specific to this channel
    # w and u are [C], so we just offset by channel_idx
    w = tl.load(w_ptr + channel_idx)
    u = tl.load(u_ptr + channel_idx)

    # -----------------------------------------------------------
    # Initialize State (Register Fast Memory)
    # -----------------------------------------------------------
    if has_initial_state:
        state_ptr_offset = batch_idx * SO_stride + channel_idx * SI_stride
        num = tl.load(state_num_ptr + state_ptr_offset)
        den = tl.load(state_den_ptr + state_ptr_offset)
        max_state = tl.load(state_max_ptr + state_ptr_offset)
    else:
        num = 0.0
        den = 0.0
        max_state = -1.0e38

    # Base pointers for this specific batch/channel sequence
    k_ptr_base = k_ptr + batch_idx * B_stride + channel_idx * C_stride
    v_ptr_base = v_ptr + batch_idx * B_stride + channel_idx * C_stride
    out_ptr_base = out_ptr + batch_idx * B_stride + channel_idx * C_stride

    # -----------------------------------------------------------
    # The Loop (Fuses seq_len operations into one kernel)
    # -----------------------------------------------------------
    for t in range(seq_len):
        # Load current token k, v
        k = tl.load(k_ptr_base + t * T_stride)
        v = tl.load(v_ptr_base + t * T_stride)

        # --- Calculate Output ---
        max_for_output = tl.maximum(max_state, k + u)
        e1 = tl.exp(max_state - max_for_output)
        e2 = tl.exp(k + u - max_for_output)

        num_out = e1 * num + e2 * v
        den_out = e1 * den + e2

        # Write output
        tl.store(out_ptr_base + t * T_stride, num_out / den_out)

        # --- Update State ---
        max_for_state = tl.maximum(max_state + w, k)
        e1_s = tl.exp(max_state + w - max_for_state)
        e2_s = tl.exp(k - max_for_state)

        num = e1_s * num + e2_s * v
        den = e1_s * den + e2_s
        max_state = max_for_state

    # -----------------------------------------------------------
    # Save Final State
    # -----------------------------------------------------------
    state_ptr_offset = batch_idx * SO_stride + channel_idx * SI_stride
    tl.store(state_num_ptr + state_ptr_offset, num)
    tl.store(state_den_ptr + state_ptr_offset, den)
    tl.store(state_max_ptr + state_ptr_offset, max_state)


def wkv_forward_wrapper(k, v, w, u, state=None):
    """Helper to launch the kernel"""
    B, T, C = k.shape

    # Ensure contiguous memory for the GPU kernel
    k, v, w, u = k.contiguous(), v.contiguous(), w.contiguous(), u.contiguous()
    output = torch.empty_like(k)

    # Prepare State
    if state is None:
        s_num = torch.zeros((B, C), device=k.device, dtype=torch.float32)
        s_den = torch.zeros((B, C), device=k.device, dtype=torch.float32)
        s_max = torch.zeros((B, C), device=k.device, dtype=torch.float32)
        has_initial = False
    else:
        s_num, s_den, s_max = state
        # Kernel requires contiguous memory for pointers to work
        s_num, s_den, s_max = s_num.contiguous(), s_den.contiguous(), s_max.contiguous()
        has_initial = True

    grid = (B, C)  # Launch a kernel for every Batch and Channel

    wkv_kernel[grid](
        k, v, w, u, output,
        s_num, s_den, s_max,
        B, T, C,
        k.stride(0), k.stride(1), k.stride(2),
        s_num.stride(0), s_num.stride(1),
        has_initial_state=has_initial
    )

    return output, (s_num, s_den, s_max)


# ==========================================
# 2. MODEL DEFINITION
# ==========================================

PREV_X_TIME = 0
NUM_STATE = 1
DEN_STATE = 2
MAX_STATE = 3
PREV_X_CHANNEL = 4


class RWKVConfig:
    def __init__(self, n_embd, n_layer=1, bias=True, intermediate_size=None):
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.bias = bias
        self.intermediate_size = intermediate_size


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class ChannelMixing(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.layer_id = layer_id
        n_embd = config.n_embd
        intermediate_size = config.intermediate_size if config.intermediate_size is not None else 4 * n_embd

        self.key_proj = nn.Linear(n_embd, intermediate_size, bias=False)
        self.value_proj = nn.Linear(intermediate_size, n_embd, bias=False)
        self.receptance_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.time_mix_key = nn.Parameter(torch.empty(1, 1, n_embd))
        self.time_mix_receptance = nn.Parameter(torch.empty(1, 1, n_embd))
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            self.time_mix_key.uniform_(0.0, 1.0)
            self.time_mix_receptance.uniform_(0.0, 1.0)

    def forward(self, x, state=None):
        if state is not None:
            prev_x = state[self.layer_id, :, [PREV_X_CHANNEL], :]
            new_state = state.clone()
            new_state[self.layer_id, :, [PREV_X_CHANNEL], :] = x[:, [-1], :]
        else:
            prev_x = self.time_shift(x)
            new_state = None

        receptance = x * self.time_mix_receptance + prev_x * (1 - self.time_mix_receptance)
        receptance = self.receptance_proj(receptance)
        key = x * self.time_mix_key + prev_x * (1 - self.time_mix_key)
        key = self.key_proj(key)
        value = self.value_proj(torch.square(torch.relu(key)))
        out = F.sigmoid(receptance) * value
        return out, new_state


class TimeMixingOptimized(nn.Module):
    """
    Optimized TimeMixing using Triton Kernel
    """

    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.layer_id = layer_id

        n_embd = config.n_embd
        attn_sz = n_embd

        self.key_proj = nn.Linear(n_embd, attn_sz, bias=False)
        self.value_proj = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance_proj = nn.Linear(n_embd, attn_sz, bias=False)
        self.output_proj = nn.Linear(attn_sz, n_embd, bias=False)

        self.time_decay = nn.Parameter(torch.empty(attn_sz))
        self.time_first = nn.Parameter(torch.empty(attn_sz))
        self.time_mix_key = nn.Parameter(torch.empty(1, 1, n_embd))
        self.time_mix_value = nn.Parameter(torch.empty(1, 1, n_embd))
        self.time_mix_receptance = nn.Parameter(torch.empty(1, 1, n_embd))
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            self.time_mix_key.uniform_(0.0, 1.0)
            self.time_mix_value.uniform_(0.0, 1.0)
            self.time_mix_receptance.uniform_(0.0, 1.0)
            self.time_decay.uniform_(-6.0, -5.0)
            self.time_first.uniform_(-3.0, 0.0)

    def forward(self, x, state=None):
        # 1. Time Shift (Channel-wise mixing)
        if state is not None:
            prev_x = state[self.layer_id, :, [PREV_X_TIME], :]
            new_state = state.clone()
            new_state[self.layer_id, :, [PREV_X_TIME], :] = x[:, [-1], :]
        else:
            prev_x = self.time_shift(x)
            new_state = None

        # 2. Projections
        receptance = x * self.time_mix_receptance + prev_x * (1 - self.time_mix_receptance)
        receptance = self.receptance_proj(receptance)

        key = x * self.time_mix_key + prev_x * (1 - self.time_mix_key)
        key = self.key_proj(key)

        value = x * self.time_mix_value + prev_x * (1 - self.time_mix_value)
        value = self.value_proj(value)

        # 3. PREPARE CUDA INPUTS
        # Convert parameters to float to ensure kernel compatibility
        w = -torch.exp(self.time_decay).float()
        u = self.time_first.float()

        # Extract WKV state if available
        wkv_state_in = None
        if new_state is not None:
            s_num = new_state[self.layer_id, :, NUM_STATE, :]
            s_den = new_state[self.layer_id, :, DEN_STATE, :]
            s_max = new_state[self.layer_id, :, MAX_STATE, :]
            wkv_state_in = (s_num, s_den, s_max)

        # 4. RUN CUDA KERNEL (Replaces the python loop)
        wkv_out, (final_num, final_den, final_max) = wkv_forward_wrapper(
            key, value, w, u, wkv_state_in
        )

        # 5. Update State tensors
        if new_state is not None:
            new_state[self.layer_id, :, NUM_STATE, :] = final_num
            new_state[self.layer_id, :, DEN_STATE, :] = final_den
            new_state[self.layer_id, :, MAX_STATE, :] = final_max

        # 6. Final Output
        rwkv = F.sigmoid(receptance) * wkv_out
        rwkv = self.output_proj(rwkv)

        return rwkv, new_state


class Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = TimeMixingOptimized(config, layer_id)  # Using Optimized version
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.ffn = ChannelMixing(config, layer_id)

    def forward(self, x, state=None):
        residual = x
        x, state = self.attn(self.ln_1(x), state=state)
        x = x + residual
        residual = x
        x, state = self.ffn(self.ln_2(x), state=state)
        x = x + residual
        return x, state


class RWKV_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0.0, bidirectional=False):
        super().__init__()
        if bidirectional: raise NotImplementedError("RWKV cannot be bidirectional")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.input_proj = nn.Linear(input_size, hidden_size, bias=bias) if input_size != hidden_size else None
        config = RWKVConfig(n_embd=hidden_size, n_layer=num_layers, bias=bias)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(num_layers)])
        self.ln_f = LayerNorm(hidden_size, bias=bias)

    def _init_state(self, batch_size, device):
        state = torch.zeros(self.num_layers, batch_size, 5, self.hidden_size, device=device, dtype=torch.float32)
        state[:, :, MAX_STATE, :] = -1e38
        return state

    def _state_to_lstm_format(self, state):
        h_n = state[:, :, NUM_STATE, :].contiguous()
        c_n = state[:, :, DEN_STATE, :].contiguous()  # Simplification for API compatibility
        return (h_n, c_n)

    def _lstm_format_to_state(self, h_0, c_0, batch_size, device):
        state = self._init_state(batch_size, device)
        if h_0 is not None: state[:, :, NUM_STATE, :] = h_0
        if c_0 is not None: state[:, :, DEN_STATE, :] = c_0
        return state

    def forward(self, input, hx=None):
        if self.batch_first: input = input.transpose(0, 1)
        seq_len, batch_size, _ = input.size()
        device = input.device

        x = self.input_proj(input) if self.input_proj else input
        # RWKV expects (Batch, Seq, Channel) for the inner attention logic usually,
        # but our Kernel assumes (Batch, Seq, Channel) as standard layout.
        # The LSTM interface inputs (Seq, Batch, Channel).
        # Let's transpose to Batch First for the core logic.
        x = x.transpose(0, 1)

        if hx is None:
            state = self._init_state(batch_size, device)
        else:
            h_0, c_0 = hx
            state = self._lstm_format_to_state(h_0, c_0, batch_size, device)

        for block in self.blocks:
            x, state = block(x, state)

        x = self.ln_f(x)

        # Return to Seq First
        output = x.transpose(0, 1)
        if self.batch_first: output = output.transpose(0, 1)

        return output, self._state_to_lstm_format(state)


# ==========================================
# 3. TEST
# ==========================================
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA detected. Running Optimized RWKV...")
        device = "cuda"

        model = RWKV_LSTM(128, 256, num_layers=2, batch_first=True).to(device)
        x = torch.randn(4, 128, 128).to(device)  # Batch=4, Seq=128, Dim=128

        # Warmup
        model(x)

        # Timing
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        out, _ = model(x)
        end.record()
        torch.cuda.synchronize()

        print(f"In: {x.shape}, Out: {out.shape}")
        print(f"Time: {start.elapsed_time(end):.2f}ms")
        print("Sanity check (No NaNs):", not torch.isnan(out).any().item())
    else:
        print("No CUDA detected. Install Torch+CUDA and Triton to run this.")
