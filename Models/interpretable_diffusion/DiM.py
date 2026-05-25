import torch
import torch.nn as nn
import numpy as np
import math
from Models.interpretable_diffusion.SMamba import StructureAwareSSM, SelectSSM
from functools import partial
from itertools import repeat
import collections.abc


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)
    
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TimestepEmbedder(nn.Module):

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(
            torch.empty(1, max_len, d_model)
        )  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        """
        # print(x.shape)
        x = x + self.pe
        return self.dropout(x)


class DiMBlock(nn.Module):

    def __init__(self, hidden_size, mlp_ratio=4.0, dstate=2, dconv=2, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        self.attn1 = SelectSSM(d_model=hidden_size,d_state=dstate,d_conv=dconv,)
        self.attn2 = SelectSSM(d_model=hidden_size,d_state=dstate,d_conv=dconv,)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        '''
        x = x + gate_msa.unsqueeze(1) * (self.attn1(modulate(self.norm1(x), shift_msa, scale_msa))+self.attn2(modulate(self.norm1(x), shift_msa, scale_msa).flip(dims=[1])).flip(dims=[1]))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x
        '''
        x1, x0, dt, A, B, C, bias = self.attn1(modulate(self.norm1(x), shift_msa, scale_msa))

        x = x + gate_msa.unsqueeze(1) * (x1+self.attn2(modulate(self.norm1(x), shift_msa, scale_msa).flip(dims=[1]))[0].flip(dims=[1]))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x, x0, dt, A, B, C, bias
    
class DiSBlock(nn.Module):

    def __init__(self, hidden_size, mlp_ratio=4.0, dstate=2, dconv=2, conv_num=3, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        self.attn1 = StructureAwareSSM(d_model=hidden_size,d_state=dstate,d_conv=dconv,conv_num=conv_num,)
        self.attn2 = StructureAwareSSM(d_model=hidden_size,d_state=dstate,d_conv=dconv,conv_num=conv_num,)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )

        x1, x0, dt, A, B, C, bias = self.attn1(modulate(self.norm1(x), shift_msa, scale_msa))

        x = x + gate_msa.unsqueeze(1) * (x1+self.attn2(modulate(self.norm1(x), shift_msa, scale_msa).flip(dims=[1]))[0].flip(dims=[1]))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x, x0, dt, A, B, C, bias


class MambaEncoderBlock(nn.Module):

    def __init__(self, hidden_size, mlp_ratio=4.0, dstate=2, dconv=2, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn1 = SelectSSM(d_model=hidden_size,d_state=dstate,d_conv=dconv,)
        self.attn2 = SelectSSM(d_model=hidden_size,d_state=dstate,d_conv=dconv,)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x):
        x = x + self.attn1(self.norm1(x))[0] + self.attn2(self.norm1(x).flip(dims=[1]))[0].flip(dims=[1])
        x = x + self.mlp(self.norm2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8, n_layers=3, mlp_ratio=4.0, d_state=2, d_conv=2):
        super().__init__()
        
        self.encoder_blocks = nn.Sequential(
            *[
                MambaEncoderBlock(
                    hidden_size=hidden_size, 
                    mlp_ratio=mlp_ratio, 
                    dstate=d_state, 
                    dconv=d_conv
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        for index in range(len(self.encoder_blocks)):
            x = self.encoder_blocks[index](x)
        return x


class Decoder_M(nn.Module):

    def __init__(self, hidden_size=512, num_heads=8, n_layers=3, mlp_ratio=4.0, d_state=2, d_conv=2):
        super().__init__()
        
        self.encoder_blocks = nn.Sequential(
            *[
                DiMBlock(
                    hidden_size=hidden_size, 
                    mlp_ratio=mlp_ratio, 
                    dstate=d_state, 
                    dconv=d_conv
                )
                for _ in range(n_layers)
            ]
        )
        self.diffusion_step_emb = TimestepEmbedder(hidden_size)

    def forward(self, x, t):
        identity = x
        toreturn = torch.zeros_like(x)
        c = self.diffusion_step_emb(t)
        x_set = []
        dt_set = []
        A_set = []
        B_set = []
        C_set = []
        bias_set = []
        for index in range(len(self.encoder_blocks)):
            x, x0, dt, A, B, C, bias = self.encoder_blocks[index](x, c)
            toreturn += x
            x += identity
            identity = x

            x_set.append(x0)
            dt_set.append(dt)
            A_set.append(A)
            B_set.append(B)
            C_set.append(C)
            bias_set.append(bias)

        return toreturn, x_set, dt_set, A_set, B_set, C_set, bias_set


class Decoder_S(nn.Module):

    def __init__(self, hidden_size=512, num_heads=8, n_layers=3, mlp_ratio=4.0, d_state=2, d_conv=2, conv_num=3):
        super().__init__()
        
        self.encoder_blocks = nn.Sequential(
            *[
                DiSBlock(
                    hidden_size=hidden_size, 
                    mlp_ratio=mlp_ratio, 
                    dstate=d_state, 
                    dconv=d_conv, 
                    conv_num=conv_num
                )
                for _ in range(n_layers)
            ]
        )
        self.diffusion_step_emb = TimestepEmbedder(hidden_size)

    def forward(self, x, t):
        identity = x
        toreturn = torch.zeros_like(x)
        c = self.diffusion_step_emb(t)
        x_set = []
        dt_set = []
        A_set = []
        B_set = []
        C_set = []
        bias_set = []
        for index in range(len(self.encoder_blocks)):

            x, x0, dt, A, B, C, bias = self.encoder_blocks[index](x, c)
            toreturn += x
            x += identity
            identity = x

            x_set.append(x0)
            dt_set.append(dt)
            A_set.append(A)
            B_set.append(B)
            C_set.append(C)
            bias_set.append(bias)

        return toreturn


class TimeSeries2EmbLinear(nn.Module):
    """
    Encode time series data alone with selected dimension.
    """

    def __init__(
        self,
        hidden_size=512,
        feature_last=True,
        shape=(24, 6),
        dim2emb="time",
        dropout=0,
    ):
        super().__init__()
        assert dim2emb in ["time", "feature"], "Please indicate which dim to emb"

        if feature_last:
            sequence_length, feature_size = shape
        else:
            feature_size, sequence_length = shape

        self.feature_last = feature_last
        self.dim2emb = dim2emb
        self.pos_emb = LearnablePositionalEncoding(
            d_model=hidden_size, max_len=sequence_length
        )
        if dim2emb == "time":
            self.processing = nn.Sequential(
                nn.Linear(feature_size, hidden_size), nn.Dropout(dropout)
            )
        else:
            self.processing = nn.Sequential(
                nn.Linear(sequence_length, hidden_size), nn.Dropout(dropout)
            )

    def forward(self, x):
        if not self.feature_last:
            x = x.permute(0, 2, 1)

        if self.dim2emb == "time":
            x = self.processing(x)
            return self.pos_emb(x)
        return self.processing(x.permute(0, 2, 1))


class AdaptiveFunctionalGraphMixer(nn.Module):
    def __init__(self, num_nodes, hidden_size, graph_rank=64, identity_bias=5.0):
        super().__init__()
        graph_rank = min(graph_rank, hidden_size)
        self.scale = graph_rank ** -0.5
        self.node_emb = nn.Parameter(torch.randn(num_nodes, graph_rank) * 0.02)
        self.identity_bias = nn.Parameter(torch.tensor(float(identity_bias)))
        self.graph_scale = nn.Parameter(torch.zeros(1))
        self.register_buffer('identity', torch.eye(num_nodes), persistent=False)
        self.graph_mlp = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.timestep_emb = TimestepEmbedder(hidden_size)
        self.gate = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )

    def forward(self, x, t):
         # 用自连接偏置避免初始邻接矩阵退化成全局均匀平均。
        identity = self.identity.to(dtype=x.dtype, device=x.device)
        adj_logits = torch.matmul(self.node_emb, self.node_emb.t()) * self.scale
        adj_logits = adj_logits.to(dtype=x.dtype, device=x.device) + self.identity_bias.to(dtype=x.dtype) * identity
        adj = torch.softmax(adj_logits, dim=-1)
        x_graph = torch.matmul(adj.to(dtype=x.dtype, device=x.device), x)
        graph_update = self.graph_mlp(x_graph)
        # 不同扩散步的噪声强度不同，用 t 控制图结构残差的注入强度。
        gate = self.gate(self.timestep_emb(t)).unsqueeze(1)
        # graph_scale 从 0 开始，使模型初始等价于不使用图模块。
        return x + self.graph_scale.to(dtype=x.dtype) * gate * graph_update


class DynamicFunctionalGraphConditioner(nn.Module):
    def __init__(
        self,
        num_nodes,
        hidden_size,
        graph_heads=4,
        graph_topk=16,
        graph_rank=64,
        use_raw_correlation=True,
        diffusion_steps=1000,
    ):
        super().__init__()
        if hidden_size % graph_heads != 0:
            raise ValueError('hidden_size must be divisible by graph_heads.')
        if graph_heads <= 0:
            raise ValueError('graph_heads must be positive.')

        self.num_nodes = num_nodes
        self.graph_heads = graph_heads
        self.graph_rank = min(graph_rank, hidden_size)
        self.graph_topk = graph_topk
        self.use_raw_correlation = use_raw_correlation
        self.diffusion_steps = max(int(diffusion_steps), 1)
        self.scale = self.graph_rank ** -0.5

        self.token_norm = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, graph_heads * self.graph_rank, bias=False)
        self.k_proj = nn.Linear(hidden_size, graph_heads * self.graph_rank, bias=False)
        self.v_proj = nn.Linear(hidden_size, graph_heads * self.graph_rank, bias=False)
        self.out_proj = nn.Linear(graph_heads * self.graph_rank, hidden_size)
        self.context_mlp = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.self_loop_bias = nn.Parameter(torch.tensor(2.0))
        self.raw_corr_scale = nn.Parameter(torch.zeros(1))
        self.register_buffer('identity', torch.eye(num_nodes), persistent=False)

    def _to_heads(self, x):
        b, n, _ = x.shape
        x = x.view(b, n, self.graph_heads, self.graph_rank)
        return x.permute(0, 2, 1, 3).contiguous()

    @staticmethod
    def _window_corrcoef(x):
        x = x - x.mean(dim=1, keepdim=True)
        denom = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True).clamp_min(1e-6))
        x = x / denom
        corr = torch.matmul(x.transpose(1, 2), x)
        return torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    def _apply_topk(self, logits):
        topk = int(self.graph_topk)
        if topk <= 0 or topk >= logits.shape[-1]:
            return logits

        b, h, n, _ = logits.shape
        mask_value = -torch.finfo(logits.dtype).max
        identity_mask = self.identity.to(device=logits.device, dtype=torch.bool).view(1, 1, n, n)
        keep_mask = identity_mask.expand(b, h, n, n).clone()

        # 每个 ROI 行保留自环和 top-k-1 个动态邻居，避免 softmax 退化成全脑平均。
        if topk > 1:
            non_self_logits = logits.masked_fill(identity_mask, mask_value)
            topk_indices = torch.topk(non_self_logits, k=topk - 1, dim=-1).indices
            keep_mask.scatter_(-1, topk_indices, True)
        return logits.masked_fill(~keep_mask, mask_value)

    def forward(self, x_t, roi_tokens, t):
        token_input = self.token_norm(roi_tokens)
        q = self._to_heads(self.q_proj(token_input))
        k = self._to_heads(self.k_proj(token_input))
        v = self._to_heads(self.v_proj(token_input))

        graph_logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        identity = self.identity.to(dtype=graph_logits.dtype, device=graph_logits.device)
        graph_logits = graph_logits + self.self_loop_bias.to(dtype=graph_logits.dtype) * identity.view(1, 1, self.num_nodes, self.num_nodes)

        if self.use_raw_correlation:
            corr = self._window_corrcoef(x_t).to(dtype=graph_logits.dtype)
            # 高噪声扩散步的瞬时相关矩阵不可靠，用 timestep gate 让低噪声步更多参考 BOLD 相关。
            t_scale = t.to(dtype=graph_logits.dtype).clamp_min(0) / max(float(self.diffusion_steps - 1), 1.0)
            corr_gate = (1.0 - t_scale.clamp(0, 1)).view(-1, 1, 1, 1)
            graph_logits = graph_logits + corr_gate * self.raw_corr_scale.sigmoid().to(dtype=graph_logits.dtype) * corr.unsqueeze(1)

        graph_logits = self._apply_topk(graph_logits)
        a_dyn = torch.softmax(graph_logits, dim=-1)
        context_heads = torch.matmul(a_dyn, v)
        context_heads = context_heads.permute(0, 2, 1, 3).contiguous()
        context_heads = context_heads.view(roi_tokens.shape[0], self.num_nodes, self.graph_heads * self.graph_rank)
        roi_context = self.out_proj(context_heads)
        graph_context = self.context_mlp(roi_context.mean(dim=1))
        return a_dyn, roi_context, graph_context


class DynamicFunctionalGraphBlock(nn.Module):
    def __init__(self, hidden_size, mlp_ratio=4.0, graph_residual_init=0.0):
        super().__init__()
        self.norm_graph = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_mlp = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.graph_mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.graph_scale = nn.Parameter(torch.tensor(float(graph_residual_init)))

    def forward(self, x, roi_context, c):
        shift_graph, scale_graph, gate_graph, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        # roi_context 已经由动态图聚合得到，这里只做残差注入，避免把 ROI 顺序当作一维序列扫描。
        graph_update = self.graph_mlp(
            modulate(self.norm_graph(roi_context), shift_graph, scale_graph)
        )
        x = x + self.graph_scale.to(dtype=x.dtype) * gate_graph.unsqueeze(1) * graph_update
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm_mlp(x), shift_mlp, scale_mlp)
        )
        return x


class DynamicFunctionalGraphDecoder(nn.Module):
    def __init__(
        self,
        num_nodes,
        hidden_size=512,
        n_layers=3,
        mlp_ratio=4.0,
        graph_heads=4,
        graph_topk=16,
        graph_rank=64,
        graph_residual_init=0.0,
        use_raw_correlation=True,
        diffusion_steps=1000,
    ):
        super().__init__()
        self.conditioner = DynamicFunctionalGraphConditioner(
            num_nodes=num_nodes,
            hidden_size=hidden_size,
            graph_heads=graph_heads,
            graph_topk=graph_topk,
            graph_rank=graph_rank,
            use_raw_correlation=use_raw_correlation,
            diffusion_steps=diffusion_steps,
        )
        self.blocks = nn.ModuleList(
            [
                DynamicFunctionalGraphBlock(
                    hidden_size=hidden_size,
                    mlp_ratio=mlp_ratio,
                    graph_residual_init=graph_residual_init,
                )
                for _ in range(n_layers)
            ]
        )
        self.diffusion_step_emb = TimestepEmbedder(hidden_size)

    def forward(self, x, x_t, t):
        c_t = self.diffusion_step_emb(t)
        graph_context_sum = torch.zeros_like(c_t)
        last_a_dyn = None
        last_roi_context = None

        for block in self.blocks:
            a_dyn, roi_context, graph_context = self.conditioner(x_t, x, t)
            x = block(x, roi_context, c_t + graph_context)
            graph_context_sum = graph_context_sum + graph_context
            last_a_dyn = a_dyn
            last_roi_context = roi_context

        if len(self.blocks) == 0:
            last_a_dyn, last_roi_context, graph_context_sum = self.conditioner(x_t, x, t)
        else:
            graph_context_sum = graph_context_sum / len(self.blocks)

        return x, last_a_dyn, last_roi_context, graph_context_sum


class DiM(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        num_heads=4,
        n_encoder=2,
        n_decoder=2,
        feature_last=True,
        mlp_ratio=4.0,
        dropout=0,
        input_shape=(24, 6),
        d_state=2,
        d_conv=2,
        conv_num=3,
        trans_mx=None,
        use_dynamic_fc_graph=False,
        graph_heads=4,
        graph_topk=16,
        graph_rank=64,
        graph_residual_init=0.0,
        use_graph_conditioned_fusion=True,
        use_raw_correlation=True,
        diffusion_steps=1000,
    ):
        super().__init__()
        if feature_last:
            sequence_length, feature_size = input_shape
        else:
            feature_size, sequence_length = input_shape

        self.use_dynamic_fc_graph = use_dynamic_fc_graph
        self.use_graph_conditioned_fusion = use_graph_conditioned_fusion

        self.time2emb = TimeSeries2EmbLinear(
            hidden_size=hidden_size,
            feature_last=feature_last,
            shape=input_shape,
            dim2emb="time",
            dropout=dropout,
        )
        self.feature2emb = TimeSeries2EmbLinear(
            hidden_size=hidden_size,
            feature_last=feature_last,
            shape=input_shape,
            dim2emb="feature",
            dropout=dropout,
        )

        self.time_encoder = Encoder(
            hidden_size=hidden_size,
            num_heads=num_heads,
            n_layers=n_encoder,
            mlp_ratio=mlp_ratio,
            d_state=d_state,
            d_conv=d_conv,
        )

        # 本实验让 time branch 使用 SelectSSM，与 feature branch 保持同类 decoder。
        self.time_blocks = Decoder_M(
            hidden_size=hidden_size, 
            num_heads=num_heads, 
            n_layers=n_decoder, 
            d_state=d_state, 
            d_conv=d_conv,
        )

        if self.use_dynamic_fc_graph:
            self.feature_blocks = DynamicFunctionalGraphDecoder(
                num_nodes=feature_size,
                hidden_size=hidden_size,
                n_layers=n_decoder,
                mlp_ratio=mlp_ratio,
                graph_heads=graph_heads,
                graph_topk=graph_topk,
                graph_rank=graph_rank,
                graph_residual_init=graph_residual_init,
                use_raw_correlation=use_raw_correlation,
                diffusion_steps=diffusion_steps,
            )
        else:
            self.feature_encoder = Encoder(
                hidden_size=hidden_size,
                num_heads=num_heads,
                n_layers=n_encoder,
                mlp_ratio=mlp_ratio,
                d_state=d_state,
                d_conv=d_conv,
            )
            self.feature_blocks = Decoder_M(
                hidden_size=hidden_size,
                num_heads=num_heads,
                n_layers=n_decoder,
                d_state=d_state,
                d_conv=d_conv,
            )
            self.feature_graph_mixer = AdaptiveFunctionalGraphMixer(
                num_nodes=feature_size,
                hidden_size=hidden_size,
            )

        self.fc_time = nn.Linear(hidden_size, input_shape[1])
        self.fc_feature = nn.Linear(hidden_size, input_shape[0])
        self.branch_fusion_emb = TimestepEmbedder(hidden_size)
        self.branch_fusion_gate = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, feature_size),
            nn.Sigmoid(),
        )

        self.trans_mx = trans_mx

    def forward(self, x, t):

        x_time = self.time2emb(x)
        x_time = self.time_encoder(x_time)
        x_time, _, _, _, _, _, _ = self.time_blocks(x_time, t)
        x_time = self.fc_time(x_time)

        if self.trans_mx is None:
            graph_input = x
        else:
            graph_input = torch.matmul(self.trans_mx, x.permute(0, 2, 1)).permute(0, 2, 1)

        x_feature = self.feature2emb(graph_input)
        if self.use_dynamic_fc_graph:
            # 动态功能图只在 feature branch 中混合 ROI，time branch 仍负责 BOLD 时间动态建模。
            x_feature, a_dyn, roi_context, graph_context = self.feature_blocks(x_feature, graph_input, t)
        else:
            x_feature = self.feature_encoder(x_feature)
            x_feature = self.feature_graph_mixer(x_feature, t)
            x_feature, x0, dt, A, B, C, bias = self.feature_blocks(x_feature, t)
            graph_context = None
        x_feature = self.fc_feature(x_feature)

        if self.trans_mx is None:
            x_feature = x_feature.permute(0, 2, 1)
        else:
            x_feature = torch.matmul(torch.linalg.inv(self.trans_mx), x_feature)
            x_feature = x_feature.permute(0, 2, 1)

        if self.use_dynamic_fc_graph and self.use_graph_conditioned_fusion:
            # 用动态图的全局状态控制 ROI 级 feature 分支强度，避免图分支在所有脑区上无差别相加。
            fusion_context = self.branch_fusion_emb(t) + graph_context
            fusion_gate = self.branch_fusion_gate(fusion_context).unsqueeze(1)
            return x_time + fusion_gate * x_feature

        return x_feature + x_time
