# DFCGraph-DiM fMRI/BOLD Time-Series Diffusion Model Architecture

本文档用于描述当前项目中的 DFCGraph-DiM 模型架构，目标是给图像生成模型绘制模型结构图。默认实验配置来自 `Config/fmri_seq256_dfc.yaml`，用于 fMRI/BOLD 时间序列生成、预测和序列延长。

## 1. 输入数据与扩散任务

输入是标准化后的 BOLD 滑动窗口：

- 输入窗口 `x_0`: `[B, T, R]`
- 默认 `T = 256`
- 默认 `R = 219`
- `B` 是 batch size
- `T` 是时间长度
- `R` 是 ROI / brain region 数量

训练时模型不是直接预测未来，而是做扩散模型的去噪重建。训练过程先采样扩散步 `t`，再把干净窗口 `x_0` 加噪得到：

```text
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
```

模型输入是：

```text
x_t: [B, 256, 219]
t:   [B]
```

模型输出是对干净窗口的估计：

```text
x_hat_0: [B, 256, 219]
```

当前模型采用 `x_0` prediction 形式，即主网络直接输出 `x_hat_0`，再由扩散公式反推噪声用于采样。

## 2. 总体结构

DFCGraph-DiM 有两个并行分支：

1. **Time Branch**
   - 关注 BOLD 时间动态。
   - 沿时间维度处理 token。
   - 使用双向 SelectSSM / Mamba 风格模块。

2. **Feature / ROI Branch**
   - 关注 ROI 之间的动态功能连接。
   - 每个 ROI 是一个 token。
   - 使用 Dynamic Functional Connectivity Graph，也就是 DFCGraph。
   - DFCGraph 根据当前样本窗口和扩散步动态生成 ROI-to-ROI 邻接矩阵。

两个分支最后通过 ROI-wise gate 融合，输出完整的 BOLD 窗口重建结果。

总体数据流：

```text
x_t [B,T,R], t
        |
        |---------------- Time Branch ----------------|
        |                                             |
        |---------------- ROI / DFCGraph Branch -------|
                                                      |
                 ROI-wise gated fusion
                          |
                    x_hat_0 [B,T,R]
```

## 3. Time Branch

### 3.1 Time Embedding

Time branch 把每个时间点当作 token。输入 `x_t [B,T,R]` 经过线性层：

```text
Linear(R -> H)
```

得到：

```text
time_tokens: [B,T,H]
```

默认 hidden size：

```text
H = 256
```

然后加入 learnable positional encoding：

```text
time_tokens = time_tokens + learnable_time_position
```

这一步保留时间顺序信息。

### 3.2 Bidirectional Encoder

Time branch 的 encoder 由多个 `MambaEncoderBlock` 组成。每个 block 包含：

- LayerNorm
- 正向 SelectSSM
- 反向 SelectSSM
- MLP
- residual connection

每个 block 的核心形式：

```text
y_forward  = SelectSSM(LN(x))
y_backward = reverse(SelectSSM(reverse(LN(x))))
x = x + y_forward + y_backward
x = x + MLP(LN(x))
```

因此它是双向时间建模：

- 正向 SSM 捕获过去到未来的动态。
- 反向 SSM 捕获未来到过去的上下文。
- 两者相加形成完整时间上下文。

### 3.3 Diffusion-Conditioned SSM Decoder

Time branch 后面接 `Decoder_M`，由多个 `DiMBlock` 组成。每个 `DiMBlock` 使用扩散步 `t` 的 embedding 进行 AdaLN modulation。

扩散步嵌入：

```text
t -> sinusoidal embedding -> MLP -> c_t [B,H]
```

每个 `DiMBlock` 包含：

- LayerNorm without affine
- AdaLN modulation from `c_t`
- 正向 SelectSSM
- 反向 SelectSSM
- MLP
- gated residual

概念形式：

```text
c_t -> shift, scale, gate
x_norm = modulate(LN(x), shift, scale)
ssm_out = SelectSSM_forward(x_norm) + SelectSSM_backward(x_norm)
x = x + gate_ssm * ssm_out
x = x + gate_mlp * MLP(modulated_LN(x))
```

Time branch 最后经过：

```text
Linear(H -> R)
```

输出：

```text
x_time: [B,T,R]
```

## 4. ROI / DFCGraph Branch

### 4.1 ROI Feature Embedding

Feature branch 把每个 ROI 当作 token。输入 `x_t [B,T,R]` 先转置为：

```text
[B,R,T]
```

每个 ROI 的完整时间轨迹 `[T]` 通过线性层映射为 hidden token：

```text
Linear(T -> H)
```

得到：

```text
roi_tokens: [B,R,H]
```

这表示每个 ROI 一个 token，每个 token 编码该 ROI 在当前窗口内的 BOLD 时间轨迹。

### 4.2 Dynamic Functional Graph Conditioner

DFCGraph 模块的核心是 `DynamicFunctionalGraphConditioner`。它根据当前样本的 ROI token、当前加噪窗口 `x_t` 和扩散步 `t` 生成动态功能连接图。

输入：

```text
x_t:        [B,T,R]
roi_tokens: [B,R,H]
t:          [B]
```

输出：

```text
A_dyn:         [B, heads, R, R]
roi_context:   [B,R,H]
graph_context: [B,H]
```

默认参数：

```text
heads = 4
graph_topk = 16
graph_rank = 64
```

#### 4.2.1 Token QK Attention Graph

先对 ROI tokens 做 LayerNorm，然后映射到多头 Q、K、V：

```text
Q = Linear(roi_tokens)
K = Linear(roi_tokens)
V = Linear(roi_tokens)
```

形状：

```text
Q, K, V: [B, heads, R, graph_rank]
```

用 QK 相似度得到 ROI-to-ROI 图 logits：

```text
graph_logits = Q @ K^T / sqrt(graph_rank)
```

形状：

```text
graph_logits: [B, heads, R, R]
```

这部分表示由神经网络根据 ROI token 学到的动态 ROI 关系。

#### 4.2.2 Self-Loop Bias

给每个 ROI 加自环偏置：

```text
graph_logits += self_loop_bias * I
```

目的是保留每个 ROI 自身信息，避免图聚合退化为全局平均。

#### 4.2.3 Raw BOLD Correlation Graph

如果 `use_raw_correlation=True`，模型会从当前窗口 `x_t` 直接计算 BOLD 相关矩阵：

```text
corr = corrcoef(x_t over time)
```

形状：

```text
corr: [B,R,R]
```

因为高噪声扩散步下的 `x_t` 含噪严重，直接相关矩阵不可靠，所以用扩散步控制相关矩阵注入强度：

```text
corr_gate = 1 - t / (diffusion_steps - 1)
```

低噪声步：

```text
corr_gate 较大，更多使用当前 BOLD 相关矩阵
```

高噪声步：

```text
corr_gate 较小，减少 raw correlation 的影响
```

融合形式：

```text
graph_logits += corr_gate * sigmoid(raw_corr_scale) * corr
```

#### 4.2.4 Top-k Sparse Graph

对每个 ROI 的连接只保留：

- 自环
- top-k 动态邻居

默认 `graph_topk=16`，即每个 ROI 最多保留 16 条连接，包括自环。

这样可以避免所有 ROI 互相平均，增强功能连接图的稀疏性和可解释性。

#### 4.2.5 Dynamic Adjacency Matrix

最后对每一行做 softmax：

```text
A_dyn = softmax(masked_graph_logits, dim=-1)
```

得到：

```text
A_dyn: [B, heads, R, R]
```

每一行表示一个 ROI 从其他 ROI 聚合信息的权重。

### 4.3 ROI Context Aggregation

用动态图聚合 V：

```text
context_heads = A_dyn @ V
```

再拼接多个 head，经过线性层：

```text
roi_context = Linear(concat_heads(context_heads))
```

得到：

```text
roi_context: [B,R,H]
```

然后对所有 ROI 求平均，再经过 MLP 得到全局图上下文：

```text
graph_context = MLP(mean(roi_context over ROI))
```

形状：

```text
graph_context: [B,H]
```

`graph_context` 表示当前窗口的全局功能网络状态。

## 5. DFCGraph Decoder Block

Feature branch 使用多个 `DynamicFunctionalGraphBlock`。每个 block 接收：

```text
roi_tokens:    [B,R,H]
roi_context:   [B,R,H]
graph_context: [B,H]
t embedding:   [B,H]
```

扩散步上下文和图上下文相加：

```text
c = t_embedding + graph_context
```

然后通过 AdaLN modulation 生成：

```text
shift_graph, scale_graph, gate_graph,
shift_mlp,   scale_mlp,   gate_mlp
```

block 内部包含两部分：

1. **Graph Residual Update**

```text
graph_update = MLP(modulate(LN(roi_context), shift_graph, scale_graph))
roi_tokens = roi_tokens + graph_scale * gate_graph * graph_update
```

`graph_scale` 默认初始化为 `0.0`，因此训练初始阶段模型接近无图结构，避免动态图一开始破坏原始表示。

2. **ROI Token MLP Update**

```text
roi_tokens = roi_tokens + gate_mlp * MLP(modulate(LN(roi_tokens), shift_mlp, scale_mlp))
```

注意：动态图分支不再把 ROI 顺序当作一维序列去做 SSM 扫描，而是通过动态图完成 ROI mixing。

Feature branch 最终输出：

```text
roi_tokens_final: [B,R,H]
```

再经过：

```text
Linear(H -> T)
```

得到：

```text
x_feature_before_transpose: [B,R,T]
```

转置回：

```text
x_feature: [B,T,R]
```

## 6. Feature Fusion

模型不是简单相加两个分支，而是用动态图上下文生成 ROI-wise fusion gate。

先计算：

```text
fusion_context = timestep_embedding(t) + graph_context
```

再通过 MLP + sigmoid：

```text
fusion_gate = sigmoid(Linear(SiLU(fusion_context)))
```

形状：

```text
fusion_gate: [B,R]
```

扩展到时间维：

```text
fusion_gate: [B,1,R]
```

最终融合：

```text
x_hat_0 = x_time + fusion_gate * x_feature
```

其中：

- `x_time [B,T,R]` 提供时间动态主干。
- `x_feature [B,T,R]` 提供动态 ROI 功能连接修正。
- `fusion_gate [B,1,R]` 控制每个 ROI 接收多少功能图分支信息。

最终输出：

```text
x_hat_0: [B,256,219]
```

## 7. 损失函数

训练目标由三部分组成：

```text
total_loss = weighted_reconstruction_loss
           + lambda1 * Fourier_loss
           + lambda2 * MMD_correlation_loss
```

在代码里：

- `lambda1` 对应 `l_loss`
- `lambda2` 对应 `mmd_alpha`
- 当前 DFCGraph 配置默认 `l_loss=0.1`
- 当前 DFCGraph 配置默认 `mmd_alpha=0.01`

### 7.1 Time-Domain Reconstruction Loss

模型输出 `x_hat_0` 和目标 `x_0` 计算逐点 MSE：

```text
L_recon = MSE(x_hat_0, x_0)
```

形状上先得到：

```text
[B,T,R]
```

再对每个样本取平均，并乘以扩散步相关权重：

```text
L_recon_weighted = mean(L_recon over T,R) * loss_weight(t)
```

### 7.2 Fourier Loss

为了约束频域结构，对预测窗口和真实窗口沿时间维做 FFT。

先把数据从：

```text
[B,T,R]
```

转成：

```text
[B,R,T]
```

然后计算：

```text
FFT(x_hat_0)
FFT(x_0)
```

Fourier loss 对实部和虚部分别做 MSE：

```text
L_fft = MSE(real(FFT_pred), real(FFT_gt))
      + MSE(imag(FFT_pred), imag(FFT_gt))
```

然后加到逐点重建损失中：

```text
train_loss += ff_weight * lambda1 * L_fft
```

其中：

```text
ff_weight = sqrt(seq_length) / 5
```

默认 `seq_length=256`，所以 `ff_weight=3.2`。

### 7.3 Correlation MMD Loss

MMD loss 用于约束 ROI 功能连接分布。

对真实窗口和预测窗口分别计算 ROI cross-correlation distribution：

```text
target_dist = corr_distribution(x_0)
pred_dist   = corr_distribution(x_hat_0)
```

对每个样本，先计算 ROI-ROI 相关矩阵，再取非对角下三角元素，得到 ROI pair correlation 分布。

然后使用 RBF kernel 的 batch MMD：

```text
L_mmd = BMMD(target_dist, pred_dist, kernel="rbf")
```

最终：

```text
total_loss = train_loss.mean() + mmd_alpha * L_mmd
```

如果 `mmd_alpha=0`，则不计算 MMD 项。

## 8. Figure Generation Requirements

请不要把所有细节塞进一张图。希望画图风格类似 DiM-TS 论文：先画一张总体结构大图，再为关键模块单独画 zoom-in 子图。最终需要生成一组图，而不是一张超复杂总图。

### 8.1 Figure Set Overview

请生成以下 5 张图：

1. **Figure 1: Overall DFCGraph-DiM Framework**
   - 一张横向大图。
   - 展示输入、两个并行分支、融合、输出和训练损失。
   - 只画高层模块，不展开每个模块内部细节。
   - 不要在图内写具体 shape，例如不要写 `[B,256,219]`；只标注 `x_t`, `x_0`, `x_time`, `x_feature`, `x_hat_0`。
   - 不要在图片底部添加默认维度说明文字。

2. **Figure 2: Time Branch Module**
   - 单独展开 Time Embedding、Bi-Mamba Encoder、Diffusion Mamba Decoder。
   - 类似论文中单独展示 DiFM / DiPM 的方式。

3. **Figure 3: Dynamic Functional Graph Conditioner**
   - 单独展开 DFCGraph 如何生成 `A_dyn`。
   - 重点展示 Q/K/V attention graph、raw BOLD correlation、timestep gate、self-loop、top-k sparse softmax。

4. **Figure 4: DFCGraph Decoder Block and ROI Mixing**
   - 单独展开 `DynamicFunctionalGraphBlock`。
   - 展示 `roi_context` 如何通过 AdaLN modulation 和 gated residual 更新 ROI tokens。

5. **Figure 5: ROI-wise Gate and Losses**
   - 单独展示 `fusion_gate` 如何作用于 ROI branch feature、`x_time + gated x_feature`、以及三个 loss。
   - 这张图可以比前几张更简洁。

### 8.2 Figure 1: Overall Framework

画成左到右的数据流图。只保留高层结构：

```text
Noisy BOLD x_t + diffusion step t
        |
        |------ Time Branch ------> x_time
        |
        |------ ROI / DFCGraph Branch ------> gated x_feature
        |
direct branch addition
        |
x_hat_0
        |
MSE + Fourier + Correlation MMD losses
```

总体图中每个大模块画成一个 block：

- `Time Embedding`
- `Bi-Mamba Encoder`
- `Diffusion Mamba Decoder`
- `ROI Embedding`
- `DFCGraph Conditioner`
- `DFCGraph Decoder Blocks`
- `ROI-wise Gated Feature` inside the ROI / DFCGraph Branch
- `Add` node for final branch fusion
- `Losses`

不要在 Figure 1 中展开 Q/K/V、top-k、FFT、MMD 等细节，这些放到后续子图。Figure 1 也不要写具体 tensor shape，例如 `[B,256,219]`、`[B,T,R]` 等；这些 shape 可以放在图注或后续模块子图里。

Figure 1 的视觉要求：

- Time Branch 用蓝色系。
- ROI / DFCGraph Branch 用绿色系。
- Diffusion step `t` 用紫色小节点，并用紫色虚线箭头只连接到真正使用扩散步条件的模块：Diffusion Mamba Decoder、DFCGraph Conditioner、ROI-wise Gated Feature。不要把 `t` 连接到 Losses 或最终加号。
- Loss 模块放在输出右侧或右下角，用橙色。
- Figure 1 不要展开 `A_dyn` 的内部使用流程，不要画 `Dynamic ROI Graph A_dyn -> Graph-guided ROI mixing` 这类细箭头；这些细节放到 Figure 3。
- 在 `DFCGraph Conditioner` 模块内部或旁边只保留一行小字：`dynamic ROI graph`。如果画矩阵或大脑网络图标，只作为这个模块的小图标，不要画成额外主流程节点。
- ROI branch 输出前应包含一个合并模块 `ROI-wise Gated Feature`，表示 ROI branch feature 已经被 gate 处理；不要把 `ROI-wise Gate -> gated x_feature` 拆成两个连续模块，也不要在 Figure 1 写 gate 公式。
- 最终融合不要画一个大的 `Graph-conditioned ROI-wise Gated Fusion` 模块；只画一个加号节点 `+`，表示 `x_hat_0 = x_time + gated x_feature`。
- 图中只标注变量名：`x_t`, `x_0`, `x_time`, `x_feature`, `gated x_feature`, `x_hat_0`。
- 不要在图片底部写 `Default dimensions: T=256, R=219, H=256` 这类说明。
- Clean BOLD `x_0` 只作为 loss target 进入 Losses 模块。`x_0` 不能连接到 fusion、gate、add node、Time Branch 或 ROI Branch；不要让 `x_0` 出现一条指向原先 `Graph-conditioned ROI-wise Gated Fusion` 或任何融合模块的线。建议把 `x_0` 放在 Losses 模块附近作为 target input，避免主图下方出现很长的横线。

### 8.3 Figure 2: Time Branch Module

这张图单独展开 Time Branch。

输入：

```text
x_t [B,T,R]
t [B]
```

模块顺序：

```text
x_t
 -> Linear(R -> H)
 -> Learnable Positional Encoding
 -> Bi-Mamba Encoder
 -> Diffusion Mamba Decoder
 -> Linear(H -> R)
 -> x_time [B,T,R]
```

Bi-Mamba Encoder 需要画成一个 zoom-in block：

```text
LN(x)
 -> Forward SelectSSM
 -> Backward SelectSSM on reversed sequence
 -> sum + residual
 -> MLP + residual
```

Diffusion Mamba Decoder 需要标出：

- `t -> sinusoidal embedding -> MLP -> c_t`
- `c_t -> AdaLN shift / scale / gate`
- forward SelectSSM
- backward SelectSSM
- gated residual

Figure 2 不需要画 ROI 图，也不需要画 loss。

### 8.4 Figure 3: Dynamic Functional Graph Conditioner

这张图是 DFCGraph 的核心模块子图。

输入：

```text
x_t [B,T,R]
roi_tokens [B,R,H]
t [B]
```

请分成两条信息来源：

1. **Learned ROI-token graph path**

```text
roi_tokens
 -> LayerNorm
 -> Q, K, V projections
 -> QK^T / sqrt(rank)
 -> learned graph logits [B,heads,R,R]
```

2. **Raw BOLD correlation path**

```text
x_t
 -> temporal centering and normalization
 -> corrcoef over time
 -> corr [B,R,R]
```

然后画 timestep gate：

```text
t -> corr_gate = 1 - t / (diffusion_steps - 1)
```

融合：

```text
graph_logits = QK logits + self-loop bias + corr_gate * raw_corr_scale * corr
```

再画：

```text
top-k sparse mask + row softmax
 -> A_dyn [B,heads,R,R]
```

最后：

```text
A_dyn @ V
 -> concat heads
 -> Linear
 -> roi_context [B,R,H]
 -> mean over ROI + MLP
 -> graph_context [B,H]
```

视觉要求：

- `A_dyn` 画成多头邻接矩阵或脑区连接网络。
- 标注 `sample-specific` 和 `timestep-conditioned`。
- 明确写出：`not a fixed static graph`。
- `top-k=16` 可以画成稀疏矩阵效果。

### 8.5 Figure 4: DFCGraph Decoder Block

这张图单独展开 `DynamicFunctionalGraphBlock`。

输入：

```text
roi_tokens [B,R,H]
roi_context [B,R,H]
graph_context [B,H]
timestep embedding c_t [B,H]
```

先画 conditioning：

```text
c = c_t + graph_context
 -> AdaLN modulation
 -> shift_graph, scale_graph, gate_graph
 -> shift_mlp, scale_mlp, gate_mlp
```

Graph update path：

```text
roi_context
 -> LayerNorm
 -> AdaLN modulate
 -> MLP
 -> graph_scale * gate_graph
 -> residual add to roi_tokens
```

ROI token update path：

```text
updated roi_tokens
 -> LayerNorm
 -> AdaLN modulate
 -> MLP
 -> gate_mlp
 -> residual add
```

输出：

```text
updated roi_tokens [B,R,H]
```

需要特别标注：

- `graph_scale initialized to 0`
- `ROI mixing is performed by dynamic graph, not by treating ROI order as a 1D sequence`

### 8.6 Figure 5: Fusion and Losses

这张图展示最终输出和训练目标。

Fusion 部分：

```text
graph_context [B,H] + timestep_embedding(t) [B,H]
 -> SiLU + Linear
 -> sigmoid
 -> fusion_gate [B,R]
 -> expand to [B,1,R]
```

融合公式：

```text
x_hat_0 = x_time + fusion_gate * x_feature
```

Loss 部分：

```text
x_hat_0 vs x_0
 -> time-domain MSE
 -> Fourier loss over temporal FFT
 -> correlation MMD over ROI-ROI correlation distribution
 -> total loss
```

总损失公式：

```text
total_loss = weighted MSE + lambda1 * Fourier loss + lambda2 * Correlation MMD loss
```

视觉要求：

- Fusion gate 可以画成一个 `sigmoid gate` 图标。
- Losses 用三个并列小框：`MSE`, `FFT`, `MMD`.
- MMD 旁边画小型 ROI correlation matrix distribution。

### 8.7 Global Style Requirements

整组图请保持统一论文风格：

- 白色背景。
- 细黑色箭头。
- 圆角矩形模块。
- 重要模块用浅色填充，不要使用深色背景。
- Time Branch: blue.
- DFCGraph Branch: green.
- Diffusion conditioning: purple.
- Fusion: gray or violet.
- Losses: orange.
- 每张图都要有清晰标题，例如 `Overall Framework`, `Dynamic Functional Graph Conditioner`。
- 每张图控制信息密度，文字不要太小。
- 图中形状标注使用 `B,T,R,H`，并在图注中注明默认 `T=256, R=219, H=256`。
- 不要画成一个巨大的 all-in-one diagram；必须是 1 张总体图 + 多张模块细节图。
