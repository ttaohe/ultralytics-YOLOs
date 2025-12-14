## SparseMemoryAttention 改进规划（草案）

### 背景与目标

- **任务目标**：在视频检测中，当当前帧目标被遮挡或模糊时，希望模型能**从历史帧中取回有用特征**，补充当前帧，从而保持检测/跟踪的稳定性。
- **当前机制**：`SparseMemoryAttention` 使用 Top‑K 稀疏化 + Global RoPE，将历史若干帧的稀疏 token 组成 memory，与当前帧 dense token 做 self‑attn + cross‑attn，再 residual 回主干。

---

### 当前 SparseMemoryAttention 的关键设计（简述）

- **稀疏选择**：对每帧 `(B, L, D)` 做 L2 norm，取 Top‑K 作为 sparse token，保存 `(mem_sparse, coords_sparse)`。
- **空间信息**：为当前帧 & 历史帧都生成 `(x,y)` 网格坐标，通过 `GlobalRoPEAttention` 在 self/cross‑attn 中注入 RoPE。
- **时间维度**：历史帧以 FIFO 形式堆入 `memory_bank`，推理时简单在 token 维 concat；无显式时间衰减。
- **更新策略**：每个新帧先用 memory 做注意力，然后把自己编码+稀疏后压入 `memory_bank`，最多保留 `max_memory` 帧。

---

### 改进方向清单（后续每条可单独开分支）

#### 1. **目标感知的 Top‑K 稀疏选择**

- **动机**：当前 `topk` 只看 L2 norm，可能更偏向纹理/背景强特征，不一定聚焦在“目标区域”，对补遮挡不够针对。
- **思路**
  - 利用检测头输出（例如 `obj/conf`、类别 logits）对特征点加权，再做 Top‑K：
    - 只在 **预测框内** 的 token 中选 Top‑K（背景只保留极少量）。  
    - 或者 score = α·L2 + β·obj_score，将目标区域 token 的优先级提高。
  - 需要考虑：inference 时如何高效拿到每个 token 对应的预测框/score。
- **建议分支名**：`exp_sparse_topk_target_aware`

#### 2. **加入时间建模与权重衰减**

- **动机**：当前 memory 中各历史帧 token 等权拼接，模型不知道“哪一帧离当前最近”，对长时间遮挡不友好。
- **思路**
  - 在 `SparseMemoryAttention` 的 `retrieve_memory_inference` 里，为不同帧的 token 乘以时间权重（如几何衰减：最近帧权重最大）。
  - 或为每帧增加简易 temporal encoding（类似 `maskmem_tpos_enc`），并在 attention 里使用。
- **潜在形式**
  - 简单版：`weight_t = gamma^(Δt)`，在拼接前对每帧 `mem_feats` 乘以各自的权重。
  - 进阶版：把时间编码也纳入 RoPE（扩展到 (x, y, t)）。
- **建议分支名**：`exp_sparse_temporal_decay`

#### 3. **按目标 / 轨迹组织的 Memory（Track‑level Memory）**

- **动机**：现在 memory 是全局 token 池，不区分目标；被遮挡的小目标如果当初不够“显著”，可能根本没有进 memory。
- **思路（中长期）**
  - 利用跟踪/ID 分配（或通过匈牙利匹配等）为每个检测目标维护一条独立的特征序列：
    - `memory_bank[track_id] = [(feat_t, coord_t, t), ...]`
  - 当前帧每个检测框只 attend 该目标自己的历史序列（或再加少量邻居/背景）。
- **难度**：需要引入简单的 tracker 或在 head 内做 ID 分配，改动范围较大。
- **建议分支名**：`exp_sparse_track_memory`

#### 4. **运动感知的坐标 / 对齐（Motion‑Aware Memory）**

- **动机**：当前用的是每帧局部 `grid (0..H-1, 0..W-1)`，RoPE 默认“同 (x,y) 在所有帧对齐”；当相机或目标移动较快时，这个假设变弱。
- **思路**
  - 简单版：在 RoPE 里加入时间维 `t`，用 `compute_global_cis(x, y, t)` 近似建模时空距离。
  - 加强版：引入光流或估计的位移场，将历史帧坐标 warp 到当前帧再喂入 `compute_global_cis`。
- **影响**：能让模型更明确知道“这个历史 token 在当前帧的大致位置”，更利于从远处/大位移处补特征。
- **建议分支名**：`exp_sparse_motion_aware`

#### 5. **可视化与调试增强（已部分完成，可继续扩展）**

- **已做**
  - Hook `GlobalRoPEAttention` 捕获 Q/K + coords；
  - 实现 **No‑RoPE vs With‑RoPE** attention 分布的 A/B 可视化；
  - 支持按 score 强度调节点的亮度。
- **后续可能扩展**
  - 对同一 memory 帧生成“热力图风格”的 dense map（而不是散点），便于肉眼看 pattern；
  - 针对特定目标/轨迹，画出随时间的 attention 分布变化曲线。
- **建议分支名**：`exp_sparse_viz_enhance`（如有大改再单独开）

#### 6. **消融实验与配置开关**

- **动机**：系统复杂之后，需要明确每个模块对性能/速度的贡献。
- **内容**
  - 在 `yaml` / model config 里给上述组件加开关：
    - 是否启用目标感知 Top‑K；
    - 是否启用 temporal decay；
    - 是否启用 motion‑aware coords 等。
  - 保持一个清晰的 baseline：当前的 `SparseMemoryAttention` 行为作为 `exp_sparse_memoryattn` 分支的“对照组”。
- **建议分支名**：`exp_sparse_ablation_cfg`

---

### Git 工作流建议（结合你给的表格）

- **当前主实验分支**：`exp_sparse_memoryattn`（作为“视频 sparse memory”的主线分支）。  
- **每个改进方向**：
  - 从 `exp_sparse_memoryattn` 切新分支，例如：
    - `git checkout exp_sparse_memoryattn`
    - `git pull`
    - `git checkout -b exp_sparse_topk_target_aware`
  - 在新分支上实现 & 实验；
  - 实验结束后：
    - `git checkout exp_sparse_memoryattn`
    - `git merge exp_sparse_topk_target_aware`
    - `git push`（如果要同步远程）。

后面你可以先选一个你最感兴趣/最容易验证收益的方向（我个人建议 **先做“目标感知的 Top‑K” 或 “时间衰减”**），然后我们按这个工作流一起在新分支里落地实现。