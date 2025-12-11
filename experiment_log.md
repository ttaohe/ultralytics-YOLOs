# 🧪 Experiment Log & Workflow

> **核心原则**：先记录计划 -> 建 Git 分支 -> 写代码 -> 跑实验 -> 填结果 -> (成功则 Merge / 失败则切回)

---

## 🛠️ Git 工作流作弊条 (Cheat Sheet)

| 场景 | Git 指令流程 | 说明 |
|:---|:---|:---|
| **开始新实验** | `git checkout main`<br>`git pull`<br>`git checkout -b exp/[name]` | 永远从最新的 main 只有切出新分支 |
| **实验开发中** | `git add .`<br>`git commit -m "[exp] implemented xyz"` | 随时存档，不用担心弄乱主分支 |
| **实验成功 ✅** | `git checkout main`<br>`git merge exp/[name]`<br>`git push` | 将成果合并回主干 |
| **实验失败 ❌** | `git checkout main`<br>`git branch -D exp/[name]` | 直接丢弃实验分支，主干不受污染 |

---

## 📅 Experiment Tracker

| ID | Status | Git Branch | Experiment Name | Hypothesis / Goal | Results / Notes |
|:---|:---|:---|:---|:---|:---|
| **EXP-001** | ❌ Fail | `main` | Baseline Initial | 测试 YOLO12n 基准效果 | 训练卡顿，Loss 不降。原因：IO 瓶颈 + 分辨率不足。 |
| **EXP-002** | ✅ Pass | `main` | Video Channel Stacking | 验证 Stacking 机制 | `verify` 脚本通过。解决了 Shape Mismatch。 |
| **EXP-003** | 🔄 Run | `main` | Baseline Optimized | 修复卡顿 + High-Res Crop | 开启 `baseline_mode`。IO 降低 50%。正在运行中... |
| **EXP-004** | ⏳ Plan | `exp/sparse-mem` | **Sparse Memory V1** | 验证 P3 层各类 Attention + TopK | 计划：实现 `select_topk`，移至 P3 层。 |

---

## 📝 Current Plan: Sparse Memory V1 (EXP-004)

**目标**：在不增加显存爆炸风险的前提下，利用 Memory Attention 捕捉小目标时序特征。

**执行步骤**：
1.  [ ] **Git**: `git checkout -b exp/sparse-memory`
2.  [ ] **Model**: 修改 `video_block.py`，实现 `select_topk_features(features, k)`。
3.  [ ] **Config**: 创建 `yolo12-video-p3.yaml`，将 Attention 插入 P3 层。
4.  [ ] **Train**: 运行 `train_video.py`，观察显存占用和前几轮 Loss。
