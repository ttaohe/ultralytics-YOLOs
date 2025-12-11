"""
详细分析 SAM2 temporal encoding 的模式，特别是为什么最远帧与相邻帧的相似度是负的
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics.utils.downloads import attempt_download_asset

def extract_and_analyze(model_name="sam2_b.pt"):
    """提取并详细分析 temporal encoding"""
    print(f"Loading SAM2 checkpoint: {model_name}")
    
    checkpoint_path = attempt_download_asset(model_name)
    with open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f, map_location='cpu')
    
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # 查找 maskmem_tpos_enc
    tpos_enc_key = None
    for key in state_dict.keys():
        if "maskmem_tpos_enc" in key:
            tpos_enc_key = key
            break
    
    tpos_enc = state_dict[tpos_enc_key]
    enc_flat = tpos_enc.squeeze(1).squeeze(1)  # (7, 64)
    num_maskmem = enc_flat.shape[0]
    
    # 归一化
    enc_normalized = F.normalize(enc_flat, p=2, dim=1)
    
    # 计算相似度矩阵
    similarity_matrix = torch.mm(enc_normalized, enc_normalized.t()).cpu().numpy()
    
    print("\n" + "="*70)
    print("详细相似度分析")
    print("="*70)
    
    # 分析每一行的相邻相似度
    print("\n每一帧与其相邻帧的相似度:")
    print("-" * 70)
    for i in range(num_maskmem):
        lag = num_maskmem - i
        print(f"\nIndex {i} (lag={lag}):")
        
        # 与所有其他帧的相似度
        similarities = []
        for j in range(num_maskmem):
            if i != j:
                sim = similarity_matrix[i, j]
                distance = abs(i - j)
                lag_j = num_maskmem - j
                similarities.append((j, lag_j, distance, sim))
        
        # 按距离排序
        similarities.sort(key=lambda x: x[2])
        
        for j, lag_j, dist, sim in similarities:
            marker = " ⭐" if dist == 1 else ""
            sign = "✅" if sim > 0 else "❌"
            print(f"  -> Index {j} (lag={lag_j}), distance={dist}: {sim:7.4f} {sign}{marker}")
    
    # 特别分析 Index 0（最远帧）
    print("\n" + "="*70)
    print("重点分析：Index 0 (最远帧, lag=7) 的特殊性")
    print("="*70)
    
    idx0_similarities = similarity_matrix[0, :]
    print(f"\nIndex 0 与所有帧的相似度:")
    for j in range(num_maskmem):
        lag_j = num_maskmem - j
        sim = idx0_similarities[j]
        sign = "✅" if sim > 0 else "❌"
        print(f"  Index {j} (lag={lag_j}): {sim:7.4f} {sign}")
    
    # 分析编码向量的方向
    print("\n" + "="*70)
    print("编码向量方向分析")
    print("="*70)
    
    # 计算每个编码相对于 Index 0 的角度
    idx0_vec = enc_normalized[0:1]  # (1, 64)
    angles = []
    for i in range(num_maskmem):
        vec = enc_normalized[i:i+1]  # (1, 64)
        cos_sim = torch.mm(vec, idx0_vec.t()).item()
        angle_rad = np.arccos(np.clip(cos_sim, -1, 1))
        angle_deg = np.degrees(angle_rad)
        angles.append((i, cos_sim, angle_deg))
    
    print("\n每个编码相对于 Index 0 的角度:")
    for i, cos_sim, angle_deg in angles:
        lag = num_maskmem - i
        print(f"  Index {i} (lag={lag}): cos_sim={cos_sim:7.4f}, angle={angle_deg:6.2f}°")
    
    # 分析编码的幅度
    print("\n" + "="*70)
    print("编码幅度分析")
    print("="*70)
    magnitudes = []
    for i in range(num_maskmem):
        lag = num_maskmem - i
        mag = torch.norm(enc_flat[i]).item()
        magnitudes.append((i, lag, mag))
    
    print("\n每个编码的 L2 范数（幅度）:")
    for i, lag, mag in magnitudes:
        print(f"  Index {i} (lag={lag}): {mag:.4f}")
    
    # 可视化
    visualize_detailed(similarity_matrix, enc_flat.numpy(), model_name.replace('.pt', ''))
    
    return similarity_matrix, enc_flat.numpy()

def visualize_detailed(similarity_matrix, enc_flat, model_name):
    """详细可视化"""
    num_maskmem = similarity_matrix.shape[0]
    save_dir = Path("runs/analysis")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. 相似度矩阵热力图
    ax1 = plt.subplot(2, 3, 1)
    im = ax1.imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax1.set_title(f'Cosine Similarity Matrix\n({model_name})', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Encoding Index (lag)', fontsize=10)
    ax1.set_ylabel('Encoding Index (lag)', fontsize=10)
    lag_labels = [f"lag={num_maskmem-i}" for i in range(num_maskmem)]
    ax1.set_xticks(range(num_maskmem))
    ax1.set_yticks(range(num_maskmem))
    ax1.set_xticklabels(lag_labels, rotation=45, ha='right')
    ax1.set_yticklabels(lag_labels)
    plt.colorbar(im, ax=ax1)
    
    # 2. Index 0 与其他帧的相似度
    ax2 = plt.subplot(2, 3, 2)
    idx0_sims = similarity_matrix[0, :]
    colors = ['red' if s < 0 else 'green' for s in idx0_sims]
    bars = ax2.bar(range(num_maskmem), idx0_sims, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_title('Index 0 (lag=7) vs All Frames', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Target Index (lag)', fontsize=10)
    ax2.set_ylabel('Cosine Similarity', fontsize=10)
    ax2.set_xticks(range(num_maskmem))
    ax2.set_xticklabels(lag_labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    for i, (bar, sim) in enumerate(zip(bars, idx0_sims)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{sim:.2f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
    
    # 3. 相邻帧相似度（按起始帧分组）
    ax3 = plt.subplot(2, 3, 3)
    adjacent_pairs = []
    for i in range(num_maskmem - 1):
        sim = similarity_matrix[i, i+1]
        lag = num_maskmem - i
        adjacent_pairs.append((i, lag, sim))
    
    indices = [p[0] for p in adjacent_pairs]
    lags = [p[1] for p in adjacent_pairs]
    sims = [p[2] for p in adjacent_pairs]
    colors = ['red' if s < 0 else 'green' for s in sims]
    bars = ax3.bar(range(len(adjacent_pairs)), sims, color=colors, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.set_title('Adjacent Frame Similarities', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Starting Index (lag)', fontsize=10)
    ax3.set_ylabel('Cosine Similarity', fontsize=10)
    ax3.set_xticks(range(len(adjacent_pairs)))
    ax3.set_xticklabels([f"lag={l}" for l in lags], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    for bar, sim in zip(bars, sims):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{sim:.2f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
    
    # 4. 编码幅度
    ax4 = plt.subplot(2, 3, 4)
    magnitudes = [np.linalg.norm(enc_flat[i]) for i in range(num_maskmem)]
    lag_labels_short = [f"lag={num_maskmem-i}" for i in range(num_maskmem)]
    bars = ax4.bar(range(num_maskmem), magnitudes, color='steelblue', alpha=0.7, edgecolor='black')
    ax4.set_title('Encoding Magnitude (L2 Norm)', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Encoding Index (lag)', fontsize=10)
    ax4.set_ylabel('Magnitude', fontsize=10)
    ax4.set_xticks(range(num_maskmem))
    ax4.set_xticklabels(lag_labels_short, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    for bar, mag in zip(bars, magnitudes):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{mag:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 5. 相对于 Index 0 的角度
    ax5 = plt.subplot(2, 3, 5)
    enc_normalized = F.normalize(torch.tensor(enc_flat), p=2, dim=1)
    idx0_vec = enc_normalized[0:1]
    angles = []
    for i in range(num_maskmem):
        vec = enc_normalized[i:i+1]
        cos_sim = torch.mm(vec, idx0_vec.t()).item()
        angle_deg = np.degrees(np.arccos(np.clip(cos_sim, -1, 1)))
        angles.append(angle_deg)
    
    bars = ax5.bar(range(num_maskmem), angles, color='orange', alpha=0.7, edgecolor='black')
    ax5.set_title('Angle from Index 0 (lag=7)', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Encoding Index (lag)', fontsize=10)
    ax5.set_ylabel('Angle (degrees)', fontsize=10)
    ax5.set_xticks(range(num_maskmem))
    ax5.set_xticklabels(lag_labels_short, rotation=45, ha='right')
    ax5.grid(True, alpha=0.3)
    for bar, angle in zip(bars, angles):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{angle:.1f}°', ha='center', va='bottom', fontsize=8)
    
    # 6. 距离 vs 相似度散点图
    ax6 = plt.subplot(2, 3, 6)
    distances = []
    similarities = []
    colors_scatter = []
    for i in range(num_maskmem):
        for j in range(i+1, num_maskmem):
            dist = abs(i - j)
            sim = similarity_matrix[i, j]
            distances.append(dist)
            similarities.append(sim)
            colors_scatter.append('red' if sim < 0 else 'green')
    
    scatter = ax6.scatter(distances, similarities, c=colors_scatter, alpha=0.6, s=50, edgecolors='black')
    ax6.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax6.set_title('Distance vs Similarity', fontsize=11, fontweight='bold')
    ax6.set_xlabel('Temporal Distance (frames)', fontsize=10)
    ax6.set_ylabel('Cosine Similarity', fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / f"{model_name}_detailed_temporal_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved detailed visualization to: {save_path}")
    plt.close()

if __name__ == "__main__":
    print("分析 SAM2-B 模型:")
    extract_and_analyze("sam2_b.pt")
    
    print("\n\n" + "="*70)
    print("分析 SAM2-L 模型:")
    print("="*70)
    extract_and_analyze("sam2_l.pt")

