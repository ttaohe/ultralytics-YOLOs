"""
分析 SAM2 模型中的 Temporal Positional Encoding (maskmem_tpos_enc)
提取编码并计算余弦相似度矩阵，验证相邻编码是否更相似
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics.utils.downloads import attempt_download_asset

def extract_temporal_encoding(model_name="sam2_b.pt"):
    """
    从 SAM2 checkpoint 中直接提取 maskmem_tpos_enc 参数
    
    Args:
        model_name: SAM2 模型名称，如 'sam2_b.pt', 'sam2_l.pt' 等
        
    Returns:
        tpos_enc: Temporal positional encoding tensor, shape (num_maskmem, 1, 1, d_model)
    """
    print(f"Loading SAM2 checkpoint: {model_name}")
    
    # 下载并加载 checkpoint
    checkpoint_path = attempt_download_asset(model_name)
    with open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f, map_location='cpu')
    
    # SAM2 checkpoint 格式: {"model": state_dict}
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # 查找 maskmem_tpos_enc 参数
    tpos_enc_key = None
    for key in state_dict.keys():
        if "maskmem_tpos_enc" in key:
            tpos_enc_key = key
            break
    
    if tpos_enc_key is None:
        # 尝试查找所有可能的键
        print("Available keys containing 'tpos' or 'maskmem':")
        for key in state_dict.keys():
            if "tpos" in key.lower() or "maskmem" in key.lower():
                print(f"  - {key}")
        raise KeyError(f"Could not find 'maskmem_tpos_enc' in checkpoint. Available keys: {list(state_dict.keys())[:10]}...")
    
    tpos_enc = state_dict[tpos_enc_key]
    print(f"Found {tpos_enc_key}: shape {tpos_enc.shape}")
    
    return tpos_enc
    
def compute_cosine_similarity_matrix(tpos_enc):
    """
    计算时间编码之间的余弦相似度矩阵
    
    Args:
        tpos_enc: Temporal positional encoding, shape (num_maskmem, 1, 1, d_model)
        
    Returns:
        similarity_matrix: Cosine similarity matrix, shape (num_maskmem, num_maskmem)
    """
    # 展平编码: (num_maskmem, 1, 1, d_model) -> (num_maskmem, d_model)
    num_maskmem = tpos_enc.shape[0]
    enc_flat = tpos_enc.squeeze(1).squeeze(1)  # (num_maskmem, d_model)
    
    # 归一化
    enc_normalized = F.normalize(enc_flat, p=2, dim=1)
    
    # 计算余弦相似度矩阵
    similarity_matrix = torch.mm(enc_normalized, enc_normalized.t())  # (num_maskmem, num_maskmem)
    
    return similarity_matrix.cpu().numpy(), enc_flat.cpu().numpy()

def visualize_similarity_matrix(similarity_matrix, model_name="sam2_b", save_dir="runs/analysis"):
    """
    可视化余弦相似度矩阵
    
    Args:
        similarity_matrix: Cosine similarity matrix
        model_name: 模型名称（用于保存文件）
        save_dir: 保存目录
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    num_maskmem = similarity_matrix.shape[0]
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. 热力图
    ax1 = axes[0]
    im = ax1.imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax1.set_title(f'SAM2 Temporal Encoding Cosine Similarity Matrix\n({model_name}, num_maskmem={num_maskmem})', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Encoding Index (lag)', fontsize=11)
    ax1.set_ylabel('Encoding Index (lag)', fontsize=11)
    
    # 设置刻度标签（lag 值）
    # 注意：SAM2 中索引映射是 num_maskmem - t_pos - 1
    # 索引 0 对应最远的帧（lag=7，7帧前），索引 num_maskmem-1 对应最近的帧（lag=1，1帧前/前一帧）
    # lag=1 表示"1帧前"，即最接近当前帧的第一帧（前一帧）
    lag_labels = [f"lag={num_maskmem-i}" for i in range(num_maskmem)]
    ax1.set_xticks(range(num_maskmem))
    ax1.set_yticks(range(num_maskmem))
    ax1.set_xticklabels(lag_labels, rotation=45, ha='right')
    ax1.set_yticklabels(lag_labels)
    
    # 添加数值标注
    for i in range(num_maskmem):
        for j in range(num_maskmem):
            text = ax1.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Cosine Similarity', rotation=270, labelpad=20)
    
    # 2. 相邻编码的相似度分析
    ax2 = axes[1]
    
    # 计算相邻编码的相似度（对角线偏移1）
    adjacent_similarities = []
    non_adjacent_similarities = []
    
    for i in range(num_maskmem):
        for j in range(num_maskmem):
            if abs(i - j) == 1:  # 相邻
                adjacent_similarities.append(similarity_matrix[i, j])
            elif abs(i - j) > 1:  # 非相邻
                non_adjacent_similarities.append(similarity_matrix[i, j])
    
    # 绘制分布对比
    ax2.hist(adjacent_similarities, bins=20, alpha=0.6, label=f'Adjacent (mean={np.mean(adjacent_similarities):.3f})', 
             color='green', edgecolor='black')
    ax2.hist(non_adjacent_similarities, bins=20, alpha=0.6, 
             label=f'Non-adjacent (mean={np.mean(non_adjacent_similarities):.3f})', 
             color='red', edgecolor='black')
    ax2.set_xlabel('Cosine Similarity', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Similarity Distribution: Adjacent vs Non-adjacent Encodings', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    save_path = save_dir / f"{model_name}_temporal_encoding_similarity.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {save_path}")
    
    # 打印统计信息
    print("\n" + "="*60)
    print("Statistical Analysis:")
    print("="*60)
    print(f"Adjacent encodings similarity:")
    print(f"  Mean: {np.mean(adjacent_similarities):.4f}")
    print(f"  Std:  {np.std(adjacent_similarities):.4f}")
    print(f"  Min:  {np.min(adjacent_similarities):.4f}")
    print(f"  Max:  {np.max(adjacent_similarities):.4f}")
    print(f"\nNon-adjacent encodings similarity:")
    print(f"  Mean: {np.mean(non_adjacent_similarities):.4f}")
    print(f"  Std:  {np.std(non_adjacent_similarities):.4f}")
    print(f"  Min:  {np.min(non_adjacent_similarities):.4f}")
    print(f"  Max:  {np.max(non_adjacent_similarities):.4f}")
    print(f"\nDifference (Adjacent - Non-adjacent): {np.mean(adjacent_similarities) - np.mean(non_adjacent_similarities):.4f}")
    
    # 检查是否相邻编码更相似
    if np.mean(adjacent_similarities) > np.mean(non_adjacent_similarities):
        print("\n✅ Hypothesis confirmed: Adjacent encodings are MORE similar!")
    else:
        print("\n❌ Hypothesis rejected: Adjacent encodings are NOT more similar.")
    
    plt.show()

def analyze_encoding_structure(tpos_enc, model_name="sam2_b"):
    """
    分析编码的结构特征
    
    Args:
        tpos_enc: Temporal positional encoding tensor
        model_name: 模型名称
    """
    enc_flat = tpos_enc.squeeze(1).squeeze(1).cpu().numpy()  # (num_maskmem, d_model)
    num_maskmem, d_model = enc_flat.shape
    
    print("\n" + "="*60)
    print("Encoding Structure Analysis:")
    print("="*60)
    print(f"Shape: ({num_maskmem}, {d_model})")
    print(f"Mean magnitude per encoding:")
    for i in range(num_maskmem):
        mag = np.linalg.norm(enc_flat[i])
        print(f"  Index {i} (lag={num_maskmem-i}): {mag:.4f}")
    
    print(f"\nEncoding variance per dimension:")
    var_per_dim = np.var(enc_flat, axis=0)
    print(f"  Mean variance: {np.mean(var_per_dim):.4f}")
    print(f"  Max variance:  {np.max(var_per_dim):.4f}")
    print(f"  Min variance:  {np.min(var_per_dim):.4f}")

def main():
    """主函数"""
    # 可以尝试不同的 SAM2 模型
    models_to_test = ["sam2_b.pt", "sam2_l.pt"]  # 可以根据需要添加更多
    
    for model_name in models_to_test:
        try:
            print(f"\n{'='*60}")
            print(f"Analyzing: {model_name}")
            print(f"{'='*60}")
            
            # 提取编码
            tpos_enc = extract_temporal_encoding(model_name)
            
            # 分析结构
            analyze_encoding_structure(tpos_enc, model_name.replace('.pt', ''))
            
            # 计算相似度矩阵
            similarity_matrix, enc_flat = compute_cosine_similarity_matrix(tpos_enc)
            
            # 可视化
            visualize_similarity_matrix(similarity_matrix, model_name.replace('.pt', ''))
            
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            continue

if __name__ == "__main__":
    main()

