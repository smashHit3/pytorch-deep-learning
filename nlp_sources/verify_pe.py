"""
===================================================================
 位置编码可视化 & 修复验证脚本 (Positional Encoding Visualizer)
===================================================================

今日学习目标：
  1. 直观理解 Sinusoidal Positional Encoding 的数学原理
  2. 验证 PositionalEncoding 的修复是否正确
  3. 学会用 "位置间余弦相似度" 检验位置编码的合理性

运行：
  cd /workspace/nlp_sources && python verify_pe.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

# ------------------------------------------------------------------
# 1) 你修复的 PositionalEncoding
# ------------------------------------------------------------------
from models.transformer import PositionalEncoding


# ------------------------------------------------------------------
# 2) 用于对比的 "错误版本"（即你原来的实现）
# ------------------------------------------------------------------
class PositionalEncodingOldBuggy(nn.Module):
    """原来的有问题的实现 —— 用来做对比"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# ------------------------------------------------------------------
# 3) 核心检验：不同位置的编码是否不同
# ------------------------------------------------------------------
def check_position_uniqueness(model, batch_size=4, seq_len=16, d_model=32):
    """每个位置的编码都应当是唯一的。"""
    x = torch.zeros(batch_size, seq_len, d_model)  # 全零输入 → 输出就是纯位置编码
    out = model(x)                                  # (batch_size, seq_len, d_model)

    # 取 batch[0] 的各个位置编码做对比
    per_pos = out[0]                               # (seq_len, d_model)

    # 计算两两余弦相似度
    per_pos_norm = per_pos / (per_pos.norm(dim=1, keepdim=True) + 1e-9)
    sim = per_pos_norm @ per_pos_norm.T             # (seq_len, seq_len)

    diag = sim.diag()                               # 自己和自己的相似度应为 1.0
    off_diag = sim.clone()
    off_diag.fill_diagonal_(0.0)                    # 非对角线上的元素：应明显 < 1.0

    max_off_diag = off_diag.abs().max().item()
    diag_mean = diag.mean().item()

    print(f"  对角（自己 vs 自己）: 平均={diag_mean:.4f} （应接近 1.0）")
    print(f"  最大非对角相似度    : {max_off_diag:.4f} （应显著 < 1.0，越小表示各位置越独特）")
    return sim


# ------------------------------------------------------------------
# 4) 打印正弦/余弦波的模式
# ------------------------------------------------------------------
def print_pe_pattern(pe_matrix, num_positions=8, num_channels=12):
    """打印位置编码的数值，直观感受 sin/cos 的规律

    预期观察：
      - 同一列（同一 channel）：随位置不同周期性变化
      - 同一行（同一位置）：各 channel 有不同频率
    """
    print(f"\n  位置编码数值矩阵 (前 {num_positions} 个位置 × 前 {num_channels} 个 channel)：")
    print("  " + "-" * (8 * num_channels + 4))

    header = f"  pos\\ch" + "".join(f"{c:>7}" for c in range(num_channels))
    print(header)
    print("  " + "-" * (8 * num_channels + 4))

    for pos in range(num_positions):
        row_vals = "".join(f"{pe_matrix[pos, c]:>+7.3f}" for c in range(num_channels))
        print(f"  {pos:>5}  {row_vals}")

    print("  " + "-" * (8 * num_channels + 4))
    print("  观察：同一列（同一channel）随位置变化呈正弦/余弦波动")
    print("  观察：小channel索引=高频波动，大channel索引=低频缓慢变化")


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def main():
    D_MODEL = 64
    SEQ_LEN = 16
    BATCH_SIZE = 4

    print("=" * 70)
    print("  Step 1: 理解 Sinusoidal 位置编码的数学直觉")
    print("=" * 70)
    print(f"""
  公式：
    PE(pos, 2i)   = sin( pos / 10000^(2i/d_model) )
    PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model) )

  为什么这么设计？
    👉 每个位置 pos 得到一个唯一的 d_model 维向量
    👉 相邻位置的编码向量很相似（余弦相似度高）
    👉 较远位置的编码向量差异较大（体现顺序信息）
    👉 具有平移不变性的相对位置表达能力
""")

    print("=" * 70)
    print("  Step 2: 可视化位置编码的数值模式")
    print("=" * 70)

    pe_layer = PositionalEncoding(d_model=D_MODEL, dropout=0.0)
    pe_matrix = pe_layer.pe[0]                        # (max_len, d_model)
    print_pe_pattern(pe_matrix.detach().numpy())

    print("\n" + "=" * 70)
    print("  Step 3: 验证你的修复 —— 每个位置都有编码")
    print("=" * 70)

    print("\n  ✅ 修复后的版本（正确）：")
    sim_fixed = check_position_uniqueness(pe_layer, BATCH_SIZE, SEQ_LEN, D_MODEL)

    print("\n  ❌ 原来的 Bug 版本（对比）：")
    pe_old = PositionalEncodingOldBuggy(d_model=D_MODEL, dropout=0.0)
    check_position_uniqueness(pe_old, BATCH_SIZE, SEQ_LEN, D_MODEL)

    print("\n" + "=" * 70)
    print("  Step 4: 位置间余弦相似度矩阵（前 8 个位置）")
    print("=" * 70)
    print("\n  矩阵 [i, j] 表示 位置 i 和 位置 j 的编码相似度：")
    print("   - 对角线 = 1.0 （自己和自己）")
    print("   - 越靠近对角线 = 越相似（相邻位置）")
    print("   - 越远离对角线 = 越不同（体现位置的绝对距离）")
    print()

    short_sim = sim_fixed[:8, :8]
    header = "      " + "".join(f"j={j:<6}" for j in range(8))
    print(header)
    print("   " + "-" * (8 * 7))
    for i in range(8):
        row = "".join(f"{short_sim[i, j].item():>+6.3f} " for j in range(8))
        print(f"  i={i}  {row}")

    print("\n" + "=" * 70)
    print("  Step 5: 端到端 smoke test —— 模型前向不 crash")
    print("=" * 70)

    from models.transformer import transformer_classifier

    vocab_size = 1024
    embedding_dim = 128
    model = transformer_classifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_heads=4,
        num_layers=2,
        hidden_dim=256,
        num_classes=4,
        dropout=0.1,
        max_seq_len=64,
    )

    x = torch.randint(0, vocab_size, (8, 64))   # (batch=8, seq_len=64)
    logits = model(x)
    print(f"  输入 shape : {tuple(x.shape)}")
    print(f"  输出 shape : {tuple(logits.shape)}")
    assert logits.shape == (8, 4), f"Shape mismatch: {logits.shape}"
    print("  ✅ 前向传播通过！位置编码修复有效。")

    # ------------------------------------------------------------------
    # 今日学习总结
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  📘 今日学习要点回顾")
    print("=" * 70)
    print("""
  1. Batch-first vs Sequence-first
     一定要注意你的输入 x 的第 0 维是什么！
     batch_first=True  → x.shape = (batch, seq, d_model)
     batch_first=False → x.shape = (seq, batch, d_model)
     这是 PyTorch / Transformer 中最常见的形状错误来源。

  2. 形状广播 (broadcasting) 是 "双刃剑"
     它让错误的代码"看起来能跑"，但实际上输出是错的。
     解决方法：写小的单元测试 + 可视化关键张量。

  3. 位置编码本身很有趣
     - sin/cos 组合：让任意两个位置 k 和 pos+k 的相对关系可被表示
     - 这也是后来 RoPE (Rotary Positional Encoding) 的动机
     - RoPE 是当前 LLM（如 LLaMA）最常用的位置编码方式
       👉 这是你接下来可以深入的方向

  4. 工程建议
     在你的训练代码中加一个 sanity check：
       小数据集上跑 1-2 个 batch，打印 loss 是否合理下降
""")


if __name__ == "__main__":
    main()
