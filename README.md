## 项目简介：视频行人重识别（BVS + RDB-Mamba）

本项目实现了文档中描述的视频识别方案，包括：

- **BVS（Bayesian Visual Stem）**：基于 ResNet-50 的贝叶斯视觉前端，输出帧级特征均值 `mu` 与不确定性 `sigma2`。
- **RDB-Mamba 骨干**：双分支、双向 Mamba Block，包含质量门控的身份流与标准步长的非身份流。
- **尾部聚合与解耦**：利用不确定性加权的时间聚合得到视频级身份向量，并结合互信息最小化、正交约束、时序平滑与 KL 正则进行训练。

目录结构简要说明：

- `src/models/bvs.py`：BVS 前端。
- `src/models/mamba_blocks.py`：RDB-Mamba 双分支骨干。
- `src/models/heads.py`：不确定性加权聚合与 MINE 互信息估计器。
- `src/models/losses.py`：各类损失函数及总损失 `VideoReIDCriterion`。
- `src/models/reid_model.py`：整体模型 `VideoReIDModel`。
- `src/train.py`：训练脚本示例。

---

## 环境准备

1. 建议使用 **Python 3.9+**，并提前安装 PyTorch（含 CUDA 版本请参考官方说明）。
2. 安装必要依赖（在项目根目录执行）：

```bash
pip install torch torchvision einops mamba-ssm
```

> 如果你已经在自己的环境中安装了这些库，可跳过对应步骤。

---

## 训练示例脚本的使用方法

### 1. 使用自带的 Dummy 数据集快速跑通

当前 `src/train.py` 自带一个演示用的 `DummyVideoDataset`，会随机生成形如 `(T, 3, 224, 224)` 的视频片段，以及随机 ID 标签：

```bash
python -m src.train
```

运行后你会看到类似输出（数字仅为示例）：

```text
Epoch 0:
{'total': 4.1234, 'id': 3.4567, 'triplet': 0.321, 'mi': 0.12, 'orth': 0.01, 'temp': 0.03, 'kl': 0.02}
Epoch 1:
...
```

这说明模型与损失组合、前向与反向传播流程均已正确打通。

### 2. 调整关键超参数

在 `src/train.py` 的 `main()` 中，可以按需修改：

- **`num_classes`**：你的行人 ID 类别数。
- **`feat_dim`**：特征维度（与文档建议一致，默认为 512）。
- **`VideoReIDModel(feat_dim, num_blocks=4)`**：
  - `num_blocks` 为堆叠的 RDB-Mamba Block 个数，可根据文档建议设置为 `6~12`。
- **`VideoReIDCriterion` 中的权重**：
  - `lambda_mi`：互信息最小化损失权重。
  - `lambda_orth`：正交约束权重。
  - `lambda_temp`：身份流时序平滑权重。
  - `lambda_kl`：不确定性 KL 正则权重。

优化器当前使用 **AdamW**，可按需要调整学习率、权重衰减等。

---

## 接入真实数据集的建议

要在真实视频 ReID 数据集（如 MARS、DukeMTMC-VideoReID 等）上训练，你只需要：

1. **实现自己的 Dataset**

   在 `src/train.py` 中参考 `DummyVideoDataset`，编写一个新的 `Dataset`，确保：

   - `__getitem__` 返回 `(video, label)`：
     - `video`：形状为 `(T, 3, H, W)` 的张量，RGB 图像序列。
     - `label`：整型 ID（`0 ~ num_classes-1`）。

2. **替换数据集与 DataLoader**

   在 `main()` 中将

   ```python
   dataset = DummyVideoDataset(...)
   ```

   替换为你的自定义数据集实例，并根据显存设置合适的 `batch_size` 与 `num_workers`。

3. **推理与检索**

   训练好模型后，你可以：

   - 加载 `VideoReIDModel` 权重；
   - 仅前向计算得到输出字典中的 `vid_id`，作为视频级行人特征；
   - 在图库与查询集之间计算余弦或欧氏距离，实现向量检索。

---

## 步长门控（质量门控）的实现说明

在 `src/models/mamba_blocks.py` 中的 `BiMambaLayer`，身份流会根据不确定性 `sigma2` 动态调整“步长”：

1. 对输入特征 `x` 计算原始步长：

   \[
   \Delta_{\text{raw}} = \text{Softplus}(\text{Linear}(x))
   \]

2. 将帧级不确定性 \(\sigma_t^2\) 注入，得到质量门控步长：

   \[
   \Delta_{\text{id}} = \Delta_{\text{raw}} \cdot \exp(-\alpha \cdot \sigma_t^2)
   \]

3. 将步长映射到 \((0, 1)\) 区间，作为对输入的门控系数：

   \[
   \text{gate} = \frac{\Delta_{\text{id}}}{1 + \Delta_{\text{id}}}, \quad
   x_{\text{gated}} = x \odot \text{gate}
   \]

当某些帧质量很低（\(\sigma_t^2\) 较大）时，\(\exp(-\alpha \sigma_t^2)\) 会变小，导致 `gate` 接近 0，从而抑制该帧对 Mamba 状态更新的贡献，近似实现“当前状态不更新，保持历史记忆”的效果，与设计文档中的形式严格一致。

---

如需进一步扩展（例如自定义检索脚本、添加评价指标等），可以在当前结构基础上直接新增对应模块即可。


