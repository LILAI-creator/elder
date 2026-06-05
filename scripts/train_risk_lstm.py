import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from classifier.risk_lstm_multitask import RiskLSTM
from dataset.le2i_dataset import Le2IDataset
# ↑ 导入你的模型和数据集


# ==================================================
# 🔥 训练函数入口
# ==================================================
def train():

    # ==================================================
    # 1️⃣ 自动选择设备（GPU优先）
    # ==================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 作用：
    # - 有GPU用GPU
    # - 没GPU用CPU

    # ==================================================
    # 2️⃣ 构建训练/验证数据集
    # ==================================================
    train_dataset = Le2IDataset(split="train")
    val_dataset = Le2IDataset(split="val")
    # 作用：
    # - split="train" → 训练数据
    # - split="val"   → 验证数据

    # ==================================================
    # 3️⃣ DataLoader（训练加载器）
    # ==================================================
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,        # 每次训练32条序列
        shuffle=True,         # 打乱数据（防止顺序学习）
        num_workers=2,        # 多线程加载数据
        drop_last=True,       # 丢掉最后不完整batch（稳定训练）
        pin_memory=True       # GPU加速数据传输
    )

    # ==================================================
    # 4️⃣ DataLoader（验证加载器）
    # ==================================================
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,        # 验证集不打乱
        num_workers=2,
        pin_memory=True
    )

    # ==================================================
    # 5️⃣ 初始化模型
    # ==================================================
    model = RiskLSTM(input_dim=102).to(device)
    # 作用：
    # - 创建LSTM多任务模型
    # - 放到GPU/CPU上

    # ==================================================
    # 6️⃣ 优化器（Adam）
    # ==================================================
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,            # 学习率
        weight_decay=1e-5   # L2正则（防过拟合）
    )

    # ==================================================
    # 7️⃣ Loss函数（多任务）
    # ==================================================

    # 分类损失（风险预测）
    loss_risk_fn = nn.BCEWithLogitsLoss()
    # ↑ logits版本更稳定（内部自动sigmoid）

    # 回归损失（时间预测）
    loss_time_fn = nn.SmoothL1Loss()
    # ↑ 比MSE更稳，对异常值不敏感

    # ==================================================
    # 8️⃣ 多任务权重
    # ==================================================
    λ = 0.5
    # 作用：
    # - 控制 time loss 在总loss中的权重
    # - 防止回归/分类某一方压制另一方

    # ==================================================
    # 9️⃣ 学习率调度器
    # ==================================================
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',      # 监控loss最小化
        factor=0.5,      # 学习率衰减50%
        patience=3       # 3轮不提升就下降
    )

    # ==================================================
    # 🔟 训练轮数 & 最优模型记录
    # ==================================================
    epochs = 30
    best_val_loss = float("inf")
    # 用于保存最优模型

    # ==================================================
    # 1️⃣1️⃣ 开始训练循环
    # ==================================================
    for epoch in range(epochs):

        model.train()
        # ↑ 开启训练模式（启用dropout等）

        total_loss = 0

        # ==================================================
        # 1️⃣2️⃣ 遍历训练数据
        # ==================================================
        for batch_idx, batch in enumerate(train_loader):

            # --------------------------
            # 取数据并搬到设备
            # --------------------------
            x = batch["x"].to(device)             # (B, 30, 102)
            y_risk = batch["risk"].to(device)     # (B, 1)
            y_time = batch["time"].to(device)     # (B, 1)

            # --------------------------
            # 前向传播
            # --------------------------
            pred_risk, pred_time = model(x)

            # --------------------------
            # 计算分类loss
            # --------------------------
            loss_risk = loss_risk_fn(pred_risk, y_risk)

            # --------------------------
            # 计算回归loss
            # --------------------------
            loss_time = loss_time_fn(pred_time, y_time)

            # --------------------------
            # 多任务总loss
            # --------------------------
            loss = loss_risk + λ * loss_time

            # --------------------------
            # 清空梯度
            # --------------------------
            optimizer.zero_grad()

            # --------------------------
            # 反向传播
            # --------------------------
            loss.backward()

            # ⭐关键：防止LSTM梯度爆炸
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            # --------------------------
            # 更新参数
            # --------------------------
            optimizer.step()

            # --------------------------
            # 累计loss（按样本）
            # --------------------------
            total_loss += loss.item() * x.size(0)

            # --------------------------
            # 打印训练日志
            # --------------------------
            if batch_idx % 20 == 0:
                print(
                    f"[Epoch {epoch+1}] "
                    f"Batch {batch_idx} | "
                    f"risk={loss_risk.item():.4f} | "
                    f"time={loss_time.item():.4f} | "
                    f"total={loss.item():.4f}"
                )

        # ==================================================
        # 1️⃣3️⃣ 验证阶段
        # ==================================================
        model.eval()
        # ↑ 关闭dropout等随机性

        val_loss = 0

        with torch.no_grad():
            # ↑ 不计算梯度（节省显存 + 加速）

            for batch in val_loader:

                x = batch["x"].to(device)
                y_risk = batch["risk"].to(device)
                y_time = batch["time"].to(device)

                pred_risk, pred_time = model(x)

                loss_risk = loss_risk_fn(pred_risk, y_risk)
                loss_time = loss_time_fn(pred_time, y_time)

                loss = loss_risk + λ * loss_time

                val_loss += loss.item() * x.size(0)

        # ==================================================
        # 1️⃣4️⃣ 计算平均loss
        # ==================================================
        train_loss = total_loss / len(train_dataset)
        val_loss = val_loss / len(val_dataset)

        print("\n====================================")
        print(f"Epoch {epoch+1} finished")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val   Loss: {val_loss:.4f}")
        print("====================================\n")

        # ==================================================
        # 1️⃣5️⃣ 学习率调度
        # ==================================================
        scheduler.step(val_loss)

        # ==================================================
        # 1️⃣6️⃣ 保存最优模型
        # ==================================================
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss
            }, "best_risk_lstm.pth")

            print("💾 Saved best model")


# ==================================================
# 🚀 程序入口
# ==================================================
if __name__ == "__main__":
    train()