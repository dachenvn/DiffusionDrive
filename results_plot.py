"""
DiffusionDrive 训练结果可视化脚本
用法: python results_plot.py
"""
import os
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.visualization.plots import (
    plot_bev_frame,
    plot_bev_with_agent,
    plot_cameras_frame,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

# ========== 配置区（根据你的实际路径修改） ==========
# 训练的模型权重
CHECKPOINT = "/data/experiments/training_diffusiondrive_agent_v2/2026.03.10.11.56.54/lightning_logs/version_0/checkpoints/epoch=199-step=2200.ckpt"
# 数据集路径
DATA_ROOT = Path(os.environ.get("OPENSCENE_DATA_ROOT", "/data/openscene_data"))
# 数据集分割
SPLIT = "mini"
# 输出目录
OUTPUT_DIR = Path("visualization_output")
# 可视化场景数量
NUM_SCENES = 100  # 可视化几个场景
# ================================================

OUTPUT_DIR.mkdir(exist_ok=True)

print("1. 加载 SceneLoader ...")
scene_filter = SceneFilter(num_history_frames=4, num_future_frames=8, frame_interval=5)
scene_loader = SceneLoader(
    data_path=DATA_ROOT / f"navsim_logs/{SPLIT}",
    sensor_blobs_path=DATA_ROOT / f"sensor_blobs/{SPLIT}",
    scene_filter=scene_filter,
    sensor_config=SensorConfig.build_all_sensors(include=[3]),
)
print(f"   共 {len(scene_loader.tokens)} 个可用场景")

print("2. 加载模型 ...")
config = TransfuserConfig(
    trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5),
)
agent = TransfuserAgent(config=config, lr=6e-4, checkpoint_path=CHECKPOINT)
agent.eval()
print("   模型加载完成（CPU 推理）")

print(f"3. 对 {NUM_SCENES} 个场景进行可视化 ...")
tokens = scene_loader.tokens[:NUM_SCENES]

for i, token in enumerate(tokens):
    print(f"   [{i+1}/{NUM_SCENES}] token={token}")
    scene = scene_loader.get_scene_from_token(token)

    # (a) BEV 俯视图 + 模型轨迹 vs 人类轨迹
    fig, ax = plot_bev_with_agent(scene, agent)
    ax.set_title(f"BEV: Agent(red) vs Human(green)\ntoken={token[:20]}...", fontsize=10)
    fig.savefig(OUTPUT_DIR / f"bev_trajectory_{i}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # (b) 多相机视角
    fig, ax = plot_cameras_frame(scene, frame_idx=scene.scene_metadata.num_history_frames - 1)
    fig.savefig(OUTPUT_DIR / f"cameras_{i}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # (c) 纯 BEV 地图 + 标注
    fig, ax = plot_bev_frame(scene, frame_idx=scene.scene_metadata.num_history_frames - 1)
    fig.savefig(OUTPUT_DIR / f"bev_map_{i}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

print(f"\n可视化完成！结果保存在: {OUTPUT_DIR.absolute()}")
print("文件列表:")
for f in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"  {f.name}")
