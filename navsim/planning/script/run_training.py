#1. 导入Python内置类型/路径/日志模块
from typing import Tuple        # 用于定义函数返回值的类型注解（元组）
from pathlib import Path        # 处理文件路径（跨平台，比os.path更友好）
import logging                  # 日志工具，输出训练过程信息（替代print）


# 2. 导入第三方核心依赖
import hydra                      # 配置管理框架，加载yaml配置文件
from hydra.utils import instantiate  # Hydra的实例化工具，从配置动态创建对象
from omegaconf import DictConfig  # Hydra的配置数据结构（类似字典，支持层级访问）
from torch.utils.data import DataLoader  # PyTorch数据加载器，批量加载训练数据
import pytorch_lightning as pl    # 封装PyTorch训练逻辑（无需手写训练循环）

# 3. 导入NAVSIM框架内部的部分模块
from navsim.agents.abstract_agent import AbstractAgent  # 自动驾驶智能体抽象接口
from navsim.common.dataclasses import SceneFilter       # 场景过滤数据类（筛选训练数据）
from navsim.common.dataloader import SceneLoader        # 场景数据加载器（读取传感器/轨迹）
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset  # 数据集类（常规/仅缓存）
from navsim.planning.training.agent_lightning_module import AgentLightningModule  # 适配PL的智能体模块

# 4. 初始化日志器（命名为当前模块名），用于输出训练日志
logger = logging.getLogger(__name__)

# 5. 定义Hydra配置文件路径和默认配置名
CONFIG_PATH = "config/training"   # 配置文件所在目录（相对路径）
CONFIG_NAME = "default_training" # 默认配置文件名（default_training.yaml 且无需 .yaml 后缀）


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary (Hydra加载的配置)
    :param agent: interface of agents in NAVSIM (自动驾驶智能体接口)
    :return: tuple for training and validation dataset (训练/验证数据集元组)
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data

# @hydra.main() 是 Hydra 提供的装饰器，专门用于标记 “程序入口函数”，它的核心作用是：
# 自动加载指定路径的配置文件（YAML）；
# 将配置解析为 DictConfig 对象，传入被装饰的 main 函数；
# 处理命令行参数覆盖配置（如 python train.py trainer.max_epochs=200）；
# 管理实验目录（自动创建输出文件夹、保存配置快照）。
@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    pl.seed_everything(cfg.seed, workers=True)   #固定所有的随机数种子，确保每次运行结果一致，保证实验可复现
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")          #实际是加载 default_training.yaml，创建 TransfuserAgent模型类
    agent: AbstractAgent = instantiate(cfg.agent)  #实例化 DiffusionDrive 模型对象，包含扩散模型的所有逻辑

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,                               #将 DiffusionDrive 模型封装为pl适配层，用于标准化训练和评估(如 training_step)
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(          #数据集构建，只从缓存中加载数据，不构建 SceneLoader
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)   #数据集构建，构建 SceneLoader 和 Dataset 对象

    logger.info("Building Datasets")    
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True)   #将训练数据集封装为批量加载器，支持多线程加载
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False)      #验证数据
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")      #加载 PL 训练器，加载训练配置，精度/DDP，传入模型自定义回调函数
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())

    logger.info("Starting Training")
    trainer.fit(                         #训练入口！触发PL自动执行训练循环 + 验证循环，最终调用扩散模型进行训练
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()
