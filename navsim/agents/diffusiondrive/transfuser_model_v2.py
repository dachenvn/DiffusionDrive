from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import copy
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_backbone import TransfuserBackbone
from navsim.agents.diffusiondrive.transfuser_features import BoundingBox2DIndex
from navsim.common.enums import StateSE2Index
from diffusers.schedulers import DDIMScheduler
from navsim.agents.diffusiondrive.modules.conditional_unet1d import ConditionalUnet1D,SinusoidalPosEmb
import torch.nn.functional as F
from navsim.agents.diffusiondrive.modules.blocks import linear_relu_ln,bias_init_with_prob, gen_sineembed_for_position, GridSampleCrossBEVAttention
from navsim.agents.diffusiondrive.modules.multimodal_loss import LossComputer
from torch.nn import TransformerDecoder,TransformerDecoderLayer
from typing import Any, List, Dict, Optional, Union
class V2TransfuserModel(nn.Module):
    """Torch module for Transfuser."""

    def __init__(self, config: TransfuserConfig):
        """
        Initializes TransFuser torch module.
        :param config: global config dataclass of TransFuser.
        """

        super().__init__()

        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]

        self._config = config
        self._backbone = TransfuserBackbone(config)   #多模态骨干网络：融合摄像头 / Lidar 特征，输出 BEV（鸟瞰图）特征
        
        # 为 Transformer 提供位置 / 类型嵌入（8x8 BEV 网格 + 轨迹 /agent 查询）
        self._keyval_embedding = nn.Embedding(8**2 + 1, config.tf_d_model)  # 8x8 feature grid + trajectory
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        # usually, the BEV features are variable in size.  特征变换：BEV 特征降维（512→tf_d_model）、自车状态编码（4+2+2→tf_d_model）
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

        # 语义分割头：输出 BEV 语义地图（车道 / 障碍物等）
        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                mode="bilinear",
                align_corners=False,
            ),
        )

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        # Transformer 解码器：基于_keyval(BEV+状态编码)，输出轨迹 / agent 查询
        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        
        # 预测头，输出障碍物预测位置
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        # 预测头，输出自车未来行驶轨迹
        self._trajectory_head = TrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,     # 需要预测的位姿 (x,y,θ) 的数量
            d_ffn=config.tf_d_ffn,                              # 前馈网络的维度
            d_model=config.tf_d_model,                          # 输入维度
            plan_anchor_path=config.plan_anchor_path,           # 计划锚点路径
            config=config,
        )

        # BEV特征投影：对融合后的 BEV 特征做线性变换 + 维度调整
        self.bev_proj = nn.Sequential(
            *linear_relu_ln(256, 1, 1,320),
        )

    # 关键代码
    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        # 1. 输入特征提取：接收 TransfuserAgent 传入的多模态特征，明确输入张量的含义。
        camera_feature: torch.Tensor = features["camera_feature"]   # 摄像头编码特征（如 [B, 256, H, W]）
        lidar_feature: torch.Tensor = features["lidar_feature"]     # Lidar 编码特征（如 [B, 128, H, W]）
        status_feature: torch.Tensor = features["status_feature"]   # 自车状态（4+2+2：速度/位置/偏航角等）
        batch_size = status_feature.shape[0]                        # 获取批次大小（后续维度调整用）

        # 2. 骨干网络输出BEV特征
        # bev_feature_upscale：高分辨率 BEV 特征（用于语义分割）
        # bev_feature：低分辨率 BEV 特征（用于 Transformer 融合）
        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
        cross_bev_feature = bev_feature_upscale
        bev_spatial_shape = bev_feature_upscale.shape[2:]
        concat_cross_bev_shape = bev_feature.shape[2:]
        
        # 3. BEV 特征降维 + 自车状态编码
        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)   #通过一个线性层进行编码

        # 4. 构建 Transformer KeyVal（键值对）
        # KeyVal = BEV 网格特征 + 自车状态特征 + 嵌入，是 Transformer 解码器的 “输入上下文”
        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        # 5. BEV 特征插值 / 拼接（跨模态融合）
        # 调整 BEV 特征维度，插值到高分辨率
        concat_cross_bev = keyval[:,:-1].permute(0,2,1).contiguous().view(batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1])

        concat_cross_bev = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode='bilinear', align_corners=False)

        # 拼接跨模态 BEV 特征
        cross_bev_feature = torch.cat([concat_cross_bev, cross_bev_feature], dim=1)
        
        # BEV 特征投影（线性变换）
        cross_bev_feature = self.bev_proj(cross_bev_feature.flatten(-2,-1).permute(0,2,1))
        cross_bev_feature = cross_bev_feature.permute(0,2,1).contiguous().view(batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1])
        
        # 6. Transformer解码器
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)     # [1, N, tf_d_model] → [B, N, tf_d_model]                               
        query_out = self._tf_decoder(query, keyval)     # Transformer解码：Query（轨迹/agent）→ KeyVal（BEV+状态）

        # 7. 输出预测结果（语义地图 + 轨迹 + Agent）
        # BEV 语义地图
        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)   # 语义分割头：输出 BEV 语义地图（车道 / 障碍物等）
        
        # 轨迹预测（拆分 Query，调用轨迹头）
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}

        trajectory = self._trajectory_head(trajectory_query,agents_query, cross_bev_feature,bev_spatial_shape,status_encoding[:, None],targets=targets,global_img=None)
        output.update(trajectory)

        # Agent（障碍物）预测（调用 Agent 头）
        agents = self._agent_head(agents_query)
        output.update(agents)

        return output

class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}

class DiffMotionPlanningRefinementModule(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        ego_fut_ts=8,
        ego_fut_mode=20,
        if_zeroinit_reg=True,
    ):
        super(DiffMotionPlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            nn.Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 3),
        )
        self.if_zeroinit_reg = False

        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_reg:
            nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
            nn.init.constant_(self.plan_reg_branch[-1].bias, 0)

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)
    def forward(
        self,
        traj_feature,
    ):
        bs, ego_fut_mode, _ = traj_feature.shape

        # 6. get final prediction
        traj_feature = traj_feature.view(bs, ego_fut_mode,-1)   # 将轨迹特征 reshape 成 [bs, n_anchor, d_model]
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)   # 通过一个线性层进行分类
        traj_delta = self.plan_reg_branch(traj_feature)   # 通过一个线性层进行回归
        plan_reg = traj_delta.reshape(bs,ego_fut_mode, self.ego_fut_ts, 3)   # 将回归结果 reshape 成 [bs, n_anchor, ego_fut_ts, 3]

        return plan_reg, plan_cls
class ModulationLayer(nn.Module):

    def __init__(self, embed_dims: int, condition_dims: int):
        super(ModulationLayer, self).__init__()
        self.if_zeroinit_scale=False
        self.embed_dims = embed_dims
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dims, embed_dims*2),
        )
        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_scale:
            nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
            nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)

    def forward(
        self,
        traj_feature,
        time_embed,
        global_cond=None,
        global_img=None,
    ):
        if global_cond is not None:
            global_feature = torch.cat([
                    global_cond, time_embed
                ], axis=-1)
        else:
            global_feature = time_embed
        if global_img is not None:
            global_img = global_img.flatten(2,3).permute(0,2,1).contiguous()
            global_feature = torch.cat([
                    global_img, global_feature
                ], axis=-1)
        
        scale_shift = self.scale_shift_mlp(global_feature)
        scale,shift = scale_shift.chunk(2,dim=-1)
        traj_feature = traj_feature * (1 + scale) + shift   # 加入了时间步的偏移，来融合时间步信息
        return traj_feature

class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, 
                 num_poses,
                 d_model,
                 d_ffn,
                 config,
                 ):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.cross_bev_attention = GridSampleCrossBEVAttention(     # 交叉注意力：输入的轨迹与BEV交互
            config.tf_d_model,
            config.tf_num_head,
            num_points=num_poses,
            config=config,
            in_bev_dims=256,
        )
        self.cross_agent_attention = nn.MultiheadAttention(     # 多头注意力：输入的轨迹与障碍物交互
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.cross_ego_attention = nn.MultiheadAttention(        # 多头注意力：输入的轨迹与自车交互
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )
        self.norm1 = nn.LayerNorm(config.tf_d_model)
        self.norm2 = nn.LayerNorm(config.tf_d_model)
        self.norm3 = nn.LayerNorm(config.tf_d_model)
        self.time_modulation = ModulationLayer(config.tf_d_model,256)   # 时间步融合模块
        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=config.tf_d_model,
            ego_fut_ts=num_poses,
            ego_fut_mode=20,
        )

    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                time_embed, 
                status_encoding,
                global_img=None):
        traj_feature = self.cross_bev_attention(traj_feature,noisy_traj_points,bev_feature,bev_spatial_shape)   #输入轨迹与 BEV 特征交互
        traj_feature = traj_feature + self.dropout(self.cross_agent_attention(traj_feature, agents_query,agents_query)[0])   #输入轨迹与障碍物交互，再加 残差链接
        traj_feature = self.norm1(traj_feature)  #再做归一化
        
        # traj_feature = traj_feature + self.dropout(self.self_attn(traj_feature, traj_feature, traj_feature)[0])

        # 4.5 cross attention with  ego query
        traj_feature = traj_feature + self.dropout1(self.cross_ego_attention(traj_feature, ego_query,ego_query)[0])   #输入轨迹与自车交互，再加 残差链接
        traj_feature = self.norm2(traj_feature)  #再做归一化
        
        # 4.6 feedforward network
        traj_feature = self.norm3(self.ffn(traj_feature))  #再经过一个前馈神经网络FFN，再做归一化
        # 4.8 modulate with time steps
        traj_feature = self.time_modulation(traj_feature, time_embed,global_cond=None,global_img=global_img)  #融合了时间步信息
        
        # 4.9 predict the offset & heading
        poses_reg, poses_cls = self.task_decoder(traj_feature) #bs,20,8,3; bs,20   正式生成预测轨迹的结果，包含位置和heading
        poses_reg[...,:2] = poses_reg[...,:2] + noisy_traj_points
        poses_reg[..., StateSE2Index.HEADING] = poses_reg[..., StateSE2Index.HEADING].tanh() * np.pi

        return poses_reg, poses_cls
def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class CustomTransformerDecoder(nn.Module):
    def __init__(
        self, 
        decoder_layer, 
        num_layers,
        norm=None,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
    
    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                time_embed, 
                status_encoding,
                global_img=None):
        poses_reg_list = []
        poses_cls_list = []
        traj_points = noisy_traj_points
        for mod in self.layers:
            poses_reg, poses_cls = mod(traj_feature, traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)
            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)
            traj_points = poses_reg[...,:2].clone().detach()
        return poses_reg_list, poses_cls_list

# 轨迹生成网络
class TrajectoryHead(nn.Module):
    """Trajectory prediction head."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int, plan_anchor_path: str,config: TransfuserConfig):
        """
        Initializes trajectory head.
        :param num_poses: number of (x,y,θ) poses to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(TrajectoryHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self.diff_loss_weight = 2.0
        self.ego_fut_mode = 20

        self.diffusion_scheduler = DDIMScheduler(      # 反向去噪过程
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )


        plan_anchor = np.load(plan_anchor_path)     # 从先验数据中聚类中的计划锚点anchor

        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        ) # 20,8,2
        self.plan_anchor_encoder = nn.Sequential(   # 对anchor进行编码
            *linear_relu_ln(d_model, 1, 1,512),
            nn.Linear(d_model, d_model),
        )
        self.time_mlp = nn.Sequential(              # 对时间步进行编码
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )

        diff_decoder_layer = CustomTransformerDecoderLayer(     # diffusion decoder
            num_poses=num_poses,
            d_model=d_model,
            d_ffn=d_ffn,
            config=config,
        )
        self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, 2)  # 2层 decoder layer 解码

        self.loss_computer = LossComputer(config)
    def norm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = 2*(odo_info_fut_x + 1.2)/56.9 -1
        odo_info_fut_y = 2*(odo_info_fut_y + 20)/46 -1
        odo_info_fut_head = 2*(odo_info_fut_head + 2)/3.9 -1
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)
    def denorm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = (odo_info_fut_x + 1)/2 * 56.9 - 1.2
        odo_info_fut_y = (odo_info_fut_y + 1)/2 * 46 - 20
        odo_info_fut_head = (odo_info_fut_head + 1)/2 * 3.9 - 2
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)
    def forward(self, ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding, targets=None,global_img=None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        if self.training:
            return self.forward_train(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,targets,global_img)
        else:
            return self.forward_test(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,global_img)

    # 训练时 forward 函数
    def forward_train(self, ego_query,agents_query,bev_feature,bev_spatial_shape,status_encoding, targets=None,global_img=None) -> Dict[str, torch.Tensor]:
        bs = ego_query.shape[0]   # 获取批次大小
        device = ego_query.device   # 获取训练的机器
        # 1. add truncated noise to the plan anchor
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs,1,1,1)  # 扩维
        odo_info_fut = self.norm_odo(plan_anchor)   # 加噪前先归一化
        timesteps = torch.randint(
            0, 50,
            (bs,), device=device
        )

        # Diffusion 加噪过程
        noise = torch.randn(odo_info_fut.shape, device=device)  # 生成随机噪声
        noisy_traj_points = self.diffusion_scheduler.add_noise(    
            original_samples=odo_info_fut,
            noise=noise,
            timesteps=timesteps,
        ).float()
        noisy_traj_points = torch.clamp(noisy_traj_points, min=-1, max=1)
        noisy_traj_points = self.denorm_odo(noisy_traj_points)   # 加噪后再逆归一化，由于对归一化样本加噪效果比较好

        ego_fut_mode = noisy_traj_points.shape[1]   # anchor数量
        # 2. proj noisy_traj_points to the query
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64)   # 将噪声位姿转换为位置嵌入，将二维坐标转换为向量，用于后续的注意力机制   
        traj_pos_embed = traj_pos_embed.flatten(-2)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed)   # 将位置嵌入转换为特征
        traj_feature = traj_feature.view(bs,ego_fut_mode,-1)      # 维度 [bs, n_anchor, d_model]
        # 3. embed the timesteps  时间编码
        time_embed = self.time_mlp(timesteps)   # 将时间步转换为特征编码，向量
        time_embed = time_embed.view(bs,1,-1)   # 维度 [bs, 1, d_model]


        # 4. begin the stacked decoder  Diffusion Decoder 解码过程
        poses_reg_list, poses_cls_list = self.diff_decoder(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)

        # 5. compute the loss  计算loss
        trajectory_loss_dict = {}
        ret_traj_loss = 0
        for idx, (poses_reg, poses_cls) in enumerate(zip(poses_reg_list, poses_cls_list)):
            trajectory_loss = self.loss_computer(poses_reg, poses_cls, targets, plan_anchor)
            trajectory_loss_dict[f"trajectory_loss_{idx}"] = trajectory_loss
            ret_traj_loss += trajectory_loss

        mode_idx = poses_cls_list[-1].argmax(dim=-1)
        mode_idx = mode_idx[...,None,None,None].repeat(1,1,self._num_poses,3)
        best_reg = torch.gather(poses_reg_list[-1], 1, mode_idx).squeeze(1)
        return {"trajectory": best_reg,"trajectory_loss":ret_traj_loss,"trajectory_loss_dict":trajectory_loss_dict}

    # 测试时 forward 函数
    def forward_test(self, ego_query,agents_query,bev_feature,bev_spatial_shape,status_encoding,global_img) -> Dict[str, torch.Tensor]:
        step_num = 2
        bs = ego_query.shape[0]
        device = ego_query.device
        self.diffusion_scheduler.set_timesteps(1000, device)
        step_ratio = 20 / step_num
        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)   # [0, 10]
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)


        # 1. add truncated noise to the plan anchor  给plan anchor 添加截断噪声
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs,1,1,1)   # 扩维
        img = self.norm_odo(plan_anchor)   # 加噪前归一化
        noise = torch.randn(img.shape, device=device)   # 生成随机噪声
        trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8   # 截断时间步
        img = self.diffusion_scheduler.add_noise(original_samples=img, noise=noise, timesteps=trunc_timesteps)   # 截断时间步加噪，img实际是加了噪声的轨迹
        noisy_trajs = self.denorm_odo(img)   # 加噪后逆归一化，由于对归一化样本加噪效果比较好
        ego_fut_mode = img.shape[1]
        for k in roll_timesteps[:]:  # [0, 10] 两步
            x_boxes = torch.clamp(img, min=-1, max=1)   # 将轨迹限制在[-1,1]范围内
            noisy_traj_points = self.denorm_odo(x_boxes)   # 逆归一化

            # 2. proj noisy_traj_points to the query
            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64)   # 生成位置编码
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(bs,ego_fut_mode,-1)

            timesteps = k
            if not torch.is_tensor(timesteps):    # 判断是 CPU 还是 GPU
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=img.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(img.device)
            
            # 3. embed the timesteps   时间步编码
            timesteps = timesteps.expand(img.shape[0])
            time_embed = self.time_mlp(timesteps)
            time_embed = time_embed.view(bs,1,-1)

            # 4. begin the stacked decoder  Diffusion Decoder 解码过程
            # 输入：轨迹特征、噪声轨迹点、BEV特征、BEV空间形状、障碍物查询、自车查询、时间步编码、状态编码、全局图像
            # 输出：预测轨迹、预测分类
            poses_reg_list, poses_cls_list = self.diff_decoder(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)
            poses_reg = poses_reg_list[-1]     # 当前 timestamp 下的预测轨迹
            poses_cls = poses_cls_list[-1]     # 当前 timestamp 下的预测分类
            x_start = poses_reg[...,:2]   # 当前 timestamp 下的预测轨迹
            x_start = self.norm_odo(x_start)   # 归一化当前 timestamp 下的预测轨迹
            img = self.diffusion_scheduler.step(     # 当前 timestamp下，根据预测轨迹sample，得到预测噪声 noisy
                model_output=x_start,
                timestep=k,
                sample=img
            ).prev_sample
        mode_idx = poses_cls.argmax(dim=-1)   # 找到预测分类中概率最大的索引
        mode_idx = mode_idx[...,None,None,None].repeat(1,1,self._num_poses,3)
        best_reg = torch.gather(poses_reg, 1, mode_idx).squeeze(1)   # 根据索引找到对应的预测轨迹
        return {"trajectory": best_reg}
