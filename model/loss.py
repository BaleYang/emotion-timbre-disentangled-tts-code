import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json

import random

class FastSpeech2LossStage1(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2LossStage1, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, inputs, predictions):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
        ) = inputs[6:12]
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        # pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        # energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            duration_loss,
        )

        

class FastSpeech2LossStage2(nn.Module):
    """ FastSpeech2 Loss with Emotion Tag """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2LossStage2, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"]["feature"]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"]["feature"]
        self.emotion_class_wt = model_config.get('emotion_class_wt', 0.3)
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.nll_loss = nn.CrossEntropyLoss()  # 添加 NLLLoss，用于处理对数概率

    def forward(self, inputs, predictions):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
            emotion_targets,  # 从 inputs 中提取情感标签
            speaker_targets
        ) = inputs[6:14]

        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
            speaker_predictions,
            emotion_predictions, 
        ) = predictions[:12]

        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, :mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        # 确保梯度不回传到目标张量
        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False
        emotion_targets.requires_grad = False  # 添加这一行
        speaker_targets.requires_grad = False  # 添加这一行

        # 处理 pitch 和 energy 的掩码
        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        elif self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        # 计算各项损失
        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)
        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        # 计算情感损失
        # 因为 emotion_predictions 已经经过了 Softmax，需要取对数
        emotion_loss = self.emotion_class_wt * self.nll_loss(emotion_predictions, emotion_targets)
        speaker_loss = self.emotion_class_wt * self.nll_loss(speaker_predictions, speaker_targets)

        # 计算总损失
        total_loss = (
            mel_loss
            + postnet_mel_loss
            + duration_loss
            + pitch_loss
            + energy_loss
            + emotion_loss  # 添加情感损失
            + speaker_loss
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
            emotion_loss,  # 返回情感损失
            speaker_loss
        )
    

class  MILosslinear(nn.Module):

    def __init__(self, preprocess_config, model_config):
        super(MILosslinear, self).__init__()
        self.loss_wt = model_config['mist_wt']

    def forward(self, reference_out, encoder_out, mine_net, src_masks=None, ma_et=None):
        margin_encoder_out, shuffled_src_masks = self.shuffle_batch_dim(encoder_out, src_masks)
        
        # encoder_out_one = self.random_non_padding_vector(encoder_out, src_masks)
        # margin_encoder_out_one = self.random_non_padding_vector(margin_encoder_out, shuffled_src_masks)

        encoder_out_one = self.average_pooling(encoder_out, src_masks)
        margin_encoder_out_one = self.average_pooling(margin_encoder_out, shuffled_src_masks)
        return self.mutual_information(reference_out, encoder_out_one, margin_encoder_out_one, mine_net, ma_et)


    def shuffle_batch_dim(self, encoder_output, src_masks):
        """
        Shuffle the first dimension (batch dimension) of a given tensor.

        Args:
            tensor (torch.Tensor): Input tensor with shape [B, L, D].

        Returns:
            torch.Tensor: Tensor with the first dimension shuffled.
        """

        # Get the batch size (B)
 
        B = encoder_output.size(0)

        # Generate a random permutation of indices for the batch dimension
        shuffled_indices = torch.randperm(B)

        # Apply the permutation to shuffle the batch dimension
        shuffled_tensor = encoder_output[shuffled_indices]

        if src_masks is None:
            shuffled_src_masks = None
        else:
            shuffled_src_masks = src_masks[shuffled_indices]

        return shuffled_tensor, shuffled_src_masks

    def mutual_information(self, reference, joint_encoder, marginal_encoder, mine_net, ma_et=None):
        T_joint = mine_net(reference, joint_encoder)
        T_marginal = mine_net(reference, marginal_encoder)

        # 对输出做 clamp
        T_joint = torch.clamp(T_joint, min=-10.0, max=100.0)
        T_marginal = torch.clamp(T_marginal, min=-10.0, max=100.0)

        t = T_joint
        et = torch.exp(T_marginal)
        # t = mine_net(reference, joint_encoder)
        # et = torch.exp(mine_net(reference, marginal_encoder))
        # print()
        # print('torch.mean(et)=',torch.mean(et))
        if ma_et is not None:
            mi_lb, ma_et = self.move_average(t, et, ma_et)
            return self.loss_wt * mi_lb, ma_et
        else:
            mi_lb = torch.mean(t) - torch.log(torch.mean(et) + 1e-9)
            return self.loss_wt * mi_lb, torch.mean(et)

    def move_average(self, t, et, ma_et, ma_rate=0.5):
        ma_et = (1-ma_rate)*ma_et + ma_rate*torch.mean(et)

        loss = (torch.mean(t) - (1/(ma_et.mean() + 1e-9)).detach()*torch.mean(et))
        
        return loss, ma_et

    
    def random_non_padding_vector(self, encoder_out, src_masks):
        """
        随机从 encoder_out 的每个 batch 中选择一个非 padding 的向量。
        
        参数:
            encoder_out (torch.Tensor): 维度为 [B, L, D] 的张量。
            src_masks (torch.Tensor): 维度为 [B, L] 的布尔张量，True 表示 padding，False 表示非 padding。
        
        返回:
            torch.Tensor: 维度为 [B, D] 的张量，每个 batch 一个随机选出的非 padding 向量。
        """
        B, L, D = encoder_out.shape
        selected_vectors = []

        for b in range(B):
            # 获取当前 batch 中非 padding 的索引
            non_padding_indices = torch.where(~src_masks[b])[0]  # 找到 False 的索引
            if len(non_padding_indices) == 0:
                raise ValueError(f"Batch {b} 全部为 padding，没有可选值。")
            
            # 从非 padding 的索引中随机选一个
            random_index = non_padding_indices[torch.randint(len(non_padding_indices), (1,)).item()]
            
            # 提取对应的向量
            selected_vectors.append(encoder_out[b, random_index])
        
        # 将选出的向量堆叠为 [B, D]
        return torch.stack(selected_vectors, dim=0)

    def average_pooling(self, encoder_out, src_masks):
        """
        将 encoder_out 的序列维度 (L) 做基于非 padding 的平均池化，得到 [B, D]。

        参数:
            encoder_out (torch.Tensor): 形状 [B, L, D] 的张量。
            src_masks (torch.Tensor):   形状 [B, L] 的布尔张量，True 表示 padding，False 表示非 padding。

        返回:
            torch.Tensor: 形状 [B, D] 的张量，每个 batch 的非 padding 向量做平均后得到的结果。
        """
        # B: batch_size, L: sequence_length, D: hidden_dim
        B, L, D = encoder_out.shape

        # 统计每个 batch 中非 padding 的数量 (形状 [B, 1])
        # src_masks: True 表示 padding，False 表示非 padding
        # 所以我们要对 ~src_masks 做 sum
        num_non_padding = (~src_masks).sum(dim=1, keepdim=True)  # [B, 1]

        # 将布尔掩码转成浮点数 (True->1.0, False->0.0)，并在最后加一维方便做广播
        # 这里我们想保留非 padding 的位置，所以要对 (~src_masks) 做 float()
        non_padding_mask = (~src_masks).unsqueeze(-1).float()    # [B, L, 1]

        # 将 padding 位置置 0，其他位置保留 encoder_out 的值
        masked_encoder_out = encoder_out * non_padding_mask      # [B, L, D]

        # 在序列维度 L 上求和
        sum_pooled = masked_encoder_out.sum(dim=1)               # [B, D]

        # 用非 padding 的数量做平均
        average_pooled = sum_pooled / (num_non_padding + 1e-9)   # [B, D]

        return average_pooled