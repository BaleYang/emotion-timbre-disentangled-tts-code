import os
import json
import copy
import math
from collections import OrderedDict
import torch.nn.init as init
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from text.symbols import symbols
from utils.tools import get_mask_from_lengths, pad

class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)

        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]
        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            energy_min, energy_max = stats["energy"][:2]

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["decoder_hidden"]
        )
        self.energy_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["decoder_hidden"]
        )

        self.use_target_pitch = model_config.get('use_target_pitch', True)
        self.use_target_energy = model_config.get('use_target_energy', True)

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None and self.use_target_pitch:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))

        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )

        return prediction, embedding
    
    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None and self.use_target_energy:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))

        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )

        return prediction, embedding

    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):


        log_duration_prediction = self.duration_predictor(x, src_mask)
        if self.pitch_feature_level == "phoneme_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, src_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, src_mask, p_control
            )
            x = x + energy_embedding

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)


        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, mel_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, mel_mask, p_control
            )
            x = x + energy_embedding

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )

class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()
        # if model_config.get('model_name', None) is not None:
        #     self.input_size = model_config["transformer"]["encoder_hidden"] + model_config['emotion_predictor']['emtion_embedding_table']['style_hidden']
        # else:
        #     self.input_size = model_config["transformer"]["encoder_hidden"]
        self.input_size = model_config["transformer"]["decoder_hidden"]

        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x




class PEPA(nn.Module):
    """PEPA

    Lightweight convolutional extractor that refines encoded phoneme features
    before cross-modal attention with the reference mel.
    """
    def __init__(self, model_config):
        super().__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["emotion_predictor"]['phoneme_extractor']["filter_size"]
        self.kernel = model_config["emotion_predictor"]['phoneme_extractor']["kernel_size"]
        self.dropout = model_config["emotion_predictor"]['phoneme_extractor']["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = out.masked_fill(mask.unsqueeze(-1), 0.0)

        return out




class ReferenceEncoderGST(nn.Module):
    """Reference Encoder for GST-based timbre extraction.

    Inputs:  [B, T, n_mels]
    Outputs: [B, ref_enc_gru_size]
    """

    def __init__(self, model_config):

        super().__init__()
        K = len(model_config['emotion_predictor']['reference_encoder']['conv_filters'])
        filters = [1] + model_config['emotion_predictor']['reference_encoder']['conv_filters']
        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=model_config['emotion_predictor']['reference_encoder']['conv_filters'][i]) for i in range(K)])
        self.n_mels = model_config['n_mels']
        out_channels = self.calculate_channels(self.n_mels, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=model_config['emotion_predictor']['reference_encoder']['conv_filters'][-1] * out_channels,
                          hidden_size=model_config['emotion_predictor']['reference_encoder']['gru_hidden'],
                          batch_first=True)

    def forward(self, inputs):
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, self.n_mels)  # [B, 1, T, n_mels]
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)

        out = out.transpose(1, 2)  # [B, T', C, M']
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)

        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # out: [1, B, E]

        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L



class StyleTokens(nn.Module):
    """Style Tokens (GST) module.

    Encodes a reference mel into a style embedding using style token attention.
    """

    def __init__(self, model_config):

        super().__init__()
        self.encoder = ReferenceEncoderGST(model_config)
        self.stl = StyleTokenLayer(model_config)

    def forward(self, inputs):
        enc_out = self.encoder(inputs)
        style_embed = self.stl(enc_out)

        return style_embed




class StyleTokenLayer(nn.Module):
    """Style token attention layer.

    Inputs:  [B, E]
    Outputs: [B, 1, style_hidden]
    """

    def __init__(self, model_config):

        super().__init__()
        self.num_emotion_vectors = model_config['emotion_predictor']['emtion_embedding_table']['n_style_token']
        self.style_hidden = model_config['emotion_predictor']['emtion_embedding_table']['style_hidden']

        self.embed = nn.Parameter(torch.FloatTensor(self.num_emotion_vectors, self.style_hidden))
        init.normal_(self.embed, mean=0, std=0.02)
        
        d_q = model_config['emotion_predictor']['reference_encoder']['gru_hidden']
        d_k = self.style_hidden
        self.attention = MultiHeadAttention(query_dim=d_q, key_dim=d_k, num_units=self.style_hidden, num_heads=model_config['emotion_predictor']['attn_head'])


    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)  # [B, 1, E]
        keys = F.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)
        style_embed = self.attention(query, keys)

        return style_embed



class MultiHeadAttention(nn.Module):
    """Standard multi-head attention used across the style modules.

    query: [B, T_q, query_dim]
    key  : [B, T_k, key_dim]
    return: [B, T_q, num_units]
    """

    def __init__(self, query_dim, key_dim, num_units, num_heads):

        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim
        
        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key, key_mask=None):

        querys = self.W_query(query)  # [B, T_q, num_units]
        keys = self.W_key(key)  # [B, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, B, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, B, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, B, T_k, num_units/h]

        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, B, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)

        if key_mask is not None:
            key_mask = key_mask.unsqueeze(0).unsqueeze(2)  # [1, 1, B, T_k]
            key_mask = key_mask.expand(self.num_heads, -1, scores.shape[2], -1)
            scores = scores.masked_fill(key_mask == True, float('-inf'))

        scores = F.softmax(scores, dim=3)
        out = torch.matmul(scores, values)  # [h, B, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)

        return out
    



class TimbreExtractor(nn.Module):
    """Timbre Extractor using GST-style tokens and a linear classifier."""
    def __init__(self, model_config, num_class):
        super(TimbreExtractor, self).__init__()
        style_hidden = model_config['emotion_predictor']['emtion_embedding_table']['style_hidden']
        self.gst = StyleTokens(model_config)
        self.classifier = TimbreClassifier(num_class, style_hidden)

    def forward(self, reference_mel):
        style_embed = self.gst(reference_mel)
        pred_id = self.classifier(style_embed)

        return style_embed, pred_id
    




class EmotionExtractor(nn.Module):
    """Emotion Extractor: PEPA + ReferenceEncoder + Multi-head attentions + style tokens."""
    def __init__(self, model_config, num_class):
        super(EmotionExtractor, self).__init__()
        self.style_hidden = model_config['emotion_predictor']['emtion_embedding_table']['style_hidden']

        K = len(model_config['emotion_predictor']['reference_encoder']['conv_filters1'])
        self.referenceEncoder_out_dim = 128 * model_config['n_mels'] // (2 ** K)

        self.pepa = PEPA(model_config)
        
        self.classifier = EmotionClassifier(num_class, self.style_hidden)

        self.linear_score = nn.Linear(model_config['transformer']['encoder_hidden'], model_config['emotion_predictor']['emtion_embedding_table']['n_style_token'])
        self.referenceEncoder = ReferenceEncoder(model_config)
        self.referenceEncoder_out_dim = self.referenceEncoder.out_dim
        self.linear_re = nn.Sequential(
            nn.Linear(self.referenceEncoder_out_dim, self.style_hidden),
            Mish(),
        )

        self.attention = MultiHeadAttention(model_config["emotion_predictor"]['phoneme_extractor']["filter_size"], 
                                             self.style_hidden, 
                                             model_config["emotion_predictor"]['attn_dim'], 
                                             model_config["emotion_predictor"]['attn_head']
                                             )
        self.attention2 = MultiHeadAttention(model_config["emotion_predictor"]['attn_dim'], 
                                             model_config['emotion_predictor']['emtion_embedding_table']['style_hidden'], 
                                             model_config['emotion_predictor']['emtion_embedding_table']['style_hidden'], 
                                             model_config["emotion_predictor"]['attn_head']
                                             )
        self.num_emotion_vectors = model_config['emotion_predictor']['emtion_embedding_table']['n_style_token']
        self.style_hidden = model_config['emotion_predictor']['emtion_embedding_table']['style_hidden']

        self.embed = nn.Parameter(torch.FloatTensor(self.num_emotion_vectors, self.style_hidden))
        init.normal_(self.embed, mean=0, std=0.02)

    def forward(self, encoder_output, reference_mel, src_masks=None, mel_mask=None):
        N = encoder_output.size(0)

        query = self.pepa(encoder_output, src_masks)
        key = self.referenceEncoder(reference_mel)
        key = self.linear_re(key)

        hidden_state = self.attention(query, key)
        hidden_state = hidden_state.masked_fill(src_masks.unsqueeze(-1), 0)
        
        keys = F.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)
        out = self.attention2(hidden_state, keys)

        out = out.masked_fill(src_masks.unsqueeze(-1), 0)

        pred_id = self.classifier(out)

        return out, pred_id
    




class EmotionClassifier(nn.Module):
    """Simple pooling classifier for emotion categories."""
    def __init__(self, num_class, input_dim):
        super(EmotionClassifier, self).__init__()

        dim_in = input_dim
        
        self.linear1 = nn.Linear(dim_in, num_class)
    
    def forward(self, inputs):
        out = inputs.mean(dim=1)

        out = out.squeeze(1)

        out = self.linear1(out)

        return out
    

class TimbreClassifier(nn.Module):
    """Linear classifier applied to a single style embedding frame."""
    def __init__(self, num_class, input_dim):
        super(TimbreClassifier, self).__init__()

        dim_in = input_dim
        
        self.linear1 = nn.Linear(dim_in, num_class)
    
    def forward(self, inputs):

        out = self.linear1(inputs)
        out = out.squeeze(1)
        return out





class ReferenceEncoder(nn.Module):
    """CNN stack as a reference feature encoder for emotion extractor.

    Inputs:  [B, T, n_mels]
    Outputs: [B, T', C'] (flattened later)
    """

    def __init__(self, model_config):
        super().__init__()
        self.n_mel = model_config['n_mels']
        K = len(model_config['emotion_predictor']['reference_encoder']['conv_filters1'])
        K1 = len(model_config['emotion_predictor']['reference_encoder']['conv_filters2'])
        filters_t = model_config['emotion_predictor']['reference_encoder']['conv_filters1'] + model_config['emotion_predictor']['reference_encoder']['conv_filters2']
        filters = [1] + model_config['emotion_predictor']['reference_encoder']['conv_filters1']
        filters1 = [filters[-1]] + model_config['emotion_predictor']['reference_encoder']['conv_filters2']

        self.dropout = nn.Dropout(p=model_config['emotion_predictor']['reference_encoder'].get('dropout', 0))

        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        convs1 = [nn.Conv2d(in_channels=filters1[i],
                           out_channels=filters1[i + 1],
                           kernel_size=(3, 3),
                           stride=(1, 1),
                           padding=(1, 1)) for i in range(K1)]
        self.convs = nn.ModuleList(convs+convs1)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=filters_t[i]) for i in range(K+K1)])
        self.out_channels = self.calculate_channels(model_config['n_mels'], 3, 2, 1, K)
        self.out_dim = 128 * self.out_channels

    def forward(self, inputs):
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, self.n_mel)  # [B, 1, T, n_mels]
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)
            out = self.dropout(out)
        
        out = out.transpose(1, 2)
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)

        return out

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L





class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))





class SelfAttentivePoolingLayer(nn.Module):
    """Self-Attentive Pooling with optional local attention window.

    query: [B, T_q, query_dim]
    key  : [B, T_k, key_dim]
    return: [B, T_q, num_units]
    """

    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim
        
        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key   = nn.Linear(in_features=key_dim,  out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim,  out_features=num_units, bias=False)

    def forward(self, 
                query, 
                key, 
                key_mask=None, 
                local_window_size=9):
        # query: [B, T_q, query_dim]
        # key  : [B, T_k, key_dim]
        N, T_q, _ = query.shape
        _, T_k, _ = key.shape

        # Project
        querys = self.W_query(query)
        keys   = self.W_key(key)
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)
        keys   = torch.stack(torch.split(keys,   split_size, dim=2), dim=0)
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)

        scores = torch.matmul(querys, keys.transpose(-2, -1))  # [h, B, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)

        if key_mask is not None:
            key_mask = key_mask.unsqueeze(1).unsqueeze(0)  # [1, 1, B, T_k]
            key_mask = key_mask.expand(self.num_heads, -1, T_q, -1)
            scores = scores.masked_fill(key_mask, float('-inf'))

        if local_window_size is not None:
            assert T_q == T_k, "Local self-attention requires T_q == T_k"
            local_mask = self.generate_local_attention_mask(T_k, local_window_size)
            local_mask = local_mask.unsqueeze(0).unsqueeze(1)
            local_mask = local_mask.expand(self.num_heads, N, T_q, T_k).to(scores.device)
            scores = scores.masked_fill(~local_mask, -1e15)

        scores = F.softmax(scores, dim=-1)
        out = torch.matmul(scores, values)
        out = torch.cat(torch.split(out, 1, dim=0), dim=-1).squeeze(0)

        return out

    def generate_local_attention_mask(self, seq_len, window_size):
        assert window_size % 2 == 1, "window_size must be odd"
        radius = (window_size - 1) // 2

        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        for i in range(seq_len):
            left = max(0, i - radius)
            right = min(seq_len, i + radius + 1)
            mask[i, left:right] = True
        return mask

class StyleEncoder(nn.Module):
    """Style Encoder that fuses Timbre and Emotion features with Add & Norm.

    - Timbre Extractor consumes reference mel and outputs a 1-step style vector.
    - Emotion Extractor consumes encoded phoneme and reference mel to produce
      sequence-aligned emotion features.
    - A self-attentive pooling layer refines the emotion features.
    - The final output adds both projected features to the phoneme encoding
      followed by LayerNorm (Add & Norm).
    """
    def __init__(self, model_config, num_speaker, num_emotion):
        super().__init__()

        self.emotion_extractor = EmotionExtractor(model_config, num_emotion)
        self.timbre_extractor = TimbreExtractor(model_config, num_speaker)

        style_hidden = model_config['emotion_predictor']['emtion_embedding_table']['style_hidden']
        encoder_hidden = model_config['transformer']['encoder_hidden']

        self.pool = SelfAttentivePoolingLayer(style_hidden, style_hidden, style_hidden, model_config['emotion_predictor']['attn_head'])

        # Projections to align feature dims with encoder hidden size
        self.proj_emotion = nn.Linear(style_hidden, encoder_hidden)
        self.proj_timbre = nn.Linear(style_hidden, encoder_hidden)

        self.layer_norm = nn.LayerNorm(encoder_hidden)
        self.local_window_size = 9

    def forward(self, encoder_output, reference_mel, src_masks=None, mel_masks=None):
        # Extract emotion features (sequence) and timbre features (global)
        emotion_feats, pred_emotion = self.emotion_extractor(encoder_output, reference_mel, src_masks, mel_masks)
        timbre_feat, pred_timbre = self.timbre_extractor(reference_mel)

        # Self-attentive pooling over emotion features
        pooled_emotion = self.pool(emotion_feats, emotion_feats, key_mask=src_masks, local_window_size=self.local_window_size)
        pooled_emotion = pooled_emotion.masked_fill(src_masks.unsqueeze(-1), 0)

        # Broadcast timbre across time
        B, L, _ = encoder_output.shape
        timbre_broadcast = timbre_feat.squeeze(1).unsqueeze(1).repeat(1, L, 1)

        # Project to encoder hidden size and Add & Norm
        emo_proj = self.proj_emotion(pooled_emotion)
        timbre_proj = self.proj_timbre(timbre_broadcast)
        fused = self.layer_norm(encoder_output + emo_proj + timbre_proj)

        return fused, pred_timbre, pred_emotion, timbre_feat, emotion_feats



        