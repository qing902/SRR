import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.models import Classifier, Temporal_Imputer
# from models.models import masking2, mask_adj_matrices_edges, GraphRecover_new
from models.models import masking2, mask_adj_matrices_edges

from models.loss import CrossEntropyLabelSmooth, EntropyLoss
from scipy.spatial.distance import cdist
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
from visualize_wavelet import visualize_trg_and_rec
import torch
import torch.nn.functional as F


@torch.no_grad()
def bn_moment_transfer_1d(x_t, mu_s, var_s, eps=1e-5):
    """
    x_t: (B, C, L) 目标域原始时序
    mu_s, var_s: (C,) 预训练阶段保存的源域通道统计
    return: x_c 作为“源风格粗样本”
    """
    # 目标 batch 的通道统计
    # 维度中除 channel(1) 外都参与求均值/方差
    reduce_dims = tuple(i for i in range(x_t.dim()) if i != 1)
    mu_t = x_t.mean(dim=reduce_dims, keepdim=True)  # (1,C,1)
    var_t = x_t.var(dim=reduce_dims, unbiased=False, keepdim=True)

    mu_s = mu_s.view(1, -1, 1).to(x_t.device)
    var_s = var_s.view(1, -1, 1).to(x_t.device)

    x_norm = (x_t - mu_t) / torch.sqrt(var_t + eps)
    x_c = x_norm * torch.sqrt(var_s + eps) + mu_s
    return x_c

class SSC1D(nn.Module):
    """
    Source Statistics Calibration for 1D temporal features
    """
    def __init__(self, num_channels, mu_s, var_s):
        super().__init__()
        self.register_buffer("mu_s", mu_s.view(1, -1, 1))
        self.register_buffer("var_s", var_s.view(1, -1, 1))

        # γ = σ(MLP([mean, var]))
        self.gamma_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        # z: (B, C, L)
        mu_t = z.mean(dim=(0, 2), keepdim=True)
        var_t = z.var(dim=(0, 2), unbiased=False, keepdim=True)

        norm_t = (z - mu_t) / torch.sqrt(var_t + 1e-5)
        norm_s = (z - self.mu_s) / torch.sqrt(self.var_s + 1e-5)

        stat = torch.cat([
            mu_t.mean().unsqueeze(0),
            var_t.mean().unsqueeze(0)
        ], dim=0).unsqueeze(0)

        gamma = self.gamma_mlp(stat).view(1, 1, 1)

        z_calib = gamma * norm_t + (1 - gamma) * norm_s
        return z_calib, gamma

def mutual_fft_mix_1d(x_t, x_c, beta):
    """
    1D 版 Mutual Fourier Transform (向量化版本)
    x_t, x_c: (B, C, L)
    beta: 低频带宽比例 (0,1]，也可传入张量 per-batch/per-channel
    return: x_g: (B, C, L)
    """
    B, C, L = x_t.shape
    # rFFT: (..., L//2+1) 复数
    X_t = torch.fft.rfft(x_t, dim=-1)  # (B,C,F)
    X_c = torch.fft.rfft(x_c, dim=-1)  # (B,C,F)
    Fbins = X_t.shape[-1]  # Fbins = L//2 + 1

    # 幅度与相位
    At, Ac = torch.abs(X_t), torch.abs(X_c)
    Pt = torch.angle(X_t)

    # 低频掩膜 M_beta
    k0 = int((Fbins - 1) * beta)  # beta 相对于 Nyquist 频
    k0 = max(1, min(k0, Fbins - 1))
    M = torch.zeros((1, 1, Fbins), device=x_t.device, dtype=At.dtype)
    M[..., :k0 + 1] = 1.0  # [0, k0] 为低频（包含 DC）

    # 幅度混合（低频来自 x_c，高频来自 x_t），相位用 x_t
    A_mix = M * Ac + (1 - M) * At
    Xg = A_mix * torch.exp(1j * Pt)

    # irFFT 回到时域
    x_g = torch.fft.irfft(Xg, n=L, dim=-1)
    return x_g


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.cross_entropy = nn.CrossEntropyLoss()

    def update(self, *args, **kwargs):
        raise NotImplementedError


class SRR(Algorithm):
    """
    SRR: Our proposed method using temporal restoration and spatial rewiring adaptation.
    """

    def __init__(self, backbone, configs, hparams, device):
        super(SRR, self).__init__(configs)
        # backbone:
        self.feature_extractor = backbone(configs)
        # classifier:
        self.classifier = Classifier(configs)
        # entire network:
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # temporal imputation.
        self.temporal_verifier = Temporal_Imputer(configs)
        # graph recover:

        # self.graph_recover = GraphRecover_new(configs)

        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams['pre_learning_rate'],
            weight_decay=hparams['weight_decay']
        )

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams['learning_rate'],
            weight_decay=hparams['weight_decay']
        )

        # self.recover_optimizer = torch.optim.Adam([
        #     {'params': self.temporal_verifier.parameters(), 'lr': hparams['learning_rate'], 'weight_decay': hparams['weight_decay']},
        #
        #     # {'params': self.graph_recover.parameters()}
        #
        # ])
        self.recover_optimizer = torch.optim.Adam(
            self.temporal_verifier.parameters(),
            lr=hparams['learning_rate'],
            weight_decay=hparams['weight_decay']
        )

        self.hparams = hparams
        self.device = device
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1)
        self.mse_loss = nn.MSELoss()
        self._register_bn_hooks()

    def _register_bn_hooks(self):
        """ 在每个 BN 层挂一个 hook，保存输入特征用于计算 batch-level mean/var """
        for name, module in self.feature_extractor.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                module.input_cache = None

                def hook(m, inp, out):
                    m.input_cache = inp[0].detach()  # (N, C, L)

                module.register_forward_hook(hook)

    def pretrain(self, src_dataloader, avg_meter, logger):
        # 用于保存源域输入的统计信息
        total_mu, total_var, total_count = 0, 0, 0

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                self.pre_optimizer.zero_grad()
                # self.recover_optimizer.zero_grad()

                ### raw data ###
                # src_temp_feat = self.feature_extractor.temporal_cnn(src_x)
                # src_adj = self.feature_extractor.graph_learner(src_temp_feat)
                # src_feat, src_flat = self.feature_extractor.spatial_gnn(src_temp_feat, src_adj)
                src_temp_feat, src_flat = self.feature_extractor(src_x)

                ### masked data ###
                masked_src_x, _ = masking2(src_x, num_splits=8, num_masked=1)
                masked_temp_feat = self.feature_extractor.temporal_cnn(masked_src_x)
                # masked_feat, masked_flat = self.feature_extractor(masked_src_x)

                ### temporal restoration ###
                src_recovered_temp_feat = self.temporal_verifier(masked_temp_feat.detach())
                # tov_loss = self.mse_loss(src_recovered_temp_feat, src_feat)
                min_len = min(src_recovered_temp_feat.shape[-1], src_temp_feat.shape[-1])
                tov_loss = self.mse_loss(
                    src_recovered_temp_feat[..., :min_len],
                    src_temp_feat[..., :min_len]
                )

                # === classifier predictions ===
                src_pred = self.classifier(src_flat)
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                # === 统计源域 BN 信息（输入通道级） ===
                reduce_dims = tuple(i for i in range(src_x.dim()) if i != 1)
                mu_batch = src_x.mean(dim=reduce_dims)  # (C,)
                var_batch = src_x.var(dim=reduce_dims, unbiased=False)  # (C,)

                total_mu += mu_batch.detach().cpu()
                total_var += var_batch.detach().cpu()
                total_count += 1

                total_loss = src_cls_loss + tov_loss
                total_loss.backward()
                self.pre_optimizer.step()
                # self.recover_optimizer.step()

                losses = {
                    'cls_loss': src_cls_loss.detach().item(),
                    'tov_loss': tov_loss.detach().item()
                }
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        # 保存源域 BN 统计
        self.source_bn_stats = {
            "mu": (total_mu / total_count),
            "var": (total_var / total_count)
        }

        src_only_model = deepcopy(self.network.state_dict())
        return src_only_model

    # 导入可视化函数


    def update(self, trg_dataloader, avg_meter, logger):
        """
        Target-domain adaptation:
        - 不再使用图结构和 GNN
        - 冻结 Temporal Imputer（只作为 teacher）
        - 更新 feature_extractor (+ 可选 classifier)
        - 使用：tov_loss + BN 正则 + FSM 一致性 + 熵正则/IM
        """
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()

        # ====== 冻结 Temporal Imputer；Class      ier 是否冻结看你策略 ======
        for p in self.temporal_verifier.parameters():
            p.requires_grad = False

        # 如果你也想冻结分类头，打开下面两行：
        # for p in self.classifier.parameters():
        #     p.requires_grad = False

        mu_s = self.source_bn_stats["mu"].to(self.device)  # [C]
        var_s = self.source_bn_stats["var"].to(self.device)  # [C]
        # ====== Lazy initialization of SSC ======
        if not hasattr(self, "ssc"):
            self.ssc = SSC1D(
                num_channels=mu_s.numel(),
                mu_s=mu_s,
                var_s=var_s
            ).to(self.device)

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (trg_x, _, _) in enumerate(trg_dataloader):
                trg_x = trg_x.float().to(self.device)

                self.optimizer.zero_grad()

                # ====== 1. 原始目标域特征 & 预测 ======
                trg_temp_feat, trg_flat = self.feature_extractor(trg_x)  # trg_temp_feat: [B, C, L_t], trg_flat: [B, D]
                trg_logits = self.classifier(trg_flat)
                trg_prob = F.softmax(trg_logits, dim=1)

                # ====== 2. 时序 Mask + 重构（使用冻结的 Temporal Imputer） ======
                masked_trg_x, _ = masking2(trg_x, num_splits=self.configs.num_splits, num_masked=1)
                masked_temp_feat, _ = self.feature_extractor(masked_trg_x)

                with torch.no_grad():
                    # Temporal Imputer 不参与梯度
                    rec_temp_feat = self.temporal_verifier(masked_temp_feat)

                # 对齐时间长度
                min_len = min(rec_temp_feat.shape[-1], trg_temp_feat.shape[-1])
                rec_align = rec_temp_feat[..., :min_len]
                trg_align = trg_temp_feat[..., :min_len]

                tov_loss = self.mse_loss(rec_align, trg_align)

                # ====== 3. BN 正则：对齐重构特征的统计与源域统计 ======
                rec_calib, gamma = self.ssc(rec_align)
                ssc_reg = torch.mean(gamma * (1 - gamma))

                # ====== 4. FSM：频域风格混合 + 一致性约束 ======
                B, C, Lx = trg_x.shape
                Lr = rec_align.shape[-1]
                if Lr >= Lx:
                    rec_for_fft = rec_align[..., :Lx]
                else:
                    repeat_factor = (Lx + Lr - 1) // Lr
                    rec_for_fft = rec_align.repeat(1, 1, repeat_factor)[..., :Lx]

                # 1D FFT 沿时间维度
                Ft = torch.fft.rfft(trg_x, dim=-1)
                rec_for_fft = rec_calib[..., :Lx] if rec_calib.shape[-1] >= Lx else \
                    rec_calib.repeat(1, 1, repeat_factor)[..., :Lx]

                Fr = torch.fft.rfft(rec_for_fft, dim=-1)

                # 低频部分用重构的幅度，高频保持原幅度（简单版 FSM）
                mask_ratio = 0.1
                k = max(1, int(Ft.shape[-1] * mask_ratio))

                At, Pt = torch.abs(Ft), torch.angle(Ft)
                Ar = torch.abs(Fr)

                Ag = At.clone()
                Ag[..., :k] = Ar[..., :k]  # 低频幅度来自重构（源式风格）

                # 使用原始相位 Pt，幅度 Ag 重建
                Fg = Ag * torch.exp(1j * Pt)
                x_g = torch.fft.irfft(Fg, n=Lx, dim=-1)

                # 通过同一特征提取和分类头得到预测
                g_temp_feat, g_flat = self.feature_extractor(x_g)
                g_logits = self.classifier(g_flat)
                g_prob = F.softmax(g_logits, dim=1)

                # 一致性损失：约束 x_t 与 x_g 的预测接近
                fsm_loss = F.mse_loss(g_prob, trg_prob.detach())

                # ====== 5. 熵最小化 + 信息最大化 ======
                ent_loss = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(trg_prob))
                im_loss = - self.hparams['im'] * torch.sum(
                    -trg_prob.mean(dim=0) * torch.log(trg_prob.mean(dim=0) + 1e-5)
                )
                ent_term = ent_loss + im_loss

                # ====== 6. 总损失 ======
                loss = (
                        self.hparams['tov_wt'] * tov_loss +
                        self.hparams.get('fsm_loss_wt', 0.1) * fsm_loss +
                        self.hparams.get('ssc_reg_wt', 0.01) * ssc_reg +
                        ent_term
                )

                loss.backward()
                self.optimizer.step()

                # ====== 7. 记录日志 ======
                losses = {
                    'tov_loss': tov_loss.detach().item(),
                    'bn_loss': ssc_reg.detach().item(),
                    'fsm_loss': fsm_loss.detach().item(),
                    'entropy_term': ent_term.detach().item()
                }
                for key, val in losses.items():
                    avg_meter[key].update(val, trg_x.size(0))

                # ====== 可视化：只在第1个epoch第1个batch保存一次 ======
                if epoch == 1 and step == 0:
                    B, C, Lx = trg_x.shape
                    Lr = rec_align.shape[-1]

                    if Lx >= Lr:
                        trg_vis = trg_x[..., :Lr]
                    else:
                        repeat_factor = (Lr + Lx - 1) // Lx
                        trg_vis = trg_x.repeat(1, 1, repeat_factor)[..., :Lr]

                    # 调用可视化函数
                    visualize_trg_and_rec(
                        trg_x=trg_vis,
                        rec_align=rec_align,
                        save_dir="./wavelet_vis",
                        batch_idx=0,
                        channel_idx=0,
                        wavelet='db1'
                    )

            self.lr_scheduler.step()

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return last_model, best_model

    # def update(self, trg_dataloader, avg_meter, logger):
    #     """
    #     Target-domain adaptation:
    #     - 不再使用图结构和 GNN
    #     - 冻结 Temporal Imputer（只作为 teacher）
    #     - 更新 feature_extractor (+ 可选 classifier)
    #     - 使用：tov_loss + BN 正则 + FSM 一致性 + 熵正则/IM
    #     """
    #     best_src_risk = float('inf')
    #     best_model = self.network.state_dict()
    #     last_model = self.network.state_dict()
    #
    #     # ====== 冻结 Temporal Imputer；Class      ier 是否冻结看你策略 ======
    #     for p in self.temporal_verifier.parameters():
    #         p.requires_grad = False
    #
    #     # 如果你也想冻结分类头，打开下面两行：
    #     # for p in self.classifier.parameters():
    #     #     p.requires_grad = False
    #
    #     mu_s = self.source_bn_stats["mu"].to(self.device)  # [C]
    #     var_s = self.source_bn_stats["var"].to(self.device)  # [C]
    #     # ====== Lazy initialization of SSC ======
    #     if not hasattr(self, "ssc"):
    #         self.ssc = SSC1D(
    #             num_channels=mu_s.numel(),
    #             mu_s=mu_s,
    #             var_s=var_s
    #         ).to(self.device)
    #
    #     for epoch in range(1, self.hparams["num_epochs"] + 1):
    #         for step, (trg_x, _, _) in enumerate(trg_dataloader):
    #             trg_x = trg_x.float().to(self.device)
    #
    #             self.optimizer.zero_grad()
    #
    #             # ====== 1. 原始目标域特征 & 预测 ======
    #             trg_temp_feat, trg_flat = self.feature_extractor(trg_x)  # trg_temp_feat: [B, C, L_t], trg_flat: [B, D]
    #             trg_logits = self.classifier(trg_flat)
    #             trg_prob = F.softmax(trg_logits, dim=1)
    #
    #             # ====== 2. 时序 Mask + 重构（使用冻结的 Temporal Imputer） ======
    #             masked_trg_x, _ = masking2(trg_x, num_splits=self.configs.num_splits, num_masked=1)
    #             masked_temp_feat, _ = self.feature_extractor(masked_trg_x)
    #
    #             with torch.no_grad():
    #                 # Temporal Imputer 不参与梯度
    #                 rec_temp_feat = self.temporal_verifier(masked_temp_feat)
    #
    #             # 对齐时间长度
    #             min_len = min(rec_temp_feat.shape[-1], trg_temp_feat.shape[-1])
    #             rec_align = rec_temp_feat[..., :min_len]
    #             trg_align = trg_temp_feat[..., :min_len]
    #
    #             tov_loss = self.mse_loss(rec_align, trg_align)
    #
    #             # ====== 3. BN 正则：对齐重构特征的统计与源域统计 ======
    #             # 对 rec_align 做通道统计: (batch, C, L) -> (C,)
    #             # reduce_dims = (0, 2)
    #
    #             rec_calib, gamma = self.ssc(rec_align)
    #             ssc_reg = torch.mean(gamma * (1 - gamma))
    #
    #             # ====== 4. FSM：频域风格混合 + 一致性约束 ======
    #             # 使用原始目标信号 trg_x 与重构特征 rec_align 构造 x_g
    #             # 这里简单地把 rec_align 截断/插值到与 trg_x 相同长度再做 FFT
    #             B, C, Lx = trg_x.shape
    #             Lr = rec_align.shape[-1]
    #             if Lr >= Lx:
    #                 rec_for_fft = rec_align[..., :Lx]
    #             else:
    #                 # 如果重构长度更短，简单 repeat/ pad（这里用 repeat 简化）
    #                 repeat_factor = (Lx + Lr - 1) // Lr
    #                 rec_for_fft = rec_align.repeat(1, 1, repeat_factor)[..., :Lx]
    #
    #             # 1D FFT 沿时间维度
    #             Ft = torch.fft.rfft(trg_x, dim=-1)
    #             # 用校准后的序列作为低频参考
    #             rec_for_fft = rec_calib[..., :Lx] if rec_calib.shape[-1] >= Lx else \
    #                 rec_calib.repeat(1, 1, repeat_factor)[..., :Lx]
    #
    #             Fr = torch.fft.rfft(rec_for_fft, dim=-1)
    #
    #             # 低频部分用重构的幅度，高频保持原幅度（简单版 FSM）
    #             # mask_ratio 表示有多少比例频率视作“风格/低频”
    #             mask_ratio = 0.1
    #             k = max(1, int(Ft.shape[-1] * mask_ratio))
    #
    #             At, Pt = torch.abs(Ft), torch.angle(Ft)
    #             Ar = torch.abs(Fr)
    #
    #             Ag = At.clone()
    #             Ag[..., :k] = Ar[..., :k]  # 低频幅度来自重构（源式风格）
    #
    #             # 使用原始相位 Pt，幅度 Ag 重建
    #             Fg = Ag * torch.exp(1j * Pt)
    #             x_g = torch.fft.irfft(Fg, n=Lx, dim=-1)
    #
    #
    #             # 通过同一特征提取和分类头得到预测
    #             g_temp_feat, g_flat = self.feature_extractor(x_g)
    #             g_logits = self.classifier(g_flat)
    #             g_prob = F.softmax(g_logits, dim=1)
    #
    #             # 一致性损失：约束 x_t 与 x_g 的预测接近
    #             fsm_loss = F.mse_loss(g_prob, trg_prob.detach())
    #
    #             # ====== 5. 熵最小化 + 信息最大化 ======
    #             ent_loss = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(trg_prob))
    #             im_loss = - self.hparams['im'] * torch.sum(
    #                 -trg_prob.mean(dim=0) * torch.log(trg_prob.mean(dim=0) + 1e-5)
    #             )
    #             ent_term = ent_loss + im_loss
    #
    #             # ====== 6. 总损失 ======
    #             loss = (
    #                     self.hparams['tov_wt'] * tov_loss +
    #                     self.hparams.get('fsm_loss_wt', 0.1) * fsm_loss +
    #                     self.hparams.get('ssc_reg_wt', 0.01) * ssc_reg +
    #                     ent_term
    #             )
    #
    #             loss.backward()
    #             self.optimizer.step()
    #
    #             # ====== 7. 记录日志 ======
    #             losses = {
    #                 'tov_loss': tov_loss.detach().item(),
    #                 'bn_loss': ssc_reg.detach().item(),
    #                 'fsm_loss': fsm_loss.detach().item(),
    #                 'entropy_term': ent_term.detach().item()
    #             }
    #             for key, val in losses.items():
    #                 avg_meter[key].update(val, trg_x.size(0))
    #
    #         self.lr_scheduler.step()
    #
    #         logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
    #         for key, val in avg_meter.items():
    #             logger.debug(f'{key}\t: {val.avg:2.4f}')
    #         logger.debug(f'-------------------------------------')
    #
    #      return last_model, best_model
    #


