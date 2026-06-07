def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class EEG_EDF_ORI():
    def __init__(self):
        super(EEG_EDF_ORI, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 100,
            'lr_decay': 0.5
        }
        self.alg_hparams = {
            'SRR': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'ent_loss_wt': 0.8621, 'im': 0.8461, 'tov_wt': 0.7,'bn_loss_wt': 1,
                      'fsm_enable': True,
                      'fsm_beta_min': 0.10,  # 低频下界 (按经验 0.1~0.3)
                      'fsm_beta_max': 0.25,  # 低频上界
                      'fsm_weight': 0.2,  # FSM 生成样本的一致性损失权重
                      # 'fsm_loss': 'consistency',  # 或 'entropy_on_g'
                      'fsm_consistency_type': 'kl'}
                # , 'graph_recover_wt': 0.4, 'gmask_ratio': 0.5}
        }

class HAR():
    def __init__(self):
        super(HAR, self).__init__()
        self.train_params = {
            'num_epochs': 100,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 100,
            'lr_decay': 0.5
        }
        self.alg_hparams = {
            'SRR': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'tov_wt': 0.9, 'ent_loss_wt': 0.4085, 'im': 0.8837,'bn_loss_wt': 1,
                      'fsm_enable': True,
                      'fsm_beta_min': 0.10,  # 低频下界 (按经验 0.1~0.3)
                      'fsm_beta_max': 0.25,  # 低频上界
                      'fsm_weight': 0.2,  # FSM 生成样本的一致性损失权重
                      # 'fsm_enable': True,
                      # 'fsm_type': 'dwt',
                      # 'fsm_dwt_level': 2,  # ⭐ 小波分解层数（1~4 常用）
                      # 'fsm_wavelet': 'db4',
                      # 'fsm_loss_wt': 0.2,

                      # 'fsm_loss': 'consistency',  # 或 'entropy_on_g'
                      'fsm_consistency_type': 'kmse'

                      }
                # , 'graph_recover_wt': 0.5, 'gmask_ratio': 0.5}
        }

class WISDM():
    def __init__(self):
        super(WISDM, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 100,
            'lr_decay': 0.5
        }
        self.alg_hparams = {
            'SRR': {'pre_learning_rate': 0.003, 'learning_rate': 0.0003, 'tov_wt': 0.7, 'ent_loss_wt': 0.8528, 'im': 0.589,'bn_loss_wt': 1,
                      'fsm_enable': True,
                      'fsm_beta_min': 0.15,  # 低频下界 (按经验 0.1~0.3)
                      'fsm_beta_max': 0.23,  # 低频上界
                      'fsm_weight': 0.1,  # FSM 生成样本的一致性损失权重
                      # 'fsm_loss': 'consistency',  # 或 'entropy_on_g'
                      'fsm_consistency_type': 'kmse'}
                # , 'graph_recover_wt': 0.6, 'gmask_ratio': 0.5}
        }
class FD():
    def __init__(self):
        super(FD, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5
        }
        self.alg_hparams = {
            'SRR': {'pre_learning_rate': 0.001, 'learning_rate': 0.000001, 'tov_wt': 0.8, 'im': 0.2983,  'ent_loss_wt': 0.8467,'bn_loss_wt': 1,
                      # 'fsm_enable': True,
                      # 'fsm_beta_min': 0.10,  # 低频下界 (按经验 0.1~0.3)
                      # 'fsm_beta_max': 0.25,  # 低频上界
                      # 'fsm_weight': 0.1,  # FSM 生成样本的一致性损失权重
                      # 'fsm_loss': 'consistency',  # 或 'entropy_on_g'
                      'fsm_enable': True,
                      'fsm_type': 'dwt',
                      'fsm_dwt_level': 2,  # ⭐ 小波分解层数（1~4 常用）
                      'fsm_wavelet': 'db4',
                      'fsm_loss_wt': 0.2,
                      'fsm_consistency_type': 'kmse'}
            # , 'graph_recover_wt': 0.6, 'gmask_ratio': 0.5}

        }

class HHAR():
    def __init__(self):
        super(HHAR, self).__init__()
        self.train_params = {
            'num_epochs': 100,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5
        }
        self.alg_hparams = {
            'SRR': {'pre_learning_rate': 0.001, 'learning_rate': 0.000001, 'tov_wt': 0.8, 'im': 0.2983,
                      'ent_loss_wt': 0.8467, 'bn_loss_wt': 1,
                      'fsm_enable': True,
                      'fsm_beta_min': 0.10,  # 低频下界 (按经验 0.1~0.3)
                      'fsm_beta_max': 0.25,  # 低频上界
                      # 'fsm_weight': 0.1,  # FSM 生成样本的一致性损失权重
                      # 'fsm_loss': 'consistency',  # 或 'entropy_on_g'
                      # 'fsm_enable': True,
                      # 'fsm_type': 'dwt',
                      # 'fsm_dwt_level': 2,  # ⭐ 小波分解层数（1~4 常用）
                      # 'fsm_wavelet': 'db4',
                      'fsm_loss_wt': 0.2,
                      'fsm_consistency_type': 'kmse'},
            # , 'graph_recover_wt': 0.6, 'gmask_ratio': 0.5}


            'SHOT': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,
                     'target_cls_wt': 0.05,  'beta': 10, 'alpha': 1},
            'TPDS': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,
                     'target_cls_wt': 0.05, 'aad_wt':0.1, 'beta': 10, 'alpha': 1},
            'GKD': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,
                     'target_cls_wt': 0.05,  'beta': 10, 'alpha': 1},
            'CESFDA': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,
                     'target_cls_wt': 0.01,  'beta': 10, 'alpha': 1, 'aad_wt':0.025},

            'AaD': {'pre_learning_rate': 0.003, 'learning_rate': 0.0001, 'beta': 10, 'alpha': 1},
            'SCLM': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,
                     'target_cls_wt': 0.05,  'beta': 10, 'alpha': 1},
            'NRC': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'epsilon': 1e-5},
            'MAPU': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5},
        }


