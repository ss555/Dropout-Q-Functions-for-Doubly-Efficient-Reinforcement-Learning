
FISH_MOVING_CONFIG={'num_steps': 100000,
               'batch_size': 128,
               'lr': 0.0003,
               'scheduler': 'cosine',
               'hidden_units': [256, 256],
               'memory_size': 5000.0,
               'gamma': 0.99,
               'tau': 0.005,
               'entropy_tuning': True,
               'ent_coef': 0.0876,
               'multi_step': 1,
               'per': 0,
               'alpha': 0.6,
               'beta': 0.4,
               'beta_annealing': 3e-07,
               'grad_clip': None,
               'critic_updates_per_step': 20,
               'gradients_step': 768,
                'eval_episodes_interval': 10,
               'start_steps': 0,
               'log_interval': 10,
               'target_update_interval': 1,
               'cuda': 0,
               'seed': 0,
               'eval_runs': 3,
               'huber': 0,
               'layer_norm': 1,
               'target_entropy': -1.0,
               'method': 'sac',
               'target_drop_rate': 0.005,
               'save_model_interval': 10,
               'critic_update_delay': 1}

FISH_STATIONARY_CONFIG={'num_steps': 100000,
               'batch_size': 128,
               'lr': 0.0001,
               'scheduler': 'cosine',
               'hidden_units': [256, 256],
               'memory_size': 5000.0,
               'gamma': 0.99,
               'tau': 0.005,
               'entropy_tuning': True,
               'ent_coef': 0.0876,
               'multi_step': 1,
               'per': 0,
               'alpha': 0.6,
               'beta': 0.4,
               'beta_annealing': 3e-07,
               'grad_clip': None,
               'critic_updates_per_step': 20,
               'gradients_step': 768,
                'eval_episodes_interval': 10,
               'start_steps': 0,
               'log_interval': 10,
               'target_update_interval': 1,
               'cuda': 0,
               'seed': 0,
               'eval_runs': 3,
               'huber': 0,
               'layer_norm': 1,
               'target_entropy': -1.0,
               'method': 'sac',
               'target_drop_rate': 0.005,
               'save_model_interval': 10,
               'critic_update_delay': 1}

FISH_STATIONARY_CONFIG2 = {'num_steps': 200000,
                          'batch_size': 256,
                          'lr': 0.0003,
                          'scheduler': 'cosine',
                          'hidden_units': [256, 256],
                          'memory_size': 10000.0,
           'gamma': 0.98,
           'tau': 0.005,
           'entropy_tuning': True,
           'ent_coef': 0.0876,
           'multi_step': 1,
           'per': 0,
           'alpha': 0.6,
           'beta': 0.4,
           'beta_annealing': 3e-07,
           'grad_clip': None,
           'critic_updates_per_step': 20,  # 20,
           'gradients_step': 768,  # 20,
           'eval_episodes_interval': 10,
           'start_steps': 0,
           'log_interval': 10,
           'target_update_interval': 1,
           'cuda': 0,
           'seed': 0,
           'eval_runs': 3,
           'huber': 0,
           'layer_norm': 1,
           'target_entropy': -1.0,
           'method': 'sac',
           'target_drop_rate': 0.01,  # 0.005,
           'save_model_interval': 10,
           'critic_update_delay': 1}