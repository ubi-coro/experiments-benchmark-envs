import os
from typing import Dict, List, Optional, Union
import torch

from hydra.conf import HydraConf, RunDir, SweepDir, JobConf
from hydra_zen import make_custom_builds_fn, zen, store

from stable_baselines3 import SAC
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, StopTrainingOnMaxEpisodes
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import CombinedExtractor
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3_extensions import HerReplayBufferExt
from stable_baselines3_extensions.common.callbacks import EvalCallbackExt
from stable_baselines3_extensions.common.lr_schedules import linear_schedule
from stable_baselines3_extensions.common.utils import save_system_info

from experiments_benchmark_envs.callbacks import SavePathExistsCallback
import experiments_benchmark_envs.env_configs as env_configs

# builds: set customized default values 
builds = make_custom_builds_fn(populate_full_signature=True)
builds_sac = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

# learning rate config
def learning_rate(initial_value: float = 1e-3, use_linear_lr_schedule: bool = True):
    if not use_linear_lr_schedule:
        return initial_value
    else:
        return linear_schedule(initial_value=initial_value)
learning_rate_conf = builds(learning_rate)

# generate configs
def replay_buffer_kwargs(
    n_sampled_goal: int = 4, 
    goal_selection_strategy: Union[GoalSelectionStrategy, str] = 'future', 
    copy_info_dict: bool = True
):
    return {'n_sampled_goal': n_sampled_goal, 
            'goal_selection_strategy': goal_selection_strategy, 
            'copy_info_dict': copy_info_dict}
replay_buffer_conf = builds(replay_buffer_kwargs)

def policy_kwargs(
    share_features_extractor: bool = False, 
    net_arch:  Optional[Union[List[int], Dict[str, List[int]]]] = [128,256,64], 
    n_critics: int = 2
):
    return {'share_features_extractor': share_features_extractor, 
            'features_extractor_class': CombinedExtractor,
            'net_arch': list(net_arch), 
            'n_critics': n_critics}
policy_conf = builds(policy_kwargs)

sac_conf = builds_sac(
    SAC,
    policy='MultiInputPolicy',
    learning_rate=learning_rate_conf(),
    buffer_size=int(1e6),
    learning_starts=int(1e4), # how many env steps to collect transitions for before learning starts
    batch_size=256, # minibatch size for each gradient update
    tau=0.05, # soft update coefficient
    gamma=0.95, # discount factor
    train_freq=1, # update the model every train_freq steps
    gradient_steps=-1, # how many gradient steps to do after each rollout 
                       # -> Set to -1 means to do as many gradient steps as steps done in the environment during the rollout 
                       # (in case of -1: depends on number of training envs!)
    ent_coef='auto', # entropy regularization coefficient ('auto': learn it automatically)
    replay_buffer_class=HerReplayBufferExt,
    replay_buffer_kwargs=replay_buffer_conf(),
    verbose=1,
    seed=0
)

# learning and callback config
def learning_cb(total_timesteps: int = int(3e6), max_train_episodes: int = 10000, num_eval_episodes: int = 100):
    # StopTrainingOnMaxEpisodes: stops training after max_train_episodes * num_train episodes regardless of total_timesteps
    return {'total_timesteps': total_timesteps, 'max_train_episodes': max_train_episodes, 'num_eval_episodes': num_eval_episodes}

learning_cb_conf = builds(learning_cb)

@store(name='train', sac=sac_conf, policy_conf=policy_conf, env_conf=env_configs.planning_bs_conf_3, learning_cb_conf=learning_cb_conf)
def train_func(sac: SAC, policy_conf: dict, env_conf: dict, learning_cb_conf: dict):
    # paths
    save_path = os.getcwd()
    log_path = os.path.join(save_path, 'logs')    
    eval_path = os.path.join(save_path, 'evaluation')
    if os.path.exists(log_path):
        raise ValueError(f"\nsave path already exists. Current log path is: {save_path}")
    
    # log system info 
    save_system_info(save_path=log_path)
    
    # make vec environments
    vec_env_cls = DummyVecEnv
    monitor_kwargs = {'info_keywords': ("is_success",), 'override_existing': True}

    train_envs = make_vec_env(
        env_id=env_conf['env_id'],
        n_envs=env_conf['num_train'],
        seed = 0,
        start_index=0,
        monitor_dir=log_path,
        wrapper_class=None,
        env_kwargs=env_conf['env_kwargs'],
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=None,
        monitor_kwargs=monitor_kwargs,
        wrapper_kwargs=None
    )

    eval_env = make_vec_env(
        env_id=env_conf['env_id'],
        n_envs=1,
        seed=env_conf['num_train'],
        start_index=0,
        monitor_dir=eval_path,
        wrapper_class=None,
        env_kwargs=env_conf['env_kwargs'],
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=None,
        monitor_kwargs=monitor_kwargs,
        wrapper_kwargs=None
    )
    
    # sac model
    model = sac(env=train_envs, 
                policy_kwargs=policy_conf,
                tensorboard_log=log_path, 
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # logger
    logger = configure(log_path, ['stdout', 'csv', 'tensorboard'])
    model.set_logger(logger)
    
    # callbacks
    stop_train_cb = StopTrainingOnMaxEpisodes(max_episodes=learning_cb_conf['max_train_episodes'], verbose=1)
    eval_cb = EvalCallbackExt(  eval_env, 
                                best_model_save_path=eval_path,
                                log_path=eval_path, 
                                eval_freq=int(5000/env_conf['num_train']), # evaluate model after num_train*eval_freq steps
                                n_eval_episodes=learning_cb_conf['num_eval_episodes'],
                                deterministic=True)
    callbacks = CallbackList([stop_train_cb, eval_cb])

    # learning
    model.learn(total_timesteps = learning_cb_conf['total_timesteps'],
                callback = callbacks,
                log_interval = env_conf['num_train'],
                progress_bar=True)
    
        
if __name__ == "__main__":
    save_path = os.path.join(os.getcwd(), 'data')
    save_dir_name = ('${env_conf.env_id}_NumMovers_${env_conf.num_movers}_MoverSize_${env_conf.mover_params.size.size}_' +
                     'LearnJerk_${env_conf.learn_jerk}_NumCycles_${env_conf.num_cycles}_Vmax_${env_conf.v_max}_' + 
                     'Amax_${env_conf.a_max}_Jmax_${env_conf.j_max}_CollisionShape_${env_conf.collision_params.shape}_' +
                     'CollisionSize_${env_conf.collision_params.size}_CollisionOffset_${env_conf.collision_params.offset}_' +
                     'CollisionOffsetWall_${env_conf.collision_params.offset_wall}_InitialLR_${sac.learning_rate.initial_value}_' +
                     'NetArch_${policy_conf.net_arch}')
    store(HydraConf(run=RunDir(dir=os.path.join(save_path, save_dir_name)), 
                    sweep=SweepDir(dir=save_path, subdir=save_dir_name),
                    job=JobConf(chdir=True),
                    callbacks={'save_path_exists': builds(SavePathExistsCallback)}))
    store.add_to_hydra_store()
    zen(train_func).hydra_main(config_name='train', version_base=None)