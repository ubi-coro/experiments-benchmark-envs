import numpy as np
from typing import Callable
from hydra_zen import make_custom_builds_fn, zen, store

from gymnasium_planar_robotics.envs.manipulation.benchmark_pushing_env import BenchmarkPushingEnv
from gymnasium_planar_robotics.envs.planning.benchmark_planning_env import BenchmarkPlanningEnv

# builds: set customized default values 
builds = make_custom_builds_fn(populate_full_signature=True)

# mover params for different mover types
def mover_size(size: str = 'medium'):
    if size == 'medium':
        return np.array([0.155 / 2, 0.155 / 2, 0.012 / 2])
    elif size == 'small':
        return np.array([0.113 / 2, 0.113 / 2, 0.012/ 2])
    else:
        raise ValueError('Unknown mover size')

mover_size_conf = builds(mover_size)

mover_params_small = {'size': mover_size_conf(size='small'), 'mass': 0.63}
mover_params_medium = {'size': mover_size_conf(size='medium'), 'mass': 1.24}

mover_store = store(group='mover')
mover_store(mover_params_small, name='small')
mover_store(mover_params_medium, name='medium')

# collision params for different mover types
def collision_params(shape: str = 'circle', size: float = 0.11, offset: float = 0.0, offset_wall: float = 0.0):
    if shape == 'circle':
        size_cs = size
    elif shape == 'box':
        size_cs = np.array([size, size])
    else:
        raise ValueError('Unknown collision shape')
    
    return {'shape': shape, 'size': size_cs, 'offset': offset, 'offset_wall': offset_wall}

collision_conf = builds(collision_params)

c_params_circle_small = collision_conf(shape='circle', size=0.11, offset=0.0, offset_wall=0.0)
c_params_circle_medium = collision_conf(shape='circle', size=0.14, offset=0.0, offset_wall=0.0)

c_params_box_small = collision_conf(shape='box', size=0.09, offset=0.0, offset_wall=0.0)
c_params_box_medium = collision_conf(shape='box', size=0.11, offset=0.0, offset_wall=0.0)

c_store = store(group='collision')
c_store(c_params_circle_small, name='circle_small')
c_store(c_params_circle_medium, name='circle_medium')

c_store(c_params_box_small, name='box_small')
c_store(c_params_box_medium, name='box_medium')

# general env config
def env(
    env_id: str,
    mover_params: dict, 
    collision_params: Callable[[str,float,float,float], dict], 
    render_mode: str | None = None,
    num_movers: int = 1, 
    num_cycles: int = 40,
    v_max: float = 2.0,
    a_max: float = 10.0,
    j_max: float = 100.0,
    learn_jerk: bool = False,
    threshold_pos: float = 0.05,
    num_train: int = 50
):
    env_kwargs = {'mover_params': mover_params, 
                  'collision_params': collision_params, 
                  'render_mode': render_mode,
                  'num_cycles': num_cycles,
                  'v_max': v_max,
                  'a_max': a_max,
                  'j_max': j_max,
                  'learn_jerk': learn_jerk,
                  'threshold_pos': threshold_pos}
    if env_id == 'BenchmarkPlanningEnv':
        env_kwargs.update({'layout_tiles': np.ones((4,3)), 'num_movers': num_movers, 'show_2D_plot': False})
    
    return {'env_id': env_id, 'env_kwargs': env_kwargs, 'num_train': num_train}

env_conf = builds(env)

# pushing 
pushing_cs_conf = env_conf(env_id='BenchmarkPushingEnv', mover_params=mover_params_small, collision_params=c_params_circle_small)
pushing_cm_conf = env_conf(env_id='BenchmarkPushingEnv', mover_params=mover_params_medium, collision_params=c_params_circle_medium)

pushing_bs_conf = env_conf(env_id='BenchmarkPushingEnv', mover_params=mover_params_small, collision_params=c_params_box_small)
pushing_bm_conf = env_conf(env_id='BenchmarkPushingEnv', mover_params=mover_params_medium, collision_params=c_params_box_medium)

# planning
# circle, small
planning_cs_conf_4 = env_conf(
    env_id='BenchmarkPlanningEnv',
    num_movers=4,
    mover_params=mover_params_small, 
    collision_params=c_params_circle_small
)
planning_cs_conf_3 = env_conf(
    env_id='BenchmarkPlanningEnv',
    num_movers=3,
    mover_params=mover_params_small, 
    collision_params=c_params_circle_small
)
planning_cs_conf_2 = env_conf(
    env_id='BenchmarkPlanningEnv',
    num_movers=2,
    mover_params=mover_params_small, 
    collision_params=c_params_circle_small
)
# circle, medium
planning_cm_conf_3 = env_conf(
    env_id='BenchmarkPlanningEnv',
    num_movers=3,
    mover_params=mover_params_medium, 
    collision_params=c_params_circle_medium
)
# box, small
planning_bs_conf_4 = env_conf(
    env_id='BenchmarkPlanningEnv',
    num_movers=4,
    mover_params=mover_params_small, 
    collision_params=c_params_box_small
)
planning_bs_conf_3 = env_conf(
    env_id='BenchmarkPlanningEnv',
    num_movers=3,
    mover_params=mover_params_small, 
    collision_params=c_params_box_small
)
planning_bs_conf_2 = env_conf(
    env_id='BenchmarkPlanningEnv',
    num_movers=2,
    mover_params=mover_params_small, 
    collision_params=c_params_box_small
)
# box, medium
planning_bm_conf_3 = env_conf(
    env_id='BenchmarkPlanningEnv',
    num_movers=3,
    mover_params=mover_params_medium, 
    collision_params=c_params_box_medium
)

# env
env_store = store(group='env_conf')
env_store(pushing_cs_conf, name='pushing_cs')
env_store(pushing_cm_conf, name='pushing_cm')
env_store(pushing_bs_conf, name='pushing_bs')
env_store(pushing_bm_conf, name='pushing_bm')
 
env_store(planning_cs_conf_4, name='planning_cs_4')
env_store(planning_cs_conf_3, name='planning_cs_3')
env_store(planning_cs_conf_2, name='planning_cs_2')

env_store(planning_cm_conf_3, name='planning_cm_3')

env_store(planning_bs_conf_4, name='planning_bs_4')
env_store(planning_bs_conf_3, name='planning_bs_3')
env_store(planning_bs_conf_2, name='planning_bs_2')

env_store(planning_bm_conf_3, name='planning_bm_3')