import gym
from datetime import datetime
from SNN_model.TE_SNN_Policy import RecurrentActorCriticPolicy
from SNN_model.TE_SNN_PPO import RecurrentPPO
import torch
import numpy as np
import os
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from stable_baselines3.common.utils import get_linear_fn

from f110_gym.envs.base_classes import Integrator

current_dir = os.path.dirname(os.getcwd())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(model_name, load_model=None, track = 'barcelona'):
    def render_callback(env_renderer):
        # custom extra drawing function

        e = env_renderer

        # update camera to follow car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800

    if track == 'barcelona':
        map_path = current_dir + '/Autonomous_driving/f1tenth_racetracks/Barcelona/barcelona_map'
        map_ext = '.png'
        wp_path = current_dir + '/Autonomous_driving/f1tenth_racetracks/Barcelona/centerline.npy'
        wp_data = np.load(wp_path)
        start = 0
    else:
        map_path = current_dir + '/Autonomous_driving/f1tenth_racetracks/YasMarina/YasMarina_map'
        map_ext = '.png'
        wp_path = current_dir + '/Autonomous_driving/f1tenth_racetracks/YasMarina/YasMarina_centerline.csv'
        csv_path = os.path.splitext(wp_path)[0] + '.csv'
        with open(csv_path) as f:
            import csv
            waypoints = [tuple(line) for line in csv.reader(f)]
            # waypoints are [x, y, speed, theta]
            wp_data = np.array([(float(pt[0]), float(pt[1]), float(pt[2]), float(pt[3]), 0) for pt in waypoints])
        start = 250


    env = gym.make('f110_gym:f110-v0', map=map_path, map_ext=map_ext, waypoint=wp_data, num_agents=1, timestep=0.01, integrator=Integrator.RK4,device=device,start_id=start)
    env.add_render_callback(render_callback)
    eval_env = gym.make('f110_gym:f110-v0', map=map_path, map_ext=map_ext, waypoint=wp_data, num_agents=1, timestep=0.01, integrator=Integrator.RK4, eval_flag=1,device=device,start_id=start)
    eval_env.add_render_callback(render_callback)

    filename = current_dir + "/Autonomous_driving/log/" + track + '_' + model_name + '_' + datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    #torch.autograd.set_detect_anomaly(True)

    try:
        if load_model is not None:
            print("********************yes*************************")
            model = RecurrentPPO.load(
                load_model,
                env,
            )
        else:
            model = RecurrentPPO(
                RecurrentActorCriticPolicy,
                env,
                verbose=2,
                n_steps=10000,
                batch_size=2048,
                gae_lambda=0.90,
                device=device,
                learning_rate=get_linear_fn(5e-4,3e-4,1),
                tensorboard_log=filename,
                policy_kwargs={
                    "snn_hidden_size": 64,
                    "net_arch" : dict(pi=[32], vf=[32]),
                    "neuron_type": 'TE-N'
                },
            )

        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
        eval_callback = EvalCallback(eval_env,
                                    callback_on_new_best=callback_on_best,
                                    verbose=1,
                                    best_model_save_path=filename,
                                    log_path=filename,
                                    eval_freq=10000,
                                    deterministic=True,
                                    render=False,
                                    n_eval_episodes=1
                                    )

        model.learn(total_timesteps=2000000,
                    log_interval=1,
                    callback=eval_callback
                    )
        model.save(filename +'/'+ model_name)
    finally:
        env.close()


if __name__ == "__main__":
    main(model_name='TE-SNN-N',track = 'barcelona')