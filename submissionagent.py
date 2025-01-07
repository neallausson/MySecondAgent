#!/usr/bin/env python3
import os
import yaml
import json
import diambra.arena
from stable_baselines3 import PPO
from diambra.arena import SpaceTypes, Roles, EnvironmentSettings,load_settings_flat_dict
from diambra.arena.utils.gym_utils import available_games
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
import random
import argparse

def main(cfg_file, trained_model, test=False):
    # Read the cfg file
    yaml_file = open(cfg_file)
    params = yaml.load(yaml_file, Loader=yaml.FullLoader)
    print("Config parameters = ", json.dumps(params, sort_keys=True, indent=4))
    yaml_file.close()

    base_path = os.path.dirname(os.path.abspath(__file__))
    model_folder = os.path.join(base_path, params["folders"]["parent_dir"], params["settings"]["game_id"],
                                params["folders"]["model_name"], "model")

    # Settings
    params["settings"]["action_space"] = SpaceTypes.DISCRETE if params["settings"]["action_space"] == "discrete" else SpaceTypes.MULTI_DISCRETE
    settings = load_settings_flat_dict(EnvironmentSettings, params["settings"])
    settings.role = Roles.P1

    # Wrappers Settings
    wrappers_settings = load_settings_flat_dict(WrappersSettings, params["wrappers_settings"])
    wrappers_settings.normalize_reward = False

    # Create environment
    env, num_envs = make_sb3_env(settings.game_id, settings, wrappers_settings, no_vec=True)
    print("Activated {} environment(s)".format(num_envs))

    # Load the trained agent
    #model_path = os.path.join(model_folder, trained_model)
    agent = PPO.load(trained_model)

    # Print policy network architecture
    print("Policy architecture:")
    print(agent.policy)

    observation, info = env.reset()

    while True:
        action, _state = agent.predict(observation, deterministic=False)
        observation, reward, terminated, truncated, info = env.step(int(action))

        if terminated or truncated:
            observation, info = env.reset()
            if info["env_done"] or test is True:
                break

    # Close the environment
    env.close()

    # Return success
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgFile", type=str, required=True, help="Configuration file")
    parser.add_argument("--trainedModel", type=str, default="model", help="Model checkpoint")
    parser.add_argument("--test", type=int, default=0, help="Test mode")
    opt = parser.parse_args()
    print(opt)

    main(opt.cfgFile, opt.trainedModel, bool(opt.test))
