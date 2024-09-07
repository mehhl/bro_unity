import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dm_control.suite import dog
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper


def make_unity_env():
    env = UnityEnvironment(
        "/Users/elan/radom_app_tst.app",
        no_graphics=True,
        additional_args=[
            '--random_initialization',
            'false',
        ]
    )
    env = UnityToGymWrapper(env)
    return env


def make_raw_mujoco_env():
    env = dog.run()
    env.physics.reset()
    return env


def run_simulation(env, is_unity=True):

    joint_angles = []
    for _ in tqdm(range(1000), desc="Simulation steps", leave=False):
        if is_unity:
            obs, _, _, _ = env.step(np.zeros(38))
            joint_angles.append(obs)
        else:
            _, _, _, obs = env.step(np.zeros(38))
            joint_angles.append(
                np.concatenate(
                    list(obs.values())
                )
            )
    return np.array(joint_angles)


def compare_joint_angles(unity_env, mujoco_env, num_runs=1):
    unity_results = []
    mujoco_results = []

    for _ in tqdm(range(num_runs), desc="Running simulations"):
        unity_results.append(run_simulation(unity_env, is_unity=True))
        mujoco_results.append(run_simulation(mujoco_env, is_unity=False))

    return np.array(unity_results), np.array(mujoco_results)


def plot_comparison(unity_data, mujoco_data):
    num_joints = unity_data.shape[2]
    rows = int(np.ceil(np.sqrt(num_joints)))
    cols = int(np.ceil(num_joints / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    fig.suptitle('Joint Angle Comparisons', fontsize=16)

    for i in range(num_joints):
        ax = axes[i // cols, i % cols] if rows > 1 else axes[i]

        unity_mean = unity_data[:, :, i].mean(axis=0)
        unity_std = unity_data[:, :, i].std(axis=0)
        mujoco_mean = mujoco_data[:, :, i].mean(axis=0)
        mujoco_std = mujoco_data[:, :, i].std(axis=0)

        ax.plot(unity_mean, label='Unity')
        ax.fill_between(range(len(unity_mean)),
                        unity_mean - unity_std,
                        unity_mean + unity_std,
                        alpha=0.3)
        ax.plot(mujoco_mean, label='MuJoCo')
        ax.fill_between(range(len(mujoco_mean)),
                        mujoco_mean - mujoco_std,
                        mujoco_mean + mujoco_std,
                        alpha=0.3)
        ax.set_title(f'Observation {i}')
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.legend()

    # Remove any unused subplots
    for i in range(num_joints, rows * cols):
        fig.delaxes(axes[i // cols, i % cols] if rows > 1 else axes[i])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    unity_env = make_unity_env()
    mujoco_env = make_raw_mujoco_env()

    unity_results, mujoco_results = compare_joint_angles(unity_env, mujoco_env)

    plot_comparison(unity_results, mujoco_results)