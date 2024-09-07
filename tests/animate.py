# Run this file as a script, not a test

from absl import app, flags
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from pathlib import Path
import numpy as np

flags.DEFINE_integer('num_seeds', None,
                     'Number of seeds that the training has been run with.')
flags.DEFINE_string('engine', 'mj',
                    'The engine that has been used to simulate the RL environment used.')
flags.DEFINE_string('date', None,
                    'The date of the training run, in YYYYMMDD format.')
flags.DEFINE_integer('num_steps', 1000,
                     'How many steps to animate, starting from the last step recorded.')
flags.DEFINE_string('filename', None,
                    'Name of the trajectory file to be read. Supersedes num_seeds / engine / date flags.')
flags.DEFINE_bool('randomize', True,
                  'Whether newly initialized episodes should be internally randomized by Unity.')
flags.DEFINE_integer('seed', 0,
                     'Seed for the Unity environemnt.')

def get_env() -> UnityToGymWrapper:
    """Generate a Unity environment from hardcoded path."""
    env = UnityEnvironment(
        "/Users/elan/radom_xr/app/macos/radom_app_latest.app",
        no_graphics=False,
        seed=flags.FLAGS.seed,
        additional_args=[
            '--random_initialization',
            str(flags.FLAGS.randomize).lower(),
        ],
    )
    env = UnityToGymWrapper(
        env,
        action_space_seed=flags.FLAGS.seed,
        allow_multiple_obs=True
    )
    return env


def animate(env: UnityToGymWrapper, controls: np.array):
    """Animate the `env` environment using `controls`."""

    for i, ctrl in enumerate(controls):
        obs, _, done, _ = env.step(ctrl)
        if done:
            env.reset()
    return


def main(_):

    # Read the trajectory, stored as a .npz archive.
    trajectory_f = flags.FLAGS.filename \
        if flags.FLAGS.filename is not None \
        else ''.join([
            f"actions_{flags.FLAGS.date}_",
            f"{flags.FLAGS.engine}_",
            f"{str(flags.FLAGS.num_seeds)}s.npz"
        ])
    print(type(trajectory_f))
    trajectory_fp = Path(
        Path.cwd() / ".." / "trajectories" / trajectory_f
    )
    trajectories_npz = np.load(trajectory_fp)

    # Convert the trajectories to a numpy array.
    singleton_key = list(trajectories_npz.keys())[0]
    controls = trajectories_npz[singleton_key]
    assert len(controls.shape) == 2 or len(controls.shape) == 3, (
        "You should provide either a two- or three-dimensional array, "
        f"but the array retrieved is {len(controls.shape)}-dimensional."
    )

    # Slice out the last `num_steps` steps.
    assert controls.shape[0] >= flags.FLAGS.num_steps, (
        f"There are {controls.shape[0]} action vectors in the archive, "
        f"but you asked to animate for {flags.FLAGS.num_steps} steps."
    )
    controls = controls[-flags.FLAGS.num_steps:]

    # Try to slice out a single sequence from a multi-seed batch.
    assert controls.shape[-1] == 38, (
        "The last dimension of the control array should be 38, or action dim "
        f"of the environment, but it is {controls.shape[-1]}."
    )
    if len(controls.shape) == 3:
        controls = controls[:, 0]

    # Prepare the environment and animate it with `controls`.
    env = get_env()
    animate(env, controls)
    env.close()
    return


if __name__ == "__main__":
    app.run(main)

