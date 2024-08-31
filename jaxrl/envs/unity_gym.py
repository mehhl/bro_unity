from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

from gym import core, spaces
from gym.wrappers import RescaleAction
import numpy as np


def _make_env_unity_gym(env_name: str, seed: int, **kwargs) -> core.Env:
    domain_name, task_name = "Dog", "Run"
    env = UnityEnvironment(
        file_name="/Users/elan/radom_app.app",
        base_port=19996 + seed,
        **kwargs
    )
    env = UnityToGymWrapper(env)
    env = RescaleAction(env, -1.0, 1.0)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


class make_env_unity_gym(core.Env):
    def __init__(self, env_name: str, seed: int, num_envs: int, max_t=1000, **kwargs):
        env_fns = [
            lambda i=i: _make_env_unity_gym(env_name, seed + i, **kwargs)
            for i in range(num_envs)
        ]
        self.envs = [env_fn() for env_fn in env_fns]
        self.max_t = max_t
        self.num_seeds = len(self.envs)

        action_space_spec = self.envs[0].action_space
        observation_space_spec = self.envs[0].observation_space
        self.action_space = spaces.Box(
            low=action_space_spec.low[None].repeat(num_envs, axis=0),
            high=action_space_spec.high[None].repeat(num_envs, axis=0),
            shape=(num_envs, action_space_spec.shape[0]),
            dtype=action_space_spec.dtype,
        )
        self.observation_space = spaces.Box(
            low=observation_space_spec.low[None].repeat(num_envs, axis=0),
            high=observation_space_spec.high[None].repeat(num_envs, axis=0),
            shape=(num_envs, observation_space_spec.shape[0]),
            dtype=observation_space_spec.dtype,
        )

    def _reset_idx(self, idx):
        return self.envs[idx].reset()

    def reset_where_done(self, observations, terms, truns):
        resets = np.zeros(terms.shape)
        for j, (term, trun) in enumerate(zip(terms, truns)):
            if (term == True) or (trun == True):
                observations[j], terms[j], truns[j] \
                    = self._reset_idx(j), False, False
                resets[j] = 1
        return observations, terms, truns, resets

    def generate_masks(self, terms, truns):
        masks = []
        for term, trun in zip(terms, truns):
            if not term or trun:
                mask = 1.0
            else:
                mask = 0.0
            masks.append(mask)
        masks = np.array(masks)
        return masks

    def reset(self):
        obs = []
        for env in self.envs:
            obs.append(env.reset())
        return np.stack(obs)

    def step(self, actions):
        obs, rews, terms, truns = [], [], [], []
        for env, action in zip(self.envs, actions):
            ob, reward, done, info = env.step(action)
            obs.append(ob)
            rews.append(reward)
            terms.append(False)
            trun = True if 'TimeLimit.truncated' in info else False
            truns.append(trun)
        return np.stack(obs), np.stack(rews), np.stack(terms), np.stack(
            truns), None

    def evaluate(self, agent, num_episodes=5, temperature=0.0):
        num_seeds = self.num_seeds
        returns_eval = []
        for episode in range(num_episodes):
            observations = self.reset()
            returns = np.zeros(num_seeds)
            for i in range(self.max_t):  # CHANGE?
                actions = agent.sample_actions(observations,
                                               temperature=temperature)
                next_observations, rewards, terms, truns, goals \
                    = self.step(actions)
                returns += rewards
                observations = next_observations
            returns_eval.append(returns)
        return {'return': np.array(returns_eval).mean(axis=0)}
