"""
environment.py
--------------
Mountain Car environment utilities and feature representations.

MountainCar-v0 dynamics:
    v_{t+1} = v_t + 0.001*(a_t - 1) - 0.0025*cos(3*x_t)
    x_{t+1} = x_t + v_{t+1}

State:  s = (x, v)  where x ∈ [-1.2, 0.6], v ∈ [-0.07, 0.07]
Goal:   x >= 0.5
Reward: -1 per step (sparse — only positive signal is reaching the goal)

The car's engine is too weak to climb directly; it must first reverse up
the left hill to build momentum — making naive greedy strategies suboptimal.

Feature representations
-----------------------
Three representations of increasing expressiveness are implemented:

1. Raw (linear):   φ(s) = [1, x_n, v_n]^T
   Produces a hyperplane — too simple for the curved value landscape.

2. Polynomial:     φ(s) = [1, x_n, v_n, x_n², v_n², x_n·v_n]^T
   Adds curvature but misses localised structure.

3. Tile coding:    N overlapping grids, one active tile per tiling.
   Sparse binary features with local generalisation — the most effective.
"""

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym

POS_MIN, POS_MAX = -1.2, 0.6
VEL_MIN, VEL_MAX = -0.07, 0.07


# ------------------------------------------------------------------
# Environment factory
# ------------------------------------------------------------------

def make_env(seed=0):
    """Create a MountainCar-v0 environment."""
    env = gym.make("MountainCar-v0")
    env.reset(seed=seed)
    return env


# ------------------------------------------------------------------
# Policies
# ------------------------------------------------------------------

def greedy_policy(state):
    """
    Naive policy: always push toward the goal (right).
    Fails because the engine is too weak to climb directly.
    """
    x, v = state
    return 2 if x < 0.5 else 0


def momentum_policy(state):
    """
    Good heuristic: push in the direction of current velocity.
    Builds momentum by rocking back and forth.
    """
    x, v = state
    return 2 if v >= 0 else 0


# ------------------------------------------------------------------
# Episode simulation
# ------------------------------------------------------------------

def run_episode(env, policy, max_steps=200):
    """
    Roll out one episode under a given policy.

    Returns
    -------
    states  : np.ndarray shape (T, 2)
    rewards : np.ndarray shape (T,)
    """
    states, rewards = [], []
    s, _ = env.reset()
    done  = False
    steps = 0
    while not done and steps < max_steps:
        states.append(s)
        a = policy(s)
        s, r, terminated, truncated, _ = env.step(a)
        rewards.append(r)
        done  = terminated or truncated
        steps += 1
    return np.array(states), np.array(rewards)


# ------------------------------------------------------------------
# State normalisation
# ------------------------------------------------------------------

def normalize(state):
    """Normalise (x, v) to [-1, 1] x [-1, 1]."""
    x, v = state
    x_n = (x - POS_MIN) / (POS_MAX - POS_MIN) * 2 - 1
    v_n = (v - VEL_MIN) / (VEL_MAX - VEL_MIN) * 2 - 1
    return x_n, v_n


def normalize01(state):
    """Normalise (x, v) to [0, 1) for tile coding."""
    x, v = state
    x01 = min(max((x - POS_MIN) / (POS_MAX - POS_MIN), 0.0), 0.999999)
    v01 = min(max((v - VEL_MIN) / (VEL_MAX - VEL_MIN), 0.0), 0.999999)
    return x01, v01


# ------------------------------------------------------------------
# Feature representations
# ------------------------------------------------------------------

def phi_raw(state):
    """
    Raw linear features: φ(s) = [1, x_n, v_n]

    Produces a linear (hyperplane) value approximation.
    Cannot represent the nonlinear value landscape of Mountain Car.
    """
    x_n, v_n = normalize(state)
    return np.array([1.0, x_n, v_n], dtype=float)


def phi_poly(state):
    """
    Second-order polynomial features: φ(s) = [1, x_n, v_n, x_n², v_n², x_n·v_n]

    Introduces curvature into the approximation.
    Better than raw features but still cannot capture localised structure.
    """
    x_n, v_n = normalize(state)
    return np.array([1.0, x_n, v_n, x_n**2, v_n**2, x_n * v_n], dtype=float)


class TileCoder:
    """
    Tile coding feature representation.

    Overlays the state space with N_TILINGS grids, each slightly offset.
    For each tiling, exactly one tile is active — giving N_TILINGS active
    features out of N_TILINGS * TILES_POS * TILES_VEL total features.

    Properties:
    - Sparse binary features: efficient dot products
    - Local generalisation: nearby states share tiles
    - Linear approximation with nonlinear expressiveness

    Parameters
    ----------
    n_tilings : int  number of overlapping grids (default 16)
    tiles_pos : int  tiles along position axis (default 16)
    tiles_vel : int  tiles along velocity axis (default 16)
    """

    def __init__(self, n_tilings=16, tiles_pos=16, tiles_vel=16):
        self.n_tilings       = n_tilings
        self.tiles_pos       = tiles_pos
        self.tiles_vel       = tiles_vel
        self.tiles_per_tiling = tiles_pos * tiles_vel
        self.dim             = n_tilings * tiles_pos * tiles_vel
        self.offsets         = np.linspace(0.0, 1.0, n_tilings, endpoint=False)

    def __call__(self, state):
        """Return dense binary feature vector of length self.dim."""
        x01, v01 = normalize01(state)
        features = np.zeros(self.dim, dtype=float)
        for t in range(self.n_tilings):
            xs = (x01 + self.offsets[t] / self.tiles_pos) % 1.0
            vs = (v01 + self.offsets[t] / self.tiles_vel) % 1.0
            i  = min(int(xs * self.tiles_pos), self.tiles_pos - 1)
            j  = min(int(vs * self.tiles_vel), self.tiles_vel - 1)
            features[t * self.tiles_per_tiling + i * self.tiles_vel + j] = 1.0
        return features


class TileCoderSA:
    """
    Tile coding for state-action pairs.

    Allocates one feature block per action. The active features for
    action a are in block [a * state_dim : (a+1) * state_dim].
    This allows a single weight vector to represent Q(s, a) for all actions.
    """

    def __init__(self, tile_coder, n_actions):
        self.tile_coder = tile_coder
        self.n_actions  = n_actions
        self.state_dim  = tile_coder.dim
        self.dim        = tile_coder.dim * n_actions

    def __call__(self, state, action):
        z       = self.tile_coder(state)
        features= np.zeros(self.dim, dtype=float)
        start   = action * self.state_dim
        features[start:start + self.state_dim] = z
        return features


# ------------------------------------------------------------------
# Reference value function (Monte Carlo grid evaluation)
# ------------------------------------------------------------------

def compute_reference_value(env, policy, n_x=31, n_v=31, n_mc=20):
    """
    Estimate V^π on a grid by setting the environment state directly
    and running Monte Carlo rollouts from each grid point.

    Note: this approach is only feasible because Mountain Car is a
    small 2D problem and the simulator allows arbitrary state resets.
    It cannot generalise to large or real-world state spaces.

    Returns
    -------
    xs : np.ndarray shape (n_x,)
    vs : np.ndarray shape (n_v,)
    V  : np.ndarray shape (n_x, n_v)
    """
    xs = np.linspace(POS_MIN, POS_MAX, n_x)
    vs = np.linspace(VEL_MIN, VEL_MAX, n_v)
    V  = np.zeros((n_x, n_v))

    for i, x in enumerate(xs):
        for j, v in enumerate(vs):
            returns = []
            for _ in range(n_mc):
                env.reset()
                env.unwrapped.state = np.array([x, v], dtype=float)
                s   = np.array([x, v], dtype=float)
                G   = 0.0
                done = False
                while not done:
                    a = policy(s)
                    s, r, terminated, truncated, _ = env.step(a)
                    G   += r
                    done = terminated or truncated
                returns.append(G)
            V[i, j] = np.mean(returns)
    return xs, vs, V


def compute_rmse(w, phi_fn, xs, vs, V_ref):
    """RMSE of w^T φ(s) vs V_ref on a grid."""
    errs = []
    for i, x in enumerate(xs):
        for j, v in enumerate(vs):
            v_hat = np.dot(w, phi_fn(np.array([x, v])))
            errs.append((v_hat - V_ref[i, j]) ** 2)
    return float(np.sqrt(np.mean(errs)))
