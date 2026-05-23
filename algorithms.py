"""
algorithms.py
-------------
Policy evaluation and control algorithms with linear function approximation.

Prediction (fixed policy):
  - Monte Carlo (MC) with semi-gradient updates
  - TD(0)           with semi-gradient updates

Control (learning a policy):
  - Semi-gradient Sarsa(0) with tile-coded action-value features

All methods use the update:
    w ← w + α * δ * φ(s)

where δ is the prediction error (MC return or TD target minus current estimate).
"""

import numpy as np
from environment import run_episode


# ------------------------------------------------------------------
# Monte Carlo prediction (semi-gradient)
# ------------------------------------------------------------------

def mc_prediction(env, policy, phi_fn, alpha=1e-3, gamma=1.0, episodes=5000):
    """
    Semi-gradient Monte Carlo prediction for V^π.

    Accumulates the full return G_t backwards through each episode, then
    applies a gradient step for each state visited:

        w ← w + α (G_t - ŵ^T φ(s_t)) φ(s_t)

    MC is unbiased (targets are true returns) but has high variance —
    particularly in Mountain Car where rewards are sparse and returns
    are either 0 or a large negative number.

    Parameters
    ----------
    phi_fn   : callable  state -> feature vector
    alpha    : float     step size
    gamma    : float     discount factor
    episodes : int       number of training episodes

    Returns
    -------
    w : np.ndarray  learned weight vector
    """
    d = phi_fn(env.reset()[0]).shape[0]
    w = np.zeros(d, dtype=float)

    for _ in range(episodes):
        states, rewards = run_episode(env, policy)
        G = 0.0
        for s, r in zip(states[::-1], rewards[::-1]):
            G      = r + gamma * G
            phi    = phi_fn(s)
            v_hat  = np.dot(w, phi)
            w     += alpha * (G - v_hat) * phi

    return w


# ------------------------------------------------------------------
# TD(0) prediction (semi-gradient)
# ------------------------------------------------------------------

def td0_prediction(env, policy, phi_fn, alpha=1e-3, gamma=1.0, episodes=5000):
    """
    Semi-gradient TD(0) prediction for V^π.

    Bootstraps from the current value estimate at each step:

        δ_t = r_{t+1} + γ ŵ^T φ(s_{t+1}) - ŵ^T φ(s_t)
        w   ← w + α δ_t φ(s_t)

    Compared to MC:
    - Lower variance (one-step target instead of full return)
    - Biased (bootstraps from its own estimate)
    - Updates online (no need to wait for episode end)
    - More sensitive to poor feature representations

    Parameters
    ----------
    Same as mc_prediction.

    Returns
    -------
    w : np.ndarray  learned weight vector
    """
    d = phi_fn(env.reset()[0]).shape[0]
    w = np.zeros(d, dtype=float)

    for _ in range(episodes):
        s, _ = env.reset()
        done  = False
        while not done:
            phi   = phi_fn(s)
            a     = policy(s)
            s2, r, terminated, truncated, _ = env.step(a)
            done  = terminated or truncated
            target = r if done else (r + gamma * np.dot(w, phi_fn(s2)))
            w     += alpha * (target - np.dot(w, phi)) * phi
            s      = s2

    return w


# ------------------------------------------------------------------
# Semi-gradient Sarsa(0) control
# ------------------------------------------------------------------

def sarsa_control(env, phi_sa_fn, n_actions, episodes=8000,
                  alpha=0.3/8, gamma=1.0,
                  eps_start=0.2, eps_end=0.01, eps_decay=4000):
    """
    Semi-gradient Sarsa(0) with linear action-value approximation.

    Extends TD prediction to control by using action-value features φ(s,a):

        Q̂(s,a;w) = w^T φ(s,a)

    On-policy update targeting the next ε-greedy action a':
        δ = r + γ Q̂(s',a';w) - Q̂(s,a;w)
        w ← w + α δ φ(s,a)

    Sarsa is on-policy: it learns the value of the ε-greedy behavior policy,
    not the greedy policy. As ε → 0, the two converge.

    Epsilon decays exponentially: ε_t = ε_end + (ε_start - ε_end)·exp(-t/decay)

    Parameters
    ----------
    phi_sa_fn : callable  (state, action) -> feature vector
    n_actions : int       number of discrete actions

    Returns
    -------
    w         : np.ndarray  learned weight vector
    returns   : np.ndarray  per-episode total return
    steps     : np.ndarray  per-episode step count
    """
    d       = phi_sa_fn(env.reset()[0], 0).shape[0]
    w       = np.zeros(d, dtype=float)
    returns = []
    steps   = []

    def q_hat(s, a):
        return float(np.dot(w, phi_sa_fn(s, a)))

    def eps_greedy(s, eps):
        if np.random.rand() < eps:
            return env.action_space.sample()
        qs = [q_hat(s, a) for a in range(n_actions)]
        return int(np.argmax(qs))

    for ep in range(episodes):
        eps   = eps_end + (eps_start - eps_end) * np.exp(-ep / eps_decay)
        s, _  = env.reset()
        a     = eps_greedy(s, eps)
        G, T  = 0.0, 0
        done   = False

        while not done:
            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            G   += r
            T   += 1

            if done:
                delta = r - q_hat(s, a)
                w    += alpha * delta * phi_sa_fn(s, a)
            else:
                a2    = eps_greedy(s2, eps)
                delta = r + gamma * q_hat(s2, a2) - q_hat(s, a)
                w    += alpha * delta * phi_sa_fn(s, a)
                s, a  = s2, a2

        returns.append(G)
        steps.append(T)

        if (ep + 1) % 1000 == 0:
            ma = np.mean(steps[-100:])
            print(f"  Episode {ep+1:5d}/{episodes}  "
                  f"steps={T:3d}  100-ep avg={ma:.0f}")

    return w, np.array(returns), np.array(steps)
