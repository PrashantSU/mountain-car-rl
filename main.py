"""
main.py
-------
Reproduces all four experiments for Mountain Car with linear function approximation:

  1. Environment exploration — trajectory and phase portrait of naive vs momentum policy
  2. Linear and polynomial features — MC and TD(0) prediction, RMSE comparison
  3. Tile coding — richer approximation, visitation heatmap, value slices
  4. Sarsa(0) control — learns to solve Mountain Car, learning curve

Run:
    python main.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from environment import (make_env, greedy_policy, momentum_policy,
                          phi_raw, phi_poly, TileCoder, TileCoderSA,
                          compute_reference_value, compute_rmse,
                          run_episode, POS_MIN, POS_MAX, VEL_MIN, VEL_MAX)
from algorithms  import mc_prediction, td0_prediction, sarsa_control
from visualize   import (plot_trajectory, plot_value_slice,
                          plot_visitation_heatmap, plot_learning_curve,
                          plot_rmse_bar, plot_value_surface)

os.makedirs("results", exist_ok=True)
GAMMA = 1.0

# ======================================================================
# 1. Environment exploration
# ======================================================================
print("=== 1. Environment exploration ===")
env = make_env(seed=0)

states_greedy, rewards_greedy = run_episode(env, greedy_policy)
states_momentum, rewards_momentum = run_episode(env, momentum_policy)

print(f"  Naive greedy policy:   {len(rewards_greedy)} steps, "
      f"return = {rewards_greedy.sum():.0f}")
print(f"  Momentum policy:       {len(rewards_momentum)} steps, "
      f"return = {rewards_momentum.sum():.0f}")

plot_trajectory(states_greedy,   "Naive greedy policy",
                save_path="results/trajectory_greedy.png")
plot_trajectory(states_momentum, "Momentum policy (push in direction of v)",
                save_path="results/trajectory_momentum.png")

# ======================================================================
# 2. Reference value function
# ======================================================================
print("\n=== 2. Computing reference value function (takes ~30s) ===")
xs_ref, vs_ref, V_ref = compute_reference_value(
    env, momentum_policy, n_x=31, n_v=31, n_mc=20)
print(f"  Reference range: [{V_ref.min():.1f}, {V_ref.max():.1f}]")

# ======================================================================
# 3. Raw and polynomial features — MC and TD(0)
# ======================================================================
print("\n=== 3. Raw and polynomial features ===")
N_EP = 8000

print("  Training MC + raw features...")
w_mc_raw  = mc_prediction(env, momentum_policy, phi_raw,
                           alpha=1e-3, gamma=GAMMA, episodes=N_EP)
print("  Training TD(0) + raw features...")
w_td_raw  = td0_prediction(env, momentum_policy, phi_raw,
                            alpha=1e-3, gamma=GAMMA, episodes=N_EP)
print("  Training MC + polynomial features...")
w_mc_poly = mc_prediction(env, momentum_policy, phi_poly,
                           alpha=5e-4, gamma=GAMMA, episodes=N_EP)
print("  Training TD(0) + polynomial features...")
w_td_poly = td0_prediction(env, momentum_policy, phi_poly,
                            alpha=5e-4, gamma=GAMMA, episodes=N_EP)

rmse_raw_mc   = compute_rmse(w_mc_raw,  phi_raw,  xs_ref, vs_ref, V_ref)
rmse_raw_td   = compute_rmse(w_td_raw,  phi_raw,  xs_ref, vs_ref, V_ref)
rmse_poly_mc  = compute_rmse(w_mc_poly, phi_poly, xs_ref, vs_ref, V_ref)
rmse_poly_td  = compute_rmse(w_td_poly, phi_poly, xs_ref, vs_ref, V_ref)

rmse_part2 = {
    "MC + Raw":   rmse_raw_mc,
    "TD + Raw":   rmse_raw_td,
    "MC + Poly":  rmse_poly_mc,
    "TD + Poly":  rmse_poly_td,
}
print("\n  RMSE vs reference:")
for k, v in rmse_part2.items():
    print(f"    {k:<18} {v:.3f}")

plot_rmse_bar(rmse_part2,
              title="RMSE: Raw and Polynomial Features",
              save_path="results/rmse_raw_poly.png")

plot_value_slice(
    w_dict  = {"MC + Raw": w_mc_raw, "TD + Raw": w_td_raw,
                "MC + Poly": w_mc_poly, "TD + Poly": w_td_poly},
    phi_dict= {"MC + Raw": phi_raw, "TD + Raw": phi_raw,
                "MC + Poly": phi_poly, "TD + Poly": phi_poly},
    xs_ref=xs_ref, vs_ref=vs_ref, V_ref=V_ref, v0=0.0,
    title="Raw vs Polynomial Features",
    save_path="results/slice_raw_poly.png"
)

# ======================================================================
# 4. Tile coding
# ======================================================================
print("\n=== 4. Tile coding ===")
tc     = TileCoder(n_tilings=16, tiles_pos=16, tiles_vel=16)
alpha_tile = 0.2 / tc.n_tilings

print("  Training MC + tile coding...")
w_mc_tile = mc_prediction(env, momentum_policy, tc,
                           alpha=alpha_tile, gamma=GAMMA, episodes=10_000)
print("  Training TD(0) + tile coding...")
w_td_tile = td0_prediction(env, momentum_policy, tc,
                            alpha=alpha_tile, gamma=GAMMA, episodes=10_000)

rmse_mc_tile = compute_rmse(w_mc_tile, tc, xs_ref, vs_ref, V_ref)
rmse_td_tile = compute_rmse(w_td_tile, tc, xs_ref, vs_ref, V_ref)

rmse_all = {**rmse_part2,
            "MC + Tile":  rmse_mc_tile,
            "TD + Tile":  rmse_td_tile}
print(f"\n  MC  + tile coding RMSE: {rmse_mc_tile:.3f}")
print(f"  TD  + tile coding RMSE: {rmse_td_tile:.3f}")

plot_rmse_bar(rmse_all,
              title="RMSE: All Feature Representations",
              save_path="results/rmse_all.png")

for v0 in [0.0, 0.04]:
    plot_value_slice(
        w_dict  = {"MC + Tile": w_mc_tile, "TD + Tile": w_td_tile},
        phi_dict= {"MC + Tile": tc, "TD + Tile": tc},
        xs_ref=xs_ref, vs_ref=vs_ref, V_ref=V_ref, v0=v0,
        title=f"Tile Coding — value slice at v₀={v0}",
        save_path=f"results/slice_tile_v{int(v0*100)}.png"
    )

# Visitation heatmap
xs_vis, vs_vis = [], []
for _ in range(300):
    states, _ = run_episode(env, momentum_policy)
    xs_vis.extend(states[:, 0])
    vs_vis.extend(states[:, 1])

plot_visitation_heatmap(np.array(xs_vis), np.array(vs_vis),
                        title="State visitation — momentum policy",
                        save_path="results/visitation_heatmap.png")

# ======================================================================
# 5. Sarsa(0) control
# ======================================================================
print("\n=== 5. Sarsa(0) control ===")
env_ctrl = make_env(seed=42)
n_actions = env_ctrl.action_space.n

tc_ctrl  = TileCoder(n_tilings=8, tiles_pos=12, tiles_vel=12)
tc_sa    = TileCoderSA(tc_ctrl, n_actions)

w_sarsa, returns, steps = sarsa_control(
    env_ctrl, tc_sa, n_actions,
    episodes  = 8000,
    alpha     = 0.3 / tc_ctrl.n_tilings,
    gamma     = GAMMA,
    eps_start = 0.2,
    eps_end   = 0.01,
    eps_decay = 4000,
)

final_avg = np.mean(steps[-200:])
print(f"\n  Final 200-episode average steps: {final_avg:.1f}")
print(f"  Best episode: {steps.min()} steps")

plot_learning_curve(steps, window=100,
                    title="Sarsa(0) + Tile Coding: Mountain Car Learning Curve",
                    save_path="results/sarsa_learning_curve.png")

# Summary
print("\n=== Summary ===")
print(f"  {'Method':<22} {'RMSE':>8}")
print("  " + "-" * 32)
for k, v in rmse_all.items():
    print(f"  {k:<22} {v:>8.3f}")
print(f"\n  Sarsa(0) final avg steps: {final_avg:.0f}  "
      f"(200-step max = 200)")
print("\nAll results saved to results/")
