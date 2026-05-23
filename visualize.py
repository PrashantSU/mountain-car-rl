"""
visualize.py
------------
Plotting utilities for Mountain Car RL experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from environment import POS_MIN, POS_MAX, VEL_MIN, VEL_MAX


def plot_trajectory(states, title="Trajectory", save_path=None):
    """
    Three-panel plot: position over time, velocity over time, phase portrait.
    """
    pos, vel = states[:, 0], states[:, 1]
    t        = np.arange(len(pos))

    fig, axes = plt.subplots(3, 1, figsize=(9, 10))

    axes[0].plot(t, pos, color='steelblue')
    axes[0].axhline(0.5, ls='--', color='tomato', label='Goal x=0.5')
    axes[0].set_ylabel("Position x"); axes[0].set_title(f"{title} — Position")
    axes[0].legend(fontsize=9)

    axes[1].plot(t, vel, color='darkorange')
    axes[1].axhline(0, ls='--', color='gray', lw=0.8)
    axes[1].set_ylabel("Velocity v"); axes[1].set_title(f"{title} — Velocity")

    axes[2].plot(pos, vel, 'o-', markersize=2, color='seagreen', alpha=0.7)
    axes[2].set_xlabel("Position x"); axes[2].set_ylabel("Velocity v")
    axes[2].set_title("Phase portrait (x vs v)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_value_surface(w_dict, phi_dict, xs, vs, V_ref,
                       title="Value function surface", save_path=None):
    """
    2D value function surface (position x velocity) for multiple approximators.
    """
    n_plots = 1 + len(w_dict)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.5), sharey=True)

    xs_g, vs_g = np.meshgrid(xs, vs, indexing='ij')
    vmin, vmax  = V_ref.min(), V_ref.max()

    axes[0].imshow(V_ref.T, origin='lower', aspect='auto', cmap='viridis',
                   extent=[POS_MIN, POS_MAX, VEL_MIN, VEL_MAX],
                   vmin=vmin, vmax=vmax)
    axes[0].set_title("Reference V^π"); axes[0].set_xlabel("Position")
    axes[0].set_ylabel("Velocity")

    for ax, (label, w), phi_fn in zip(axes[1:], w_dict.items(), phi_dict.values()):
        V_hat = np.array([[np.dot(w, phi_fn([x, v]))
                           for v in vs] for x in xs])
        im = ax.imshow(V_hat.T, origin='lower', aspect='auto', cmap='viridis',
                       extent=[POS_MIN, POS_MAX, VEL_MIN, VEL_MAX],
                       vmin=vmin, vmax=vmax)
        ax.set_title(label); ax.set_xlabel("Position")

    plt.colorbar(im, ax=axes[-1], label='V(s)')
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_value_slice(w_dict, phi_dict, xs_ref, vs_ref, V_ref,
                     v0=0.0, title="Value slice at fixed velocity",
                     save_path=None):
    """
    Value function slice along position axis at fixed velocity v0.
    Compares multiple approximations against the reference.
    """
    xs_plot = np.linspace(POS_MIN, POS_MAX, 300)
    j       = np.argmin(np.abs(vs_ref - v0))
    V_slice = np.interp(xs_plot, xs_ref, V_ref[:, j])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs_plot, V_slice, lw=3, color='black',
            label=f"Reference (v≈{vs_ref[j]:.3f})")

    colors = ['#2980b9', '#e74c3c', '#27ae60', '#8e44ad', '#e67e22']
    for (label, w), phi_fn, color in zip(w_dict.items(), phi_dict.values(), colors):
        vals = [np.dot(w, phi_fn([x, v0])) for x in xs_plot]
        ax.plot(xs_plot, vals, label=label, color=color)

    ax.axvline(0.5, ls='--', color='tomato', lw=1.2, label='Goal x=0.5')
    ax.set_xlabel("Position x"); ax.set_ylabel("Estimated V(s)")
    ax.set_title(f"{title}  (v₀ = {v0:.3f})", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_visitation_heatmap(xs_vis, vs_vis, title="State visitation", save_path=None):
    """
    2D histogram of visited (x, v) states under a policy.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    H, xedges, vedges = np.histogram2d(
        xs_vis, vs_vis, bins=[60, 60],
        range=[[POS_MIN, POS_MAX], [VEL_MIN, VEL_MAX]])
    extent = [xedges[0], xedges[-1], vedges[0], vedges[-1]]
    im = ax.imshow(H.T, origin='lower', extent=extent, aspect='auto', cmap='plasma')
    plt.colorbar(im, ax=ax, label='Visit count')
    ax.set_xlabel("Position x"); ax.set_ylabel("Velocity v")
    ax.set_title(title, fontsize=11)
    ax.axvline(0.5, ls='--', color='white', lw=1.2, label='Goal')
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_learning_curve(steps, window=100, title="Sarsa(0): Learning curve",
                        save_path=None):
    """
    Steps-per-episode learning curve with moving average overlay.
    """
    ma = np.convolve(steps, np.ones(window) / window, mode='valid')
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, alpha=0.25, color='steelblue', label='Steps per episode')
    ax.plot(np.arange(window - 1, len(steps)), ma, lw=2, color='steelblue',
            label=f'{window}-ep moving average')
    ax.set_xlabel("Episode"); ax.set_ylabel("Steps until termination")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_rmse_bar(rmse_dict, title="RMSE vs reference value function",
                  save_path=None):
    """
    Bar chart comparing RMSE for multiple methods.
    """
    labels = list(rmse_dict.keys())
    values = list(rmse_dict.values())
    colors = ['#2980b9', '#e74c3c', '#27ae60', '#8e44ad', '#e67e22']

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=colors[:len(labels)], edgecolor='white')
    ax.bar_label(bars, fmt='%.2f', padding=4, fontsize=10)
    ax.set_ylabel("RMSE"); ax.set_title(title, fontsize=11)
    ax.set_ylim(0, max(values) * 1.25)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
