import io
import base64
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from django.shortcuts import render
from .mse_algorithm import build_dataset, gradient_descent, compute_mse


PALETTE = {
    'bg':      '#0f1117',
    'surface': '#1a1d27',
    'accent':  '#6c63ff',
    'accent2': '#ff6584',
    'text':    '#e8e8f0',
    'muted':   '#6b7280',
    'grid':    '#2a2d3a',
}

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=PALETTE['bg'], edgecolor='none')
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return data

def _style_ax(ax, title=''):
    ax.set_facecolor(PALETTE['bg'])
    ax.tick_params(colors=PALETTE['muted'], labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE['grid'])
    ax.xaxis.label.set_color(PALETTE['muted'])
    ax.yaxis.label.set_color(PALETTE['muted'])
    ax.grid(color=PALETTE['grid'], linewidth=0.6, linestyle='--')
    if title:
        ax.set_title(title, color=PALETTE['text'], fontsize=11, pad=10, fontweight='bold')

X_raw, X, y_true = build_dataset()
theta_final, cost_history, theta_history = gradient_descent(X, y_true, learning_rate=0.01, iterations=500)
ITERATIONS = list(range(len(cost_history)))

def home(request):
    return render(request, 'home.html')

def formulation(request):
    return render(request, 'formulation.html')

def algorithm_view(request):
    return render(request, 'algorithm.html')

def results(request):
    # 1. Convergence curve
    fig1, ax1 = plt.subplots(figsize=(7, 3.8))
    fig1.patch.set_facecolor(PALETTE['bg'])
    ax1.plot(ITERATIONS, cost_history, color=PALETTE['accent'], linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('MSE Loss')
    _style_ax(ax1, 'Convergence — MSE over Iterations')
    convergence_plot = _fig_to_b64(fig1)

    # 2. Regression fit
    fig2, ax2 = plt.subplots(figsize=(7, 3.8))
    fig2.patch.set_facecolor(PALETTE['bg'])
    ax2.scatter(X_raw, y_true, color=PALETTE['accent2'], alpha=0.55, s=28, label='Data points')
    y_fit = X @ theta_final
    ax2.plot(X_raw, y_fit, color=PALETTE['accent'], linewidth=2.5, label='Fitted line')
    ax2.set_xlabel('X')
    ax2.set_ylabel('y')
    ax2.legend(facecolor=PALETTE['surface'], edgecolor=PALETTE['grid'], labelcolor=PALETTE['text'], fontsize=9)
    _style_ax(ax2, 'Regression Fit — Final Parameters')
    fit_plot = _fig_to_b64(fig2)

    # 3. Cost surface contour
    t0_vals = np.linspace(theta_final[0] - 12, theta_final[0] + 12, 80)
    t1_vals = np.linspace(theta_final[1] - 2,  theta_final[1] + 2,  80)
    T0, T1 = np.meshgrid(t0_vals, t1_vals)
    Z = np.array([
        compute_mse(y_true, np.column_stack([np.ones(len(X_raw)), X_raw]) @ np.array([t0, t1]))
        for t0, t1 in zip(T0.ravel(), T1.ravel())
    ]).reshape(T0.shape)

    th_arr = np.array(theta_history)
    fig3, ax3 = plt.subplots(figsize=(7, 4.2))
    fig3.patch.set_facecolor(PALETTE['bg'])
    cf = ax3.contourf(T0, T1, Z, levels=30, cmap='plasma', alpha=0.85)
    ax3.contour(T0, T1, Z, levels=12, colors='white', linewidths=0.3, alpha=0.3)
    ax3.plot(th_arr[:, 0], th_arr[:, 1], color=PALETTE['accent'], linewidth=1.8, zorder=3)
    ax3.scatter([theta_final[0]], [theta_final[1]], color='white', s=70, zorder=4, label='Optimum')
    cbar = fig3.colorbar(cf, ax=ax3)
    cbar.ax.tick_params(colors=PALETTE['muted'])
    ax3.set_xlabel('θ₀  (intercept)')
    ax3.set_ylabel('θ₁  (slope)')
    ax3.legend(facecolor=PALETTE['surface'], edgecolor=PALETTE['grid'], labelcolor=PALETTE['text'], fontsize=9)
    _style_ax(ax3, 'Cost Surface — Gradient Descent Path')
    contour_plot = _fig_to_b64(fig3)

    final_mse   = cost_history[-1]
    initial_mse = cost_history[0]
    reduction   = (1 - final_mse / initial_mse) * 100

    context = {
        'convergence_plot': convergence_plot,
        'fit_plot':         fit_plot,
        'contour_plot':     contour_plot,
        'theta0':           round(float(theta_final[0]), 4),
        'theta1':           round(float(theta_final[1]), 4),
        'final_mse':        round(final_mse, 4),
        'initial_mse':      round(initial_mse, 4),
        'reduction':        round(reduction, 2),
        'iterations':       len(cost_history),
        'lr':               0.01,
        'n_samples':        len(y_true),
    }
    return render(request, 'results.html', context)

def discussion(request):
    return render(request, 'discussion.html')
