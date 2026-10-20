import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Paths ---
# Attention : on pointe maintenant vers tb_logs_mobile !
chemin_eval = "../tb_logs_mobile/eval/evaluations.npz"

if not os.path.exists(chemin_eval):
    print(f"❌ File not found: {chemin_eval}")
    exit()

# --- 2. Load Data ---
data = np.load(chemin_eval)
timesteps = data['timesteps']
mean_rewards = np.mean(data['results'], axis=1)
mean_lengths = np.mean(data['ep_lengths'], axis=1)

# --- 3. Create Interactive Subplots ---
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("Mean Reward Evolution", "Mean Episode Length Evolution"),
    vertical_spacing=0.15
)

# Trace 1: Reward
fig.add_trace(
    go.Scatter(
        x=timesteps, y=mean_rewards,
        mode='lines+markers',
        name="Reward",
        line=dict(color='royalblue', width=2),
        marker=dict(size=6)
    ),
    row=1, col=1
)

# Trace 2: Episode Length
fig.add_trace(
    go.Scatter(
        x=timesteps, y=mean_lengths,
        mode='lines+markers',
        name="Steps",
        line=dict(color='firebrick', width=2),
        marker=dict(size=6, symbol='square')
    ),
    row=2, col=1
)

# --- 4. Layout ---
fig.update_layout(
    title_text="PPO Model Performance Analysis - Mobile Platform (Evaluation)",
    height=800,
    showlegend=False,
    template="plotly_white",
    hovermode="x unified"
)

fig.update_xaxes(title_text="Timesteps", row=1, col=1)
fig.update_yaxes(title_text="Mean Reward", range=[0, 1200], row=1, col=1)

fig.update_xaxes(title_text="Timesteps", row=2, col=1)
fig.update_yaxes(title_text="Mean Steps", row=2, col=1)

# --- 5. Export ---
# Sauvegarde sous un nouveau nom
fig.write_image("mes_courbes_mobile.png")
print("✅ Graphique généré avec succès : mes_courbes_mobile.png")