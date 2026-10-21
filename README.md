# 🚁 Drone Precision Landing on a Dynamic Mobile Platform using Deep Reinforcement Learning (PPO)

This project implements a complete Deep Reinforcement Learning framework to train and fine-tune an autonomous quadrotor drone capable of performing high-precision landings on a moving, dynamic platform within a MuJoCo physics simulation.

<table align="center">
  <tr>
    <td align="center"><img src="static.gif" width="450" alt="Static Platform Landing"><br><em>Landing on a static platform</em></td>
    <td align="center"><img src="mobile.gif" width="450" alt="Dynamic Mobile Platform Landing"><br><em>Landing on a moving platform</em></td>
  </tr>
</table>

---

## ⚡ Quick Start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cd phase3_mujocu_improve_model/src/scripts
python3 test_best_model_mobile_platform.py   # opens the MuJoCo viewer with the trained model
```

---

## 🚀 Project Architecture & Roadmap

The project is structured into three progressive development phases, from theoretical physics validation to robust deployment in simulation:

1. **Phase 1: Theoretical Foundations (`phase1_drone_physics/`)**
   - Custom 3D physics engine built purely in NumPy (Newton-Euler equations, PFD, TDM, Runge-Kutta 4 integrator, and quaternion kinematics).
   - Validation through a custom 3D Matplotlib visualizer and a basic hover control environment.

2. **Phase 2: MuJoCo Discovery (`phase2_mujocu_understanding/`)**
   - Integration of the MuJoCo physics engine via XML markup for accurate contact dynamics, friction, and rigid body collisions.
   - First successful steps with static landing environments and Gymnasium wrappers.

3. **Phase 3: Fine-Tuning, Curriculum Learning & Robustness (`phase3_mujocu_improve_model/`)**
   - Transfer learning from a pre-trained static-platform model.
   - Fine-tuning with Stable-Baselines3 (PPO) using custom evaluation and checkpoint callbacks (`EvalCallback`, `CheckpointCallback`).
   - Domain randomization (platform start position, trajectory amplitude, frequency and phase) for robustness.

---

## 🏗️ Project File Structure

```text
Reinforcement_Learning/
├── documentation/                      # Course notes (PPO, RL basics) and project write-up
├── static.gif, mobile.gif              # Demonstrations
├── Training_stats_*.png                # Training curves
├── requirements.txt
│
├── phase1_drone_physics/               # Phase 1: Custom NumPy physics & visualizer
├── phase2_mujocu_understanding/        # Phase 2: MuJoCo discovery & static landing
│
└── phase3_mujocu_improve_model/        # Phase 3: Mobile landing & PPO fine-tuning
    └── src/
        ├── env/                        # Custom Gymnasium environments
        │   ├── drone_env_static_platform.py
        │   └── drone_env_mobile_platform.py
        ├── xmls/                       # MuJoCo XML models (drone & cart, `final.xml`)
        ├── models/                     # Static baseline models
        ├── models_mobile/              # Mobile fine-tuned models & checkpoints
        ├── tb_logs/                    # TensorBoard logs (static training)
        ├── tb_logs_mobile/             # TensorBoard logs (mobile fine-tuning)
        └── scripts/
            ├── train_static_platform.py
            ├── train_mobile_platform.py
            ├── test_best_model_static_platform.py
            ├── test_best_model_mobile_platform.py
            └── plot_eval_*.py          # Evaluation curves
```

---

## 🧠 Environment Specifications (`DroneLandingEnv`)

- **Observation space (13 dimensions):**
  - Drone position relative to the platform, normalized.
  - Drone velocity relative to the platform, normalized.
  - Drone orientation (quaternion: w, x, y, z).
  - Drone angular velocities, normalized.
- **Action space (4 dimensions):** continuous thrust commands in [-1, 1] for the four rotors, mapped to [0, 1].
- **Dynamic target:** the platform starts at a random position and follows an unpredictable 2D sinusoidal trajectory with random amplitude (1.5–3 m), frequency (0.2–0.45 rad/s) and phase.
- **Precision success criteria** (episode ends with +1000):
  - Horizontal distance `dist_xy` < 0.10 m
  - Vertical distance `dist_z` < 0.05 m
  - Relative velocity < 0.20 m/s (soft touchdown)
  - Orientation alignment `z_alignement` > 0.85
- **Episode limit:** 1000 steps (each step = 5 physics sub-steps).

### Reward shaping

| Term | Effect |
|---|---|
| Progress toward the platform | `+10 × (previous distance − current distance)` |
| Tilt penalty | `−15 × (1 − z_alignement)²` |
| Angular velocity penalty | `−0.1 × ‖ω‖²` |
| Action smoothness | `−0.2 × ‖Δaction‖²` |
| Fast descent (v_z < −1 m/s) | proportional penalty |
| Crash (altitude < 0.1 m or tilt too large) | `−200`, episode ends |
| Successful landing | `+1000`, episode ends |

---

## 💻 Installation & Prerequisites

1. Clone the repository and navigate to the project directory.
2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃‍♂️ Usage

All scripts are run from `phase3_mujocu_improve_model/src/scripts`.

### 1. Training / fine-tuning

Mobile platform (fine-tuning on top of the pre-trained static model):

```bash
python3 train_mobile_platform.py
```

Static platform (baseline training):

```bash
python3 train_static_platform.py
```

### 2. Testing & visualization

Launch the MuJoCo passive viewer and run the trained model over 200 episodes (a trained model is already provided):

```bash
python3 test_best_model_mobile_platform.py
```

Static platform:

```bash
python3 test_best_model_static_platform.py
```

The default model is `src/models_mobile/best_model.zip`. To measure success rate, failure causes and final landing error without rendering, run `python3 evaluate_mobile_platform.py --episodes 200`.

### 3. Monitoring with TensorBoard

```bash
tensorboard --logdir=phase3_mujocu_improve_model/src/tb_logs_mobile
```

---

## 📊 Performance & Results Analysis

- **Success rate (200 random episodes, seed 0):** 75.5 % (95 % CI ± 6 pts). A success gives +1000 only if all precision criteria are met, so this measures precise landings.
- **Landing error on successes:** median 6.5 cm horizontal, 1.7 cm vertical, 14 cm/s relative velocity.
- **Failures (24.5 %):** all end by loss of attitude (tilt above the `z_alignement` limit), none by ground crash or timeout. About two thirds (32/49) first get within 30 cm of the platform without meeting the landing criteria, then climb and flip; the rest (17/49) flip within 3–6 steps of the start without approaching the platform.
- **Robustness:** the agent generalizes to variable platform velocities without overfitting to specific trajectories.

### 1. Static platform baseline

The agent converges toward a stable reward of about 1000, giving a solid baseline policy for precision landing.

<img src="Training_stats_static_platform.png" width="700">

### 2. Dynamic mobile platform fine-tuning

Using transfer learning, the agent adapts to sinusoidal moving targets and recovers from trajectory variations.

<img src="Training_stats_mobile_platform.png" width="700">

---

## 🔭 Limitations & Future Work

- The observation contains the relative velocity but not the platform's absolute velocity or the previous action; adding them should help anticipation.
- The reward has no intermediate term encouraging velocity matching near the platform; the landing bonus is the only signal.
- Evaluation uses few episodes; reporting a success rate over many seeded episodes would be more reliable than the mean reward alone.
