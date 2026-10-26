# RAVEN
### Latency-Aware Predictive Digital Twin for Humanoid Robotic Cinematography

**RAVEN** is a robotics research framework investigating autonomous cinematography using the **Unitree G1 humanoid robot**, **NVIDIA Isaac Sim**, RGB-D perception, and model-based reinforcement learning.

The project aims to develop a predictive control system capable of tracking moving subjects, maintaining stable cinematic framing, and compensating for perception and actuation latency within a physics-based digital twin.

RAVEN is being developed as part of PhD research at the **University of York, Department of Computer Science**.

> **Research status:** Active development. G1 camera tracking, physical simulation, neural dynamics modelling and iterative actor–critic training have been demonstrated in simulation. Independent closed-loop evaluation of the latest learned policy and physical robot deployment remain future milestones.

---

## Research Objectives

RAVEN investigates how learned predictive control can improve humanoid robotic cinematography, particularly under uncertain dynamics and system latency.

The principal objectives are:

- Develop a physics-based digital twin of a humanoid robotic camera operator.
- Implement RGB-D perception for actor detection and tracking.
- Learn robot and camera dynamics from simulation experience.
- Train control policies for accurate, smooth and stable camera movements.
- Investigate predictive compensation for sensor and actuation latency.
- Evaluate learned controllers against conventional control approaches.
- Establish a foundation for eventual sim-to-real transfer to a physical Unitree G1.

## System Architecture

RAVEN combines several interconnected components:

| Component | Description |
|---|---|
| **Unitree G1** | Humanoid robot model providing articulated camera-control movements |
| **NVIDIA Isaac Sim / Isaac Lab** | Physics-based digital twin and simulation environment |
| **RGB-D Perception** | Actor detection, depth sensing and visual tracking |
| **Baseline Controller** | Conventional actor-following camera controller |
| **Neural World Model** | Predicts system dynamics and the consequences of control actions |
| **Actor–Critic Policy** | Learns control actions through model-based optimisation |
| **Experience Replay** | Reuses historical and newly collected simulation transitions |
| **Evaluation Pipeline** | Measures tracking accuracy, stability and policy performance |

### Learning Architecture

RAVEN currently uses a **custom SuperMimic-inspired model-based reinforcement learning architecture**.

The learning process consists of:

1. **Experience collection:** G1 performs camera tracking within Isaac Sim while recording robot states, actions and sensor observations.
2. **Dynamics learning:** A neural World Model learns to predict system behaviour from recorded physical transitions.
3. **Policy optimisation:** An Actor–Critic network learns control actions through differentiable imagined trajectories.
4. **Policy evaluation:** Candidate policies are assessed against tracking and stability constraints.
5. **Iterative refinement:** Accepted policies are reused in subsequent simulation experience-collection cycles.

The implementation is inspired by concepts associated with SuperMimic but is **not a reproduction or complete implementation of the original framework**.

## Current Capabilities

### Humanoid Camera Control

- Simulated Unitree G1 with articulated joint control.
- Physical tripod-pan interaction driven by G1 movements.
- RGB-D-based actor tracking.
- Constrained camera pan operation.
- Temporary physics-assisted grasp for experimental stability.
- Camera calibration and tracking measurements.

### Learning and Training

- Neural dynamics prediction using a learned World Model.
- Actor–Critic policy optimisation.
- Differentiable multi-step prediction.
- Simulation-based experience collection at 120 Hz.
- Cumulative experience replay across learning cycles.
- Actor-motion curriculum including sinusoidal, triangular, dwell and multisine trajectories.
- Checkpoint evaluation, selection and training resumption.
- Large-scale policy optimisation experiments.

## Experimental Progress

### Physics-Based Actor Tracking

RAVEN has demonstrated actor tracking through a simulated G1 operating a camera tripod.

The system has completed tracking episodes with 351/351 valid RGB-D detections and bidirectional camera-pan control.

Following improvements to perception and trajectory generation, curriculum experiments achieved **0% pan-target saturation** across multiple actor-motion profiles.

### Iterative Learning

Four simulation-learning cycles have been completed, demonstrating repeated experience collection, World Model updates and Actor–Critic training.

The World Model has shown reductions in validation loss across successive experiments, while candidate policy checkpoints are retained or rejected according to configured performance criteria.

### 5,000-Update Actor–Critic Experiment

A training experiment comprising **5,000 actor updates and 5,000 critic updates** was completed using accumulated simulation experience.

The strongest accepted checkpoint occurred at **iteration 1,400**.

| Metric | Starting Policy | Selected Policy |
|---|---:|---:|
| Model-predicted pan error | 0.20580° | 0.14248° |
| Model-predicted tilt deviation | 0.03639° | 0.01422° |
| Evaluation score | 0.28101 | 0.17453 |

This represents:

- **30.8% reduction** in model-predicted pan error.
- **60.9% reduction** in model-predicted tilt deviation.
- **37.9% reduction** in the combined evaluation score.

Performance deteriorated after the selected iteration, highlighting the importance of checkpoint selection, training stability and independent evaluation.

**These improvements are measured using the learned dynamics model. They do not yet establish equivalent performance improvements in independent closed-loop Isaac Sim trials.**

## Technology Stack

| Technology | Application |
|---|---|
| Python | Simulation, training and evaluation |
| PyTorch | Neural networks and gradient-based optimisation |
| NVIDIA Isaac Sim | Robotics simulation and physics |
| NVIDIA Isaac Lab | Robotics research environment |
| USD / PhysX | Scene representation and physical interactions |
| RGB-D Cameras | Visual perception and depth sensing |
| Git / GitHub | Source control and research development |

## Repository Structure

The repository is organised around simulation, learning and evaluation scripts.

```text
raven-g1/
├── scripts/
│   ├── raven_g1_actor_tracking.py
│   ├── raven_supermimic_world_model_v2.py
│   ├── raven_supermimic_actor_critic_v1_1.py
│   ├── raven_supermimic_experience_collect_v2.py
│   ├── raven_supermimic_training_v2.py
│   ├── raven_supermimic_policy_train_v3.py
│   └── raven_curriculum_diagnostics.py
│
├── raven_training_results/       # Local experiment outputs
├── raven_tracking_results/       # Local tracking measurements
└── README.md
```

*The structure above highlights the principal research components rather than listing every repository file. Training datasets and model checkpoints may be stored locally and excluded from Git.*

## Running the Learning Pipeline

The current experimental workflow is designed for a configured Isaac Sim / Isaac Lab environment.

To execute the iterative learning pipeline:

```powershell
python .\scripts\raven_supermimic_training_v2.py `
    --cycles 4 `
    --resume `
    --headless `
    --bearing-source pixel
```

To inspect the collected curriculum experience:

```powershell
python .\scripts\raven_curriculum_diagnostics.py `
    --root .\raven_training_results\supermimic_online_v2
```

Training requires the appropriate Python environment, dependencies, simulation assets, and existing model checkpoints where applicable. These commands are examples of the research workflow, not a complete clean-install procedure.

## Research Roadmap

- [x] Unitree G1 digital twin in Isaac Sim
- [x] RGB-D actor detection and tracking
- [x] Physical tripod-pan interaction
- [x] Baseline camera-control system
- [x] Neural World Model
- [x] SuperMimic-inspired Actor–Critic learning
- [x] Iterative simulation experience collection
- [x] Diverse actor-motion curriculum
- [x] 5,000-update policy training experiment
- [ ] Independent closed-loop evaluation on unseen trajectories
- [ ] Improved long-horizon optimisation stability
- [ ] Image-space cinematic framing objectives
- [ ] Explicit observation and actuation latency modelling
- [ ] Predictive latency compensation
- [ ] Robust grasp and contact modelling
- [ ] Supervised physical G1 validation
- [ ] Autonomous real-world cinematography

## Research Context

**Project:** RAVEN — A Latency-Aware Predictive Digital Twin Framework for Humanoid Robotic Cinematography in Virtual Production

**Institution:** University of York  
**Department:** Computer Science  
**Research Area:** Humanoid Robotics, Model-Based Reinforcement Learning, Digital Twins and Virtual Production

RAVEN investigates how learned predictive models and adaptive control can enable humanoid robots to perform increasingly autonomous cinematographic tasks.

## Disclaimer

RAVEN is an experimental academic research project.

Current learned policies are intended for simulation research and are **not yet validated for deployment on physical robotic hardware**. Physical deployment will require additional safety engineering, hardware calibration, contact validation and supervised testing.
