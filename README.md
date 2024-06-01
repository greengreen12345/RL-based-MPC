# RL-based-MPC

The code is developed to implement the RL-based MPC, which control the agent to reach the final goal under a series of sub-goals generated by the model trained by "PPO2MPC" reinforcement learning strategy. The referenced open-source code is from https://github.com/tud-amr/go-mpc.git. The training was conducted on the hardware of NVIDIA GeForce RTX 2060 GPU.


## Installation:

0. Grab the simulation environment from github, initialize submodules, install dependencies and src code:
```
git clone https://github.com/greengreen12345/setup.git
cd setup
./install.sh
```

1. Download the RL-based-MPC source code:
```
git clone https://github.com/greengreen12345/RL-based-MPC.git
```
2. Environment setup:
```
source <setup directory>/venv/bin/activate
cd <RL-based-MPC directoy>
```
   Download "gym_collision_avoidance" package from
```
git clone https://github.com/bbrito/gym-collision-avoidance.git
```
```
 python setup.py install
 pip install nvidia_tensorflow-1.15.4+nv20.12-cp38-cp38-linux_x86_64.whl(Please 
   ensure that the python version is 3.8.16, download from https://developer.download.nvidia.com/compute/redist/nvidia-tensorflow/)
 pip install mujoco_py==2.1.2.14
 pip install gym==0.22.0
```
3. Start training:
```
python train.py
```
4. To test the trained network do:
```
python test.py
```
5. The trained models are located in：
```
<RL-based-MPC directoy>/logs/ppo2-mpc
```

6. To view the logs of one of the training models, such as "PointNMaze-v0_248", run:
```
<RL-based-MPC directoy>/logs/ppo2-mpc/PointNMaze-v0_248/tf_log
```
