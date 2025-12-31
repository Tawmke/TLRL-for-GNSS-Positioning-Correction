# TLRL-for-GNSS-Positioning-Correction
Paper: "Transform Learning based Reinforcement Learning Algorithm and Application in Improving GNSS Positioning"

## Project structure
```
project-root/
├── README.md     # this file
├── env/          # environment files
├── model/        # some DRL files
├── sb3_161/      # stable_baseline3 model files
├── funcs/        # TLRL models and functions
├── ...
DTL_control4GNSS_preNet_v2.py                    # control training with Q-learning (discrete action)
DTL_control4GNSS_preNet_PPO.py                   # control training with PPO (continuous action)
rl_control_losposAcat_gpsl1.py                   # control training with LSTMPPO (continuous action)
rl_control_POS_discrete.py                       # control training with A2C (discrete action)
TL4GNSS_par_process_preNet_pkl_113.py            # plot results
TLRL4GNSS_pretrain.py                            # TLRL model pretraining
environment.yml                                  # environment requirement
trainX_data.pkl                                  # Original observation data
trainY_data.pkl                                  # Value data
```
## Quick Start
### Step 1: pre-training stage
Creat the environment with Anaconda using `environment.yml`. 
Run `TLRL4GNSS_pretrain.py` to train a transform dictionary and a action-value weight. It is noted that the project only provides the GSDC2022 data, because the GZ-GNSS data are not publicly available due to confidentiality agreements but are available from the corresponding author on reasonable request. You can set:
```python
datasets= 'GSDC2022urban' # 'GSDC2022urban'/ 'TDurbancanyon'
dir_path = '/mnt/sdb/home/tangjh/DLRL4GNSSpos/' # Set your project path
```
After running, parameters of pretrained models are saved in the folder `matlab_Model`.
### Step 2: Control training stage
Run `DTL_control4GNSS_preNet_v2.py` for the discrete action setting or `DTL_control4GNSS_preNet_PPO.py` for the continuous action setting. You can set:
```python
datasets= ['GSDC2022urban'] # environment: 'GSDC2022urban' 'GSDC2022highway' 'TDurbancanyon'
reg_method1 = 'L0' # sparse regulerizer: 'L0', 'L1' 'wo_sparse'
```
It is noted that you should set the correct paths need to be set to read the data (download in [Dropbox](https://www.dropbox.com/scl/fo/zk8tuop7whgtpojypldn9/ANG-oa8F2xDxEUvOPVsUyoQ?rlkey=q0awgo0tpctkhnex4uxxkzufo&st=j9i4og94&dl=0)) in [GSDC_2022_environment](./env/GSDC_los_env.py):
```python
dir_path = '/mnt/sdb/home/tangjh/smartphone-decimeter-2022/'#'/home/tangjianhao/smartphone-decimeter-2022/' # '/home/tangjh/smartphone-decimeter-2022/''D:/jianhao/smartphone-decimeter-2022/'
project_path = '/mnt/sdb/home/tangjh/DLRL4GNSSpos/'
```
If you want to test the noise robustness of the model, you can set `noisedB = None # 10,20,..` in [control training](./funcs/control_model_DTL_preNet.py):
### Step 3: Plot results
Run `TL4GNSS_par_process_preNet_pkl_113.py` to plot results and select parameters.



