# SBTO
Sampling Based Trajectory Optimization (SBTO)

## Dependencies

- python=3.12.11
- numpy=2.3.4
- mujoco=3.3.7
- numba=0.62.1
- scipy=1.16.2
- matplotlib=3.10.6
- pyyaml=6.0.3
- opencv-python=4.12.0
- hydra-core==1.3.2

### Install

#### Environment
```bash
https://github.com/Atarilab/sbto.git
cd sbto
conda create -n sbto python=3.12.11
conda activate sbto
pip install -y --upgrade pip mujoco==3.3.7 numba==0.62.1 scipy==1.16.2 matplotlib==3.10.6 pyyaml==6.0.3 hydra-core==1.3.2 seaborn==0.13.2
conda install -c conda-forge opencv
pip install -e .
```

#### OmniRetarget

Download robot-object motion references from Omniretarget dataset.

```bash
mkdir datasets && cd datasets
wget "https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset/resolve/main/robot-object.zip"
unizip robot-object.zip
```

### Test
```python
python3 sbto/main.py task=g1/robot_object_ref warm_start=incremental task.cfg_ref.motion_path="datasets/robot-object/datasets/robot-object/sub10_largebox_000_original.npz"
```