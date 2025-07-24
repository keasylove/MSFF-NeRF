# Getting Started
* Clone our repo:
```shell
git clone https://github.com/keasylove/MSFF-NeRF.git
```
* Run the setup script to create a conda environment and install the required packages.
```shell
sh conda_setup.sh
```
# Set up datasets
Obtain the required dataset from [here](https://github.com/zju3dv/animatable_nerf/blob/master/INSTALL.md)
# Training
Take the training on `313` as an example. The command lines for training are recorded in [train.sh](train.sh).
```shell
python train_net.py --cfg_file configs/sdf_pdf/anisdf_pdf_313.yaml exp_name msff-nerf_313 resume False
```
# Test
Take the test on `313` as an example. The command lines for test are recorded in [test.sh](test.sh).
  1. Test on training human poses:
  ```shell
  python run.py --type evaluate --cfg_file configs/sdf_pdf/anisdf_pdf_313.yaml exp_name msff-nerf_313 resume True
  ```
  2. Test on unseen human poses:
  ```shell
  python run.py --type evaluate --cfg_file configs/sdf_pdf/anisdf_pdf_313.yaml exp_name msff-nerf_313 resume True test_novel_pose True
  ```
