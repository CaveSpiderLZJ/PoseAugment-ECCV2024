# PoseAugment

The official code for our ECCV 2024 paper *PoseAugment: Generative Human Pose Data Augmentation with Physical Plausibility for IMU-based Motion Capture*. In this project, we will provide a detailed implementation of our method, including PoseAugment model training and evaluation. Readers are recommended to read the original paper first.

## Before Started

PoseAugment is a light-weight data augmentation method for human poses. This project will provide three functionalities for users with different needs:

* Choice 1 (the easiest): use the pretrained PoseAugment model (without training) to augment human pose dataset to your needs.
* Choice 2: reproduce or customize the PoseAugment model (involves training), then use it to augment data.
* Choice 3 (the hardest): reproduce our evaluations of PoseAugment against the baseline methods (MotionAug, ACTOR, MDM-T2M, and MDM-M2M).

Choices 1 and 2 do not need to reproduce the baseline methods and their datasets. Readers with these two needs can skip the baseline part (marked as optional in the following instructions).

WARNING: this project involves multiple sub projects, and the code during developing are rapidly changing. A small part of the code may need you to modify manually to work normally. But do not worry, we simplied this procedure as simple as possible. You only need to search for `# NOTE` globally to locate them, and detailed instructions are provided in comments.

## Environment

We run our project on MacOS Sonoma 14.2.1 (M1 Pro chip) and Ubuntu 20.04.6, with Python `3.9`. We haven't tested it on Windows.

First, install the latest packages with pip:

```
pip install numpy torch opencv-python scipy matplotlib tqdm pandas aitviewer qpsolvers cvxopt quadprog clarabel proxsuite tensorboard
```

Then, you should install [rbdl](https://github.com/rbdl/rbdl) with *python wrapper* and *urdf reader addon* to use the *physical optimization* module. You can either add the final built `rbdl-build` dir to your system paths or just place it in the project root dir. We will add it to the system path before importing it. To build the *python wrapper* and *urdf reader addon*, you should turn on these two options in `CMakeLists.txt`:

```
OPTION (RBDL_BUILD_ADDON_URDFREADER "Build the (experimental) urdf reader" ON)
OPTION (RBDL_BUILD_PYTHON_WRAPPER "Build experimental python wrapper" ON)
```

Note: you should better make sure the Python compile version when building the `rbdl` module matches the runtime version, otherwise warnings like this may occur (but it won't cause any exceptions and runs normally in our experiments):

```
<frozen importlib._bootstrap>:228: RuntimeWarning: compile time Python version 3.8 of module 'rbdl' does not match runtime version 3.9
```

## Prepare the Datasets

### Prepare AMASS dataset

We use 18 datasets of the AMASS collection to train the VAE model. The datasets are the same with [TransPose](https://github.com/Xinyu-Yi/TransPose) and be listed in `config.py:General.amass`. You should download the raw datasets from [AMASS](https://amass.is.tue.mpg.de), and place them in `data/dataset_raw/amass` as specified in `config.py:Paths`.

Note: the dataset `BioMotionLab_NTroje` is `BMLrub` in [AMASS](https://amass.is.tue.mpg.de), please refer to [issue](https://github.com/mkocabas/VIBE/issues/78).

### Prepare the DIP-IMU test dataset

This step is identical with [TransPose](https://github.com/Xinyu-Yi/TransPose), you can follow it or download the original DIP-IMU dataset (without normalization) from [here](https://dip.is.tue.mpg.de/index.html). The raw dip dataset should be placed at `data/dataset_raw/dip`.

### Prepare the SMPL body model

Download the SMPL body model from [here](https://smpl.is.tue.mpg.de). You should choose `[Download version 1.0.0 for Python 2.7 (female/male. 10 shape PCs)]`, and copy the models from `smpl/models` into `data/model`, and change `config.py:Paths.smpl_file` to it accordingly. You can either use the male model or famale model.

### (Optional) Prepare the SMPL-X model for visualization

We visualize the human poses using `aitviewer`. You should follow [this instruction](https://github.com/eth-ait/aitviewer), download the SMPL-X model from [here](https://eth-ait.github.io/aitviewer/parametric_human_models/supported_models.html) to configure the SMPL-X model for the `aitviewer`, and modify the model path in `aitvconfig.yaml`. The dir the path in `aitvconfig.yaml` point to should be look like this:

```
- smpl_model/
	- mano/
	- smpl/
	- smplh/
	- smplx/
```

Note that the SMPL-X model used for visualization is not exactly the same with the SMPL model used for generating poses.

### Prepare the physical models

You can follow [PIP](https://github.com/Xinyu-Yi/PIP) to prepare physical models and place them in `data/model` to use the physical optimizer. The models include `physics.urdf`, `plane.obj` and `plane.urdf`. After that, check `config.py:Paths.physics_model_file` accordingly.

## Preprocess the Dataset

We now have the AMASS and DIP raw datasets. To preprocess the datasets for model training, you need to follow `preprocess.py` to preprocess AMASS, DIP train, and DIP test datasets. This step will resample and align the data to 60 Hz, and synthesize the IMU data from raw pose datasets. The preprocessed data will be stored in `data/dataset_work`.

## (Optional) Train the VAE Model

Note: this step is optional unless you would like to train the VAE model on your own datasets or tune it for your own purposes. Otherwise, you can simply use the released VAE model to augment your pose data.

The VAE model trained by ourselves is in `data/model/mvae_weights.pt`. If you want to train VAE yourself, please use `train_mvae.py`. The trained model will be placed in `data/vae/<model_name>/best.pt`.

## Augment Pose Data

You can use `augment.py` to augment the pose and the corresponding IMU datasets using the trained PoseAugment model. The data augmentation process will use the pretrained VAE model and physical optimization in `VAEOptimizer` to optimize the generated poses. Change `n_aug` at the beginning of `augment_data` to modify the augmented data scale. The augmented data is cut to lengths `<= 200` for training the MoCap models. The augmented datasets will be placed in `data/dataset_aug`.

We implemented augmenting AMASS, MotionAug, ACTOR, MDM-T2M, and MDM-M2M datasets. The last four are our baselines in the evaluation. We showcase the usage of PoseAugment in the `augment_amass()` function in `augment.py`. Readers can refer to the code for more details.

Till now, you can use PoseAugment to generate human pose data. **You can stop now if you only want to use PoseAugment to augment your own data.** In the following sections, we will walk you through the evaluation of PoseAugment in our work, including the comparison with baselines, and training TransPose model, which are all optional.

## (Optional) Prepare Baseline Datasets

We compare our method with MotionAug, ACTOR, MDM-T2M, and MDM-M2M by training IMU-based motion capture models using these methods. For MotionAug, we directly use the released dataset. Please refer to `preprocess_motionaug()` and `augment_motionaug()` in `augment.py`. For ACTOR and MDM, we first cutomized their code to generate data. Please refer to `actor/README.md` and `mdm/README.md` for more details. After that, use `preprocess_xxx()` and `augment_xxx()` in `augment.py` to generate the datasets.

## (Optional) Evaluation

After preparing the baseline datasets, we now have the training datasets for TransPose.

You can refer to `train_transpose.py` to run `test_xxx()` functions to test the data augmentation performance by training the TransPose model. In each training process, we will compare PoseAugment with Jitter, and the corresponding baseline method. We will train the models with different data augmentation scale (`n_aug` from 1 to 5) as described in the training details in the paper. The trained models will be placed in `data/transpose`.

After that, you should run `evaluate_xxx()` functions in `evaluate_transpose.py`. They will test the trained TransPose models on different datasets using DIP test data, and store the final results in `data/analysis/`.

Congrats! We have finished the evaluations by now.

## (Optional) Visualization

We also provided a visualization demo in `visualization.py`, which is based on `aitviewer`. You can use it to visualized the augmented poses.
