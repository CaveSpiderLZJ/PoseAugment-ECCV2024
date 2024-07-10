# MDM: Human Motion Diffusion Model

Customized to generate pose data for PoseAugment.

## Preparation

You should follow the official instructions of [MDM](https://github.com/GuyTevet/motion-diffusion-model) to prepare the environment, pretrained models, and dataset. Since the code is based on python 3.7, we recommend to create a new conda env instead of using the env for PoseAugment. We only need to generate pose data in **test2motion** style using the **HumanML3D** datasets. So, you can only prepare these parts.

## Pose Data Generation

The generation process involves MDM-T2M and MDM-M2M two parts.

### MDM-T2M

You should run

```bash
python -m custom.text2motion --model_path ./save/humanml_trans_enc_512/model000200000.pt --num_samples 100 --num_repetitions 5 --output_dir data/mdm --batch_size 100
```

and

```bash
python -m custom.joint2smpl
```

to generate T2M pose data. The first command will use all text descriptions in HumanML3D test dataset to generate poses. Each pose will be sampled repeatedly for 5 times. Due to the limitation of HumanML3D, the generated poses only contain joint positions. The second command will convert joint positions to SMPL joint angles using Inverse Kinematics. This process is extremely time consuming and it may take days to finish.

The generated data would be placed in `data/mdm`. You should copy them to `../data/dataset_work/mdm`, and run the `preprocess_mdm()` and `augment_mdm()` functions in `augment.py` in the root dir of PoseAugment. `preprocess_mdm()` will convert the mdm data to the data structure of PoseAugment and place them in `data/dataset_aug/mdm/mdm`. `augment_mdm()` will use PoseAugment to augment 4x of one repetition of the mdm data, and place them in `data/dataset_aug/mdm/mdm_PoseAugment`.

### MDM-M2M

MDM-M2M is a modification of MDM-T2M, where we use the diffusion model to denoise partially noised data to generate poses in motion-to-motion manner.

You shoud run

```bash
python -m custom.motion2motion --model_path ./save/humanml_trans_enc_512/model000200000.pt --num_samples 100 --num_repetitions 5 --output_dir data/mdm_m2m --batch_size 100

```

to generate the motions. After that, change the `data_dir` and `save_dir` accordingly in `custom/joint2smpl.py` (see `# NOTE`), and run `python -m custom.joint2smpl` to convert motions to SMPL format.

The following steps are similar to MDM-T2M. The generated data would be placed in `data/mdm_m2m`. Copy them to  `../data/dataset_work/mdm_m2m` and run `preprocess_mdm_m2m()` and `augment_mdm_m2m()` in `../augment.py`.