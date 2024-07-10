# ACTOR

Customized to generate pose data for PoseAugment.

## Preparation

You should follow the official instructions of [ACTOR](https://github.com/Mathux/ACTOR) to prepare the environment, pretrained models, and dataset. We recommend to create a new conda env instead of using the env for PoseAugment.

## Pose Data Generation

You should run

```bash
python -m src.generate.generate_sequences pretrained_models/humanact12/checkpoint_5000.pth.tar --num_samples_per_action 1000 --cpu
```

This command will sample 1000 poses for each of the 12 action classes, among which 1/5 are regarded as the basic data. The other 4/5 will be regarded as the augmented poses by ACTOR.

The generated data will be placed at `pretrained_models/humanact12/generation_smpl.npy`. You should copy this file to `../data/dataset_work/actor` for further preprocessing.

In the root dir of PoseAugment, run the `preprocess_actor()` and `augment_actor()` functions in `augment.py`. `preprocess_actor()` will align the ACTOR generated data with PoseAugment and store them in `data/dataset_aug/actor/actor`. `augment_actor()` will use 1/5 of data to augment other 4/5 of data, and place them in `data/dataset_aug/actor/actor_PoseAugment`.

Note: In our evaluation, we regard 1/5 of the generated data as the basic dataset. Other 4/5 are regarded as the augmented data by ACTOR. We will use PoseAugment on 1/5 of the data to generate other 4/5 for comparison.
