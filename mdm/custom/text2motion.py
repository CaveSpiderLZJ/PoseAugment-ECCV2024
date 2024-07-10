# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
import time
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import shutil
from data_loaders.tensors import collate


def load_dataset(args, max_frames, n_frames):
    # max_frames = 196, n_frames = 120 (6s)
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size,
        num_frames=max_frames, split='test', hml_mode='text_only')
    if args.dataset in ['kit', 'humanml']:
        data.dataset.t2m_dataset.fixed_length = n_frames
    return data


def main():
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    n_frames = min(max_frames, int(args.motion_length*fps))
    is_using_data = not any([args.input_text, args.text_prompt, args.action_file, args.action_name])
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
            'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')
            
    # this block must be called BEFORE the dataset is loaded
    # text_prompt, input_text, action_name, action_file are all ''
    if args.text_prompt != '':
        texts = [args.text_prompt]
        args.num_samples = 1
    elif args.input_text != '':
        assert os.path.exists(args.input_text)
        with open(args.input_text, 'r') as fr:
            texts = fr.readlines()
        texts = [s.replace('\n', '') for s in texts]
        args.num_samples = len(texts)
    elif args.action_name:
        action_text = [args.action_name]
        args.num_samples = 1
    elif args.action_file != '':
        assert os.path.exists(args.action_file)
        with open(args.action_file, 'r') as fr:
            action_text = fr.readlines()
        action_text = [s.replace('\n', '') for s in action_text]
        args.num_samples = len(action_text)

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    # DONE
    data = load_dataset(args, max_frames, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:    # guidance_param == 2.5
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking
    
    if os.path.exists(out_path): shutil.rmtree(out_path)
    os.makedirs(out_path)
    
    for batch_idx, (_, model_kwargs) in enumerate(data):
        
        if len(model_kwargs['y']['text']) != args.batch_size: break
        all_motions, all_lengths, all_text = [], [], []
        
        for rep_i in range(args.num_repetitions):
            print(f'### Sampling [repetitions #{rep_i}]')
            
            # add CFG scale to batch
            if args.guidance_param != 1:
                model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

            sample_fn = diffusion.p_sample_loop

            sample = sample_fn(
                model,
                (args.batch_size, model.njoints, model.nfeats, max_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )   # [xxx, 263, xxx, 196]
            
            # Recover XYZ *positions* from HumanML3D vector representation
            if model.data_rep == 'hml_vec':
                n_joints = 22 if sample.shape[1] == 263 else 21
                sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
                sample = recover_from_ric(sample, n_joints)
                sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

            rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
            rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else \
                model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
            sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep,
                glob=True, translation=True, jointstype='smpl', vertstrans=True, betas=None,
                beta=0, glob_rot=None, get_rotations_back=False)

            if args.unconstrained:
                all_text += ['unconstrained'] * args.num_samples
            else:
                text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
                all_text += model_kwargs['y'][text_key]

            all_motions.append(sample.cpu().numpy())
            all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

            print(f"created {len(all_motions) * args.batch_size} samples")


        all_motions = np.concatenate(all_motions, axis=0)
        all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
        all_text = all_text[:total_num_samples]
        all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

        npy_path = os.path.join(out_path, f'results_{batch_idx}.npy')
        print(f"saving results file to [{npy_path}]")
        np.save(npy_path, {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
            'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
        with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
            fw.write('\n'.join(all_text))
        with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
            fw.write('\n'.join([str(l) for l in all_lengths]))
            

if __name__ == "__main__":
    main()
    