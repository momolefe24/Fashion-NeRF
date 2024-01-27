# File based on https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py

import argparse
import logging
import os
import shutil
from pathlib import Path
import subprocess
PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
# command = "source ~/.bashrc && conda activate ladi-vton && conda env list | grep ladi"
# subprocess.run(command, shell=True, executable='/bin/bash')
import diffusers
import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")
from tqdm.auto import tqdm

from VITON.Parser_Based.Ladi_VTON.src.dataset.dresscode import DressCodeDataset
from VITON.Parser_Based.Ladi_VTON.src.dataset.vitonhd import VitonHDDataset
from VITON.Parser_Based.Ladi_VTON.src.models.AutoencoderKL import AutoencoderKL
from VITON.Parser_Based.Ladi_VTON.src.models.emasc import EMASC
from VITON.Parser_Based.Ladi_VTON.src.utils.data_utils import mask_features
from VITON.Parser_Based.Ladi_VTON.src.utils.image_from_pipe import extract_save_vae_images
from VITON.Parser_Based.Ladi_VTON.src.utils.set_seeds import set_seed
from VITON.Parser_Based.Ladi_VTON.src.utils.val_metrics import compute_metrics
from VITON.Parser_Based.Ladi_VTON.src.utils.vgg_loss import VGGLoss


logger = get_logger(__name__, log_level="INFO")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB_START_METHOD"] = "thread"

fix = lambda path: os.path.normpath(path)

   
def get_wandb_image(image, wandb):
    if image.max() <= 1.0:
        image = image*255 
    image_numpy = image.permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
    return wandb.Image(image_numpy)     


def print_log(log_path, content, to_print=True):
    import os
    if os.path.exists(log_path):
        with open(log_path, 'a') as f:
            f.write(content)
            f.write('\n')

        if to_print:
            print(content)

def get_root_experiment_runs(root_opt):
    root_opt.experiment_run = root_opt.experiment_run.format(root_opt.experiment_number, root_opt.run_number)
    root_opt.experiment_from_run = root_opt.experiment_from_run.format(root_opt.experiment_from_number, root_opt.run_from_number)
    root_opt.emasc_experiment_from_run = root_opt.emasc_experiment_from_run.format(root_opt.emasc_experiment_from_number, root_opt.emasc_run_from_number)
    return root_opt

def get_root_opt_experiment_dir(root_opt):
    root_opt.rail_dir = root_opt.rail_dir.format(root_opt.dataset_name, root_opt.res, root_opt.datamode)    
    root_opt.original_dir = root_opt.original_dir.format(root_opt.dataset_name, root_opt.res, root_opt.datamode)
    if root_opt.res == 'low_res':
        root_opt.original_dir = root_opt.original_dir.replace(root_opt.res, os.path.join(root_opt.res, root_opt.low_res_dataset_name))
    # Current model
    root_opt.this_viton_save_to_dir = os.path.join(root_opt.this_viton_save_to_dir, root_opt.VITON_Model)
    root_opt.this_viton_load_from_dir = root_opt.this_viton_load_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.this_viton_load_from_dir)
    root_opt.this_viton_load_from_dir = os.path.join(root_opt.this_viton_load_from_dir, root_opt.VITON_Model)
    
    # TOCG
    root_opt.emasc_experiment_from_dir = root_opt.emasc_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.emasc_load_from_model)
    root_opt.emasc_experiment_from_dir = os.path.join(root_opt.emasc_experiment_from_dir, root_opt.VITON_Model)
    
    return root_opt

def get_root_opt_results_dir(parser, root_opt):
    root_opt.transforms_dir = root_opt.transforms_dir.format(root_opt.dataset_name)
    parser.tensorboard_dir = parser.tensorboard_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.results_dir = parser.results_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    return parser, root_opt

def copy_root_opt_to_opt(parser, root_opt):
    parser.display_count = root_opt.display_count
    parser.cuda = root_opt.cuda
    parser.device = int(root_opt.device)
    parser.dataset_name = root_opt.dataset_name
    parser.warp_load_from_model = root_opt.warp_load_from_model
    parser.load_last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
    parser.run_wandb = root_opt.run_wandb
    parser.viton_batch_size = root_opt.viton_batch_size
    parser.save_period = root_opt.save_period
    parser.print_step = root_opt.print_step
    parser.niter = root_opt.niter
    parser.niter_decay = root_opt.niter_decay
    parser.VITON_Type = root_opt.VITON_Type
    parser.VITON_selection_dir = parser.VITON_selection_dir.format(parser.VITON_Type, parser.VITON_Name)
    return parser

def get_root_opt_checkpoint_dir(opt, root_opt):
    last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
    sort_digit = lambda name: int(name.split('_')[-1].split('.')[0])
    # ================================= TOCG =================================
    opt.emasc_save_step_checkpoint_dir = opt.emasc_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.emasc_save_step_checkpoint_dir = fix(opt.emasc_save_step_checkpoint_dir)
    opt.emasc_save_step_checkpoint = os.path.join(opt.emasc_save_step_checkpoint_dir, opt.emasc_save_step_checkpoint)
    
    opt.emasc_save_final_checkpoint_dir = opt.emasc_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.emasc_save_final_checkpoint_dir = fix(opt.emasc_save_final_checkpoint_dir)
    opt.emasc_save_final_checkpoint = os.path.join(opt.emasc_save_final_checkpoint_dir, opt.emasc_save_final_checkpoint)
    
    opt.emasc_load_final_checkpoint_dir = opt.emasc_load_final_checkpoint_dir.format(root_opt.vgg_experiment_from_run, root_opt.emasc_experiment_from_dir)
    opt.emasc_load_final_checkpoint_dir = fix(opt.emasc_load_final_checkpoint_dir)
    opt.emasc_load_final_checkpoint = os.path.join(opt.emasc_load_final_checkpoint_dir, opt.emasc_load_final_checkpoint)

    if not last_step:
        opt.emasc_load_step_checkpoint_dir = opt.emasc_load_step_checkpoint_dir.format(root_opt.emasc_experiment_from_run, root_opt.emasc_experiment_from_dir)
    else:
        opt.emasc_load_step_checkpoint_dir = opt.emasc_load_step_checkpoint_dir.format(root_opt.emasc_experiment_from_run, root_opt.this_viton_save_to_dir)
    opt.emasc_load_step_checkpoint_dir = fix(opt.emasc_load_step_checkpoint_dir)
    if not last_step:
        opt.emasc_load_step_checkpoint = os.path.join(opt.emasc_load_step_checkpoint_dir, opt.emasc_load_step_checkpoint)
    else:
        if os.path.isdir(opt.emasc_load_step_checkpoint_dir.format(root_opt.emasc_experiment_from_run, root_opt.this_viton_save_to_dir)):
            os_list = os.listdir(opt.emasc_load_step_checkpoint_dir.format(root_opt.emasc_experiment_from_run, root_opt.this_viton_save_to_dir))
            os_list = [string for string in os_list if "tps" in string]
            last_step = sorted(os_list, key=sort_digit)[-1]
            opt.emasc_load_step_checkpoint = os.path.join(opt.emasc_load_step_checkpoint_dir, last_step)
        
    return opt

def get_root_opt_results_dir(parser, root_opt):
    root_opt.transforms_dir = root_opt.transforms_dir.format(root_opt.dataset_name)
    parser.tensorboard_dir = parser.tensorboard_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.results_dir = parser.results_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    return parser, root_opt

def copy_root_opt_to_opt(parser, root_opt):
    parser.display_count = root_opt.display_count
    parser.cuda = root_opt.cuda
    parser.device = int(root_opt.device)
    parser.dataset_name = root_opt.dataset_name
    parser.warp_load_from_model = root_opt.warp_load_from_model
    parser.load_last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
    parser.run_wandb = root_opt.run_wandb
    parser.viton_batch_size = root_opt.viton_batch_size
    parser.save_period = root_opt.save_period
    parser.print_step = root_opt.print_step
    parser.niter = root_opt.niter
    parser.niter_decay = root_opt.niter_decay
    parser.VITON_Type = root_opt.VITON_Type
    parser.VITON_selection_dir = parser.VITON_selection_dir.format(parser.VITON_Type, parser.VITON_Name)
    return parser

def process_opt(opt, root_opt):
    parser = opt
    parser = argparse.Namespace(**parser)
    root_opt = get_root_experiment_runs(root_opt)
    root_opt = get_root_opt_experiment_dir(root_opt)
    parser = get_root_opt_checkpoint_dir(parser, root_opt)
    parser, root_opt = get_root_opt_results_dir(parser, root_opt)    
    parser = copy_root_opt_to_opt(parser, root_opt)
    return parser, root_opt

import subprocess
def train_emasc_(opt, root_opt, run_wandb=False):
    # os.system('conda activate ladi-vton && conda env list | grep ladi')
    opt,root_opt = process_opt(opt, root_opt)
    if run_wandb:
        import wandb
        wandb.login()
        wandb.init(
        project="Fashion-NeRF",
        entity='prime_lab',
        notes=f"{root_opt.question}",
        tags=[f"{root_opt.experiment_run}"],
        config=vars(opt)
        )
    else:
        wandb = None
    if wandb is not None:
        temp_opt = vars(opt)
        temp_opt['wandb_name'] = wandb.run.name
        opt = argparse.Namespace(**temp_opt)
    _train_emasc_(opt, root_opt,  wandb=wandb)

import yaml
def _train_emasc_(args, root_opt,  wandb=None):
    # Setup accelerator.
    if not os.path.exists(args.tensorboard_dir):
        os.makedirs(args.tensorboard_dir)
    # board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.VITON_Model))
    with open(os.path.join(root_opt.experiment_run_yaml, f"{root_opt.experiment_run.replace('/','_')}_{root_opt.opt_vton_yaml}"), 'w') as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)
    # Directories
    log_path = os.path.join(args.results_dir, 'log.txt')
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
        with open(log_path, 'w') as file:
            file.write(f"Hello, this is experiment {root_opt.experiment_run} \n")
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load VAE model.
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae.eval()

    # Define EMASC model.
    in_feature_channels = [128, 128, 128, 256, 512]
    out_feature_channels = [128, 256, 512, 512, 512]
    int_layers = [1, 2, 3, 4, 5]

    emasc = EMASC(in_feature_channels,
                  out_feature_channels,
                  kernel_size=args.emasc_kernel,
                  padding=args.emasc_padding,
                  stride=1,
                  type=args.emasc_type)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        emasc.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=eval(args.adam_epsilon),
    )
    dataroot = os.path.join(root_opt.root_dir, root_opt.original_dir)
    # Define datasets and dataloaders.
    if args.dataset == "dresscode":
        train_dataset = DressCodeDataset(
            dataroot_path=args.dresscode_dataroot,
            phase='train',
            order='paired',
            radius=5,
            category=['dresses', 'upper_body', 'lower_body'],
            size=(512, 384),
            outputlist=('image', 'pose_map', 'inpaint_mask', 'im_mask', 'category', 'im_name')
        )

        test_dataset = DressCodeDataset(
            dataroot_path=args.dresscode_dataroot,
            phase='test',
            order=args.test_order,
            radius=5,
            category=['dresses', 'upper_body', 'lower_body'],
            size=(512, 384),
            outputlist=('image', 'pose_map', 'inpaint_mask', 'im_mask', 'category', 'im_name')
        )
    elif args.dataset == "vitonhd":
        train_dataset = VitonHDDataset(
            args,root_opt,
            dataroot_path=dataroot,
            phase='train',
            order='paired',
            radius=5,
            size=(512, 384),
            outputlist=('image', 'pose_map', 'inpaint_mask', 'im_mask', 'category', 'im_name')
        )

        test_dataset = VitonHDDataset(
            args,root_opt,
            dataroot_path=dataroot,
            phase='test',
            order=args.test_order,
            radius=5,
            size=(512, 384),
            outputlist=('image', 'pose_map', 'inpaint_mask', 'im_mask', 'category', 'im_name')
        )
    else:
        raise NotImplementedError("Dataset not implemented")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers_test,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Define VGG loss when vgg_weight > 0
    if args.vgg_weight > 0:
        criterion_vgg = VGGLoss()
    else:
        criterion_vgg = None

    # Prepare everything with our `accelerator`.
    emasc, vae, train_dataloader, lr_scheduler, test_dataloader, criterion_vgg = accelerator.prepare(
        emasc, vae, train_dataloader, lr_scheduler, test_dataloader, criterion_vgg)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("LaDI_VTON_EMASC", config=vars(args),
                                  init_kwargs={"wandb": {"name": os.path.basename(args.results_dir)}})
        if args.report_to == 'wandb':
            wandb_tracker = accelerator.get_tracker("wandb")
            wandb_tracker.name = os.path.basename(args.results_dir)
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        try:
            if args.resume_from_checkpoint != "latest":
                path = os.path.basename(os.path.join("checkpoint", args.resume_from_checkpoint))
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(os.path.join(args.results_dir, "checkpoint"))
                dirs = [d for d in dirs if d.startswith("emasc")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1]
            accelerator.print(f"Resuming from checkpoint {path}")

            accelerator.load_state(os.path.join(args.results_dir, "checkpoint", path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = global_step % num_update_steps_per_epoch
        except Exception as e:
            print("Failed to load checkpoint, training from scratch:")
            print(e)
            resume_step = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        emasc.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(emasc):
                # Convert images to latent space
                with torch.no_grad():
                    # take latents from the encoded image and intermediate features from the encoded masked image
                    posterior_im, _ = vae.encode(batch["image"])
                    _, intermediate_features = vae.encode(batch["im_mask"])

                    intermediate_features = [intermediate_features[i] for i in int_layers]

                # Use EMASC to process the intermediate features
                processed_intermediate_features = emasc(intermediate_features)

                # Mask the features
                processed_intermediate_features = mask_features(processed_intermediate_features, batch["inpaint_mask"])

                # Decode the image from the latent space use the EMASC module
                latents = posterior_im.latent_dist.sample()
                reconstructed_image = vae.decode(z=latents,
                                                 intermediate_features=processed_intermediate_features,
                                                 int_layers=int_layers).sample

                # Compute the loss
                with accelerator.autocast():
                    loss = F.l1_loss(reconstructed_image, batch["image"], reduction="mean")
                    if criterion_vgg:
                        loss += args.vgg_weight * (criterion_vgg(reconstructed_image, batch["image"]))

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate and update gradients
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(emasc.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # Save checkpoint every checkpointing_steps steps
                if global_step % args.checkpointing_steps == 0:
                    # Validation Step
                    emasc.eval()
                    if accelerator.is_main_process:
                        # Save model checkpoint
                        # os.makedirs(os.path.join(args.results_dir, "checkpoint"), exist_ok=True)
                        accelerator_state_path = args.emasc_vitonhd_save_step_checkpoint_dir % global_step
                        accelerator.save_state(accelerator_state_path)

                        # Unwrap the EMASC model
                        unwrapped_emasc = accelerator.unwrap_model(emasc, keep_fp32_wrapper=True)
                        with torch.no_grad():
                            # Extract the images
                            with torch.cuda.amp.autocast():
                                extract_save_vae_images(vae, unwrapped_emasc, test_dataloader, int_layers,
                                                        args.results_dir, args.test_order,
                                                        save_name=f"imgs_step_{global_step}",
                                                        emasc_type=args.emasc_type)

                            # Compute the metrics
                            metrics = compute_metrics(
                                os.path.join(args.results_dir, f"imgs_step_{global_step}_{args.test_order}"),
                                args.test_order, args.dataset, 'all', ['all'], args.dresscode_dataroot,
                                dataroot)

                            print(metrics, flush=True)
                            accelerator.log(metrics, step=global_step)

                            # Delete the previous checkpoint
                            dirs = os.listdir(args.emasc_vitonhd_save_step_checkpoint_dir)
                            dirs = [d for d in dirs if d.startswith("emasc")]
                            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                            # try:
                            #     path = dirs[-2]
                            #     shutil.rmtree(args.emasc_vitonhd_save_step_checkpoint_dir, ignore_errors=True)
                            # except:
                            #     print("No checkpoint to delete")

                            # Save EMASC model
                            emasc_path = os.path.join(args.emasc_vitonhd_save_step_checkpoint_dir, f"emasc_{global_step}.pth")
                            accelerator.save(unwrapped_emasc.state_dict(), emasc_path)
                            del unwrapped_emasc

                        emasc.train()
                break
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # End of training
    accelerator.wait_for_everyone()
    accelerator.end_training()