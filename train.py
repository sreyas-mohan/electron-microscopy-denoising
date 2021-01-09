import argparse
import logging
import torch
import numpy as np
import torchvision
import torch.nn.functional as F

from torch.distributions import Poisson
from torch.utils.tensorboard import SummaryWriter

import data
import models
import utils


def main(args):
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    utils.setup_experiment(args)
    utils.init_logging(args)

    # Build data loaders, a model and an optimizer
    train_loader, valid_loader, test_loader = data.build_dataset(args.dataset, args.data_path,
                                                                 batch_size=args.batch_size,
                                                                 image_size=args.image_size,
                                                                 contrast=args.contrast,
                                                                 repeat_train=args.repeat_train, 
                                                                 rotation_aug = not args.no_rotation, 
                                                                 resize_aug = not args.no_resize,
																 generalization_exp = args.generalization_exp,
																 allowed_gen_values = args.allowed_gen_values)
    model = models.build_model(args).to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)
    logging.info(
        f"Built a model consisting of {sum(p.numel() for p in model.parameters()):,} parameters")

    # Track moving average of loss values
    train_meters = {name: utils.RunningAverageMeter(
        0.98) for name in (["train_loss", "train_psnr"])}
    valid_meters = {name: utils.AverageMeter() for name in (["valid_psnr"])}
    writer = SummaryWriter(
        log_dir=args.experiment_dir) if not args.no_visual else None

    global_step = -1
    for epoch in range(args.num_epochs):
        train_bar = utils.ProgressBar(train_loader, epoch)
        for meter in train_meters.values():
            meter.reset()

        for batch_id, sample in enumerate(train_bar):
            model.train()
            global_step += 1
            clean = sample["image"].to(device)
            clean = clean * args.noise_scale
            noisy = Poisson(clean).sample()
            denoised = model(noisy)
            loss = F.mse_loss(denoised, clean, reduction="sum") / len(sample)
            # loss = (-inputs * outputs.log() + outputs).mean()
            # loss = (outputs * ((outputs + 1e-6).log() - (inputs + 1e-6).log()) - outputs).mean()
            # loss = ((outputs - inputs) * outputs.log() - outputs * inputs.log()).mean()

            model.zero_grad()
            loss.backward()
            optimizer.step()

            train_psnr = utils.psnr(clean/args.noise_scale, denoised/args.noise_scale)
            train_meters["train_loss"].update(loss.item())
            train_meters["train_psnr"].update(train_psnr.item())
            train_bar.log(
                dict(
                    **train_meters,
                    lr=optimizer.param_groups[0]["lr"]),
                verbose=True)

            if writer is not None and global_step % args.log_interval == 0:
                writer.add_scalar(
                    "lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("loss/train", loss.item(), global_step)
                writer.add_scalar("psnr/train", train_psnr.item(), global_step)
                gradients = torch.cat(
                    [p.grad.view(-1) for p in model.parameters() if p.grad is not None], dim=0)
                writer.add_histogram("gradients", gradients, global_step)
#                 for idx in range(len(clean)):
#                     image = torch.stack([clean[idx], noisy[idx] / args.noise_scale, denoised[idx]], dim=0)
#                     image = torchvision.utils.make_grid(image.clamp(0, 1), nrow=3, normalize=False)
#                     writer.add_image(f"train_images/{sample['name'][idx]}", image, global_step)

        if epoch % args.valid_interval == 0:
            model.eval()
            for meter in valid_meters.values():
                meter.reset()

            valid_bar = utils.ProgressBar(valid_loader)
            sample_id_to_plot = np.random.choice(
                np.arange(len(valid_loader)), 5, replace=True)
            for sample_id, sample in enumerate(valid_bar):
                with torch.no_grad():
                    clean = sample["image"].to(device)
#                     print(clean.shape)
                    clean = clean * args.noise_scale
                    noisy = noisy = Poisson(clean).sample()
                    denoised = model(noisy)
                    valid_psnr = utils.psnr(clean / args.noise_scale, denoised / args.noise_scale)
                    valid_meters["valid_psnr"].update(valid_psnr.item())

                    if sample_id in sample_id_to_plot:
                        for idx in range(min(len(clean), 3)):
                            image = torch.stack(
                                [clean[idx] / args.noise_scale, noisy[idx] / args.noise_scale, denoised[idx] / args.noise_scale], dim=0)
                            image = torchvision.utils.make_grid(
                                image.clamp(0, 1), nrow=3, normalize=False)
                            writer.add_image(
                                f"valid_images/{sample['name'][idx]}", image, global_step)

            if writer is not None:
                writer.add_scalar(
                    "psnr/valid",
                    valid_meters["valid_psnr"].avg,
                    global_step)
            logging.info(
                train_bar.print(
                    dict(
                        **train_meters,
                        **valid_meters,
                        lr=optimizer.param_groups[0]["lr"])))
            utils.save_checkpoint(
                args,
                global_step,
                model,
                optimizer,
                score=valid_meters["valid_psnr"].avg,
                mode="max")
                                
        scheduler.step(valid_meters["valid_psnr"].avg)
    logging.info(
        f"Done training! Best PSNR {utils.save_checkpoint.best_score:.3f} obtained after step {utils.save_checkpoint.best_step}.")


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Add data arguments
    parser.add_argument(
        "--data-path",
        default="dataset",
        help="path to data directory")
    parser.add_argument(
        "--dataset",
        default="ptceo2",
        help="train dataset name")
    parser.add_argument(
        "--batch-size",
        default=128,
        type=int,
        help="train batch size")
    parser.add_argument(
        "--image-size",
        default=100,
        type=int,
        help="size of the patch")
    parser.add_argument(
        "--contrast",
        default="white",
        help="which contrasts to train on. white-black-intermediate")
    parser.add_argument(
        "--generalization-exp",
        default="None",
        help="which gen exp to run. structure-defect")
    parser.add_argument(
        "--allowed-gen-values",
        default="None",
        help="What are the allowed values. PtNp1-PtNp3")
    parser.add_argument(
        "--repeat-train",
        default=1,
        type=int,
        help="number of times to repeat dataset obj")
    parser.add_argument(
            "--no-rotation",
            action="store_true",
            help="don't use rotation augmentation")
    parser.add_argument(
            "--no-resize",
            action="store_true",
            help="don't resize")

    # Add model arguments
    parser.add_argument("--model", default="dncnn", help="model architecture")
    parser.add_argument(
        "--noise-scale",
        default=1,
        type=int,
        help="multiply the signal by this factor. equivalent to temporally summing frames")

    # Add optimization arguments
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
#     parser.add_argument("--lr-step-size", default=30, type=int, help="step size for learning rate scheduler")
#     parser.add_argument("--lr-gamma", default=0.1, type=float, help="learning rate multiplier")
    parser.add_argument(
        "--num-epochs",
        default=200,
        type=int,
        help="force stop training at specified epoch")
    parser.add_argument(
        "--valid-interval",
        default=1,
        type=int,
        help="evaluate every N epochs")
    parser.add_argument(
        "--save-interval",
        default=1,
        type=int,
        help="save a checkpoint every N steps")

    # Parse twice as model arguments are not known the first time
    parser = utils.add_logging_arguments(parser)
    args, _ = parser.parse_known_args()
    models.MODEL_REGISTRY[args.model].add_args(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
