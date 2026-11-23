"""Train the 3VP avatar.

Stage 1 (default): train the conditional NeRF with the deformation-based correctives.
Stage 2 (--trainMapping): train the mapping network on top of the frozen stage-1 model.
"""

import os
from functools import partial

import numpy as np
import torch
import torch.optim as optim

from loss import huber_loss
from nerf.network import MappingNetwork, NeRFNetworkCond
from nerf.options import parse_args
from nerf.provider import NeRFDatasetCondOr
from nerf.utils import PSNRMeter, Trainer, seed_everything


def build_model(cls, opt):
    return cls(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
    )


def main():
    opt = parse_args("train")
    print(opt)
    seed_everything(opt.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(opt.gpu_mem_fraction)

    # written only after the last epoch: the BVH library exits with status 0 on CUDA errors
    done_file = os.path.join(opt.workspace, "stage2.done" if opt.trainMapping else "stage1.done")
    if os.path.exists(done_file):
        os.remove(done_file)

    criterion = partial(huber_loss, reduction="none")
    # same construction order as the original code (keeps the random initialisation identical)
    model = build_model(NeRFNetworkCond, opt)
    model2 = build_model(NeRFNetworkCond, opt)
    if opt.trainMapping:
        model = build_model(MappingNetwork, opt)

    optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
    # decay to 0.1 * init_lr at the last iteration
    scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda it: 0.1 ** min(it / opt.iters, 1))
    train_loader = NeRFDatasetCondOr(opt, device=device, type="train", freeze=opt.freeze).dataloader()
    max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)

    trainer = Trainer(
        "ngp",
        opt,
        model,
        device=device,
        workspace=opt.workspace,
        optimizer=optimizer,
        criterion=criterion,
        ema_decay=0.95,
        fp16=opt.fp16,
        lr_scheduler=scheduler,
        scheduler_update_every_step=not opt.trainMapping,
        metrics=[PSNRMeter()],
        use_checkpoint=opt.ckpt,
        eval_interval=50,
        model2=model2,
    )

    if opt.trainMapping:
        trainer.trainMapping(train_loader, None, max_epoch, model2)
    else:
        trainer.train_correctives(train_loader, None, max_epoch)
    open(done_file, "w").close()


if __name__ == "__main__":
    main()
