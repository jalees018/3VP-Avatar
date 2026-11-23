"""Render a trained 3VP avatar (novel views / re-enactment).

--modes single      : render with the stage-1 model only
otherwise           : render with stage-1 model + mapping network (--trainMapping)
--drive             : drive the avatar with the tracking of --driver
"""

import torch

from nerf.network import MappingNetwork, NeRFNetworkCond
from nerf.options import parse_args
from nerf.provider import NeRFDatasetCond
from nerf.utils import PSNRMeter, Trainer, seed_everything
from train import build_model


def main():
    opt = parse_args("render")
    print(opt)
    seed_everything(opt.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.MSELoss(reduction="none")
    model = build_model(NeRFNetworkCond, opt)
    model2 = build_model(NeRFNetworkCond, opt)
    if opt.trainMapping:
        model = build_model(MappingNetwork, opt)
    if opt.modes == "single":
        model2 = None

    trainer = Trainer(
        "ngp",
        opt,
        model,
        device=device,
        workspace=opt.workspace,
        criterion=criterion,
        fp16=opt.fp16,
        metrics=[PSNRMeter()],
        use_checkpoint=opt.ckpt,
    )
    test_loader = NeRFDatasetCond(opt, device=device, type="test").dataloader()
    if opt.trainMapping:
        trainer.animateSampleDepth(test_loader, model2, model)
    else:
        trainer.animateSampleDepth(test_loader, model2)


if __name__ == "__main__":
    main()
