import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm

import fiftyone as fo
from fiftyone import ViewField as F

import pandas as pd
import plotly.express as px

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, Dropout
from torch.cuda import amp
from torch.optim.lr_scheduler import LambdaLR

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import TQDM_BAR_FORMAT, LOGGER  # noqa: E402
from utils.torch_utils import smart_optimizer, ModelEMA  # noqa: E402
from utils.downloads import attempt_download  # noqa: E402

from models.custom import HorizonModel  # noqa: E402
from horizon.dataloaders import (  # noqa: E402
    get_train_rgb_dataloader,
    get_val_rgb_dataloader,
    get_train_ir16bit_dataseloader,
    get_val_ir16bit_dataloader,
)


def get_dataloaders(
    dataset_name: str,
    train_tag: str,
    val_tag: str,
    imgsz: int,
    field: str = "ground_truth_pl.polylines.closed",
):
    # TODO: add tag check
    if "RGB" in dataset_name:
        train_dataloader = get_train_rgb_dataloader(
            dataset=(
                fo.load_dataset(dataset_name)
                .match(F(field) == [False])
                .match_tags(train_tag)
            ),
            imgsz=imgsz,
            batch_size=64 if imgsz == 640 else 16,
        )

        val_dataloader = get_val_rgb_dataloader(
            dataset=(
                fo.load_dataset(dataset_name)
                .match(F(field) == [False])
                .match_tags(val_tag)
                # .take(5000, seed=51)
            ),
            imgsz=imgsz,
            batch_size=64 if imgsz == 640 else 16,
        )
    else:
        train_dataloader = get_train_ir16bit_dataseloader(
            dataset=(
                fo.load_dataset(dataset_name)
                .match(F(field) == [False])
                .match_tags(train_tag)
            ),
            imgsz=imgsz,
        )

        val_dataloader = get_val_ir16bit_dataloader(
            dataset=(
                fo.load_dataset(dataset_name)
                .match(F(field) == [False])
                .match_tags(val_tag)
            ),
            imgsz=imgsz,
        )

    return train_dataloader, val_dataloader


def update(
    model: HorizonModel,
    train_dataloader: DataLoader,
    loss_pitch: CrossEntropyLoss,
    loss_theta: CrossEntropyLoss,
    pitch_weight: float,
    theta_weight: float,
    scaler: amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    ema: ModelEMA,
    epoch: int,
    epochs: int,
):
    """Update model weights via backpropagation."""

    # initialize mean losses
    t_loss, t_ploss, t_tloss = 0.0, 0.0, 0.0

    # iterate over batches
    pbar = tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader),
        bar_format=TQDM_BAR_FORMAT,
    )

    for i, data in pbar:
        images, targets = data
        images, targets = images.to(model.device), targets.to(model.device)

        # process targets
        pitch_i, theta_i = model.to_discrete(pitch=targets[..., 0], theta=targets[..., 1])

        # forward
        x_pitch, x_theta = model(images)

        # backward
        _loss_pitch = loss_pitch(x_pitch, pitch_i)
        _loss_theta = loss_theta(x_theta, theta_i)
        loss = pitch_weight * _loss_pitch + theta_weight * _loss_theta
        scaler.scale(loss).backward()

        # Optimize
        scaler.unscale_(optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=10.0
        )  # clip gradients
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if ema:
            ema.update(model)

        t_loss = (t_loss * i + loss.item()) / (i + 1)  # update mean losses
        t_ploss = (t_ploss * i + _loss_pitch.item()) / (i + 1)
        t_tloss = (t_tloss * i + _loss_theta.item()) / (i + 1)

        mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.2f} GB"
        losses_str = f"{t_loss:>12.3g}{t_ploss:>12.3g}{t_tloss:>12.3g}"
        pbar.desc = (
            f"{'train':>6}{f'{epoch + 1}/{epochs}':>10}{mem:>10}{losses_str}" + " " * 10
        )

    return t_loss, t_ploss, t_tloss


def evaluate(
    model: HorizonModel,
    val_dataloader: DataLoader,
    loss_pitch: CrossEntropyLoss,
    loss_theta: CrossEntropyLoss,
    pitch_weight: float,
    theta_weight: float,
    ema: ModelEMA,
    epoch: int,
    epochs: int,
):
    """Evaluate model on validation set."""

    # use MSE as metric
    criterion_pitch = torch.nn.MSELoss()
    criterion_theta = torch.nn.MSELoss()
    mse_pitch, mse_theta = 0.0, 0.0

    # initialize mean losses
    v_loss, v_ploss, v_tloss = 0.0, 0.0, 0.0

    # iterate over batches
    pbar = tqdm(
        enumerate(val_dataloader),
        total=len(val_dataloader),
        bar_format=TQDM_BAR_FORMAT,
    )

    for i, data in pbar:
        images, targets = data
        images, targets = images.to(model.device), targets.to(model.device)

        with torch.no_grad():
            x_pitch, x_theta = ema.ema(images)

        # process targets
        pitch_i, theta_i = model.to_discrete(pitch=targets[..., 0], theta=targets[..., 1])

        # store losses
        _loss_pitch = loss_pitch(x_pitch, pitch_i)
        _loss_theta = loss_theta(x_theta, theta_i)
        loss = pitch_weight * _loss_pitch + theta_weight * _loss_theta

        v_loss = (v_loss * i + loss.item()) / (i + 1)  # update mean losses
        v_ploss = (v_ploss * i + _loss_pitch.item()) / (i + 1)
        v_tloss = (v_tloss * i + _loss_theta.item()) / (i + 1)

        # calculate MSE
        (y_pitch, _), (y_theta, _) = model.postprocess(x_pitch, x_theta)
        mse_pitch += criterion_pitch(y_pitch, targets[..., 0])
        mse_theta += criterion_theta(y_theta, targets[..., 1])

        mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.2f} GB"
        losses_str = f"{v_loss:>12.3g}{v_ploss:>12.3g}{v_tloss:>12.3g}"
        pbar.desc = (
            f"{'train':>6}{f'{epoch + 1}/{epochs}':>10}{mem:>10}{losses_str}" + " " * 10
        )

    return v_loss, v_ploss, v_tloss, mse_pitch, mse_theta


def run(
    dataset_name: str,  # fiftyone dataset name
    train_tag: str = "train",  # fiftyone dataset tag
    val_tag: str = "val",  # fiftyone dataset tag
    weights: str = "yolov5n.pt",  # initial weights path
    nc_pitch: int = 500,  # number of pitch classes
    nc_theta: int = 500,  # number of theta classes
    pitch_weight: float = 1.0,  # pitch loss weight
    theta_weight: float = 1.0,  # theta loss weight
    imgsz: int = 640,  # model input size (assumes squared input)
    epochs: int = 100,
    dropout: float = 0.25,  # dropout rate for classification heads
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):

    # create dir to store checkpoints
    ckpt_dir = ROOT / "runs" / "horizon" / "train" / dataset_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"{ckpt_dir=}\n")

    if not os.path.exists(weights):
        weights = attempt_download(weights)  # download if not found locally

    # load as horizon model
    model = HorizonModel(weights, nc_pitch, nc_theta, device=device)
    for m in model.model:
        print(m.i, m.f, m.type)
    for m in model.modules():
        if isinstance(m, Dropout) and dropout is not None:
            m.p = dropout  # set dropout
    for p in model.parameters():
        p.requires_grad = True  # all params trainable
    LOGGER.info(f"{model.nc_pitch=}, {model.nc_theta=}\n")

    # load dataloaders
    train_dataloader, val_dataloader = get_dataloaders(
        dataset_name, train_tag, val_tag, imgsz
    )
    LOGGER.info(f"{len(train_dataloader)=}, {len(val_dataloader)=}")

    optimizer = smart_optimizer(
        model, name="Adam", lr=0.001, momentum=0.9, decay=0.0001
    )

    lrf = 0.001  # final lr (fraction of lr0)
    lf = lambda x: (1 - x / epochs) * (1 - lrf) + lrf  # linear
    scheduler = LambdaLR(optimizer, lr_lambda=lf)

    loss_pitch = CrossEntropyLoss(label_smoothing=0.0)
    loss_theta = CrossEntropyLoss(label_smoothing=0.0)

    scaler = amp.GradScaler(enabled=model.device != "cpu")

    ema = ModelEMA(model)
    best_loss = 1e10

    model.info()
    LOGGER.info("Starting training...\n")
    train_losses = dict(pitch=[], theta=[], total=[])
    val_losses = dict(pitch=[], theta=[], total=[])
    mses = dict(pitch=[], theta=[])

    for epoch in range(epochs):
        t_loss, t_ploss, t_tloss = 0.0, 0.0, 0.0
        v_loss, v_ploss, v_tloss = 0.0, 0.0, 0.0

        model.train()
        t_loss, t_ploss, t_tloss = update(
            model,
            train_dataloader,
            loss_pitch,
            loss_theta,
            pitch_weight,
            theta_weight,
            scaler,
            optimizer,
            ema,
            epoch,
            epochs,
        )

        scheduler.step()

        model.eval()
        v_loss, v_ploss, v_tloss, mse_pitch, mse_theta = evaluate(
            model,
            val_dataloader,
            loss_pitch,
            loss_theta,
            pitch_weight,
            theta_weight,
            ema,
            epoch,
            epochs,
        )

        if v_loss < best_loss:
            best_loss = v_loss
            ckpt = {
                "epoch": epoch,
                "model": deepcopy(ema.ema),  # deepcopy(de_parallel(model)).half(),
                "ema": None,  # deepcopy(ema.ema).half(),
                "updates": ema.updates,
                "optimizer": None,  # optimizer.state_dict(),
                "losses": dict(pitch=v_ploss, theta=v_tloss, total=v_loss),
                "date": datetime.now().isoformat(),
            }
            torch.save(ckpt, ckpt_dir / "best.pt")
            del ckpt

        # Save latest checkpoint
        ckpt = {
            "epoch": epoch,
            "model": deepcopy(ema.ema),  # deepcopy(de_parallel(model)).half(),
            "ema": None,  # deepcopy(ema.ema).half(),
            "updates": ema.updates,
            "optimizer": None,  # optimizer.state_dict(),
            "losses": dict(pitch=v_ploss, theta=v_tloss, total=v_loss),
            "date": datetime.now().isoformat(),
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        del ckpt

        # plot losses
        train_losses["pitch"].append(t_ploss)
        train_losses["theta"].append(t_tloss)
        train_losses["total"].append(t_loss)
        val_losses["pitch"].append(v_ploss)
        val_losses["theta"].append(v_tloss)
        val_losses["total"].append(v_loss)
        mses["pitch"].append(mse_pitch)
        mses["theta"].append(mse_theta)
        plot_losses(ckpt_dir, train_losses, val_losses, mses)


def plot_losses(ckpt_dir, train_losses, val_losses, mses):
    fig_train = px.line(
        pd.DataFrame(train_losses),
        labels={"x": "epoch", "y": "loss"},
        title="Training Loss",
    )
    fig_val = px.line(
        pd.DataFrame(val_losses),
        labels={"x": "epoch", "y": "loss"},
        title="Validation Loss",
    )
    fig_mse = px.line(
        pd.DataFrame(mses),
        labels={"x": "epoch", "y": "mse"},
        title="Mean Squared Error",
    )

    fig_train.write_image(str(ckpt_dir / "train_losses.png"))
    fig_val.write_image(str(ckpt_dir / "val_losses.png"))
    fig_mse.write_image(str(ckpt_dir / "mse.png"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, help="dataset name", required=True)
    parser.add_argument("--train_tag", type=str, default="train", help="train tag")
    parser.add_argument("--val_tag", type=str, default="val", help="val tag")
    parser.add_argument(
        "--weights", type=str, default="yolov5n.pt", help="initial weights path"
    )
    parser.add_argument(
        "--nc_pitch", type=int, default=500, help="number of pitch classes"
    )
    parser.add_argument(
        "--nc_theta", type=int, default=500, help="number of theta classes"
    )
    parser.add_argument(
        "--pitch_weight", type=float, default=1.0, help="pitch loss weight"
    )
    parser.add_argument(
        "--theta_weight", type=float, default=1.0, help="theta loss weight"
    )
    parser.add_argument("--imgsz", type=int, default=640, help="train, val image size")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--dropout", type=float, default=0.25, help="dropout rate")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda device, i.e. 0 or 0,1,2,3 or cpu",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(**vars(args))
