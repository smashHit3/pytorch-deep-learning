"""Sequence model demo translated from chapter_recurrent-neural-networks/sequence.ipynb."""

from dataclasses import dataclass
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils import data


@dataclass(frozen=True)
class SequenceModelConfig:
    total_steps: int = 1000
    tau: int = 4
    batch_size: int = 16
    n_train: int = 600
    epochs: int = 5
    lr: float = 0.01
    freq: float = 0.01
    noise_std: float = 0.2
    hidden_dim: int = 10
    max_steps: int = 64


def make_sequence(
    total_steps: int = 1000,
    freq: float = 0.01,
    noise_std: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    time = torch.arange(1, total_steps + 1, dtype=torch.float32)
    x = torch.sin(freq * time) + torch.normal(0, noise_std, (total_steps,))
    return time, x


def _has_one_axis(values) -> bool:
    return (hasattr(values, "ndim") and values.ndim == 1) or (
        isinstance(values, list) and values and not hasattr(values[0], "__len__")
    )


def _to_plot_values(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return values


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend) -> None:
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid(True, alpha=0.3)


def plot(
    X,
    Y=None,
    xlabel=None,
    ylabel=None,
    legend=None,
    xlim=None,
    ylim=None,
    xscale: str = "linear",
    yscale: str = "linear",
    fmts: Sequence[str] = ("-", "m--", "g-.", "r:"),
    figsize: tuple[int, int] = (6, 3),
    axes=None,
) -> None:
    legend = legend or []

    if _has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif _has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)

    if axes is None:
        _, axes = plt.subplots(figsize=figsize)
    axes.cla()

    for x, y, fmt in zip(X, Y, fmts):
        x = _to_plot_values(x)
        y = _to_plot_values(y)
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)

    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    plt.tight_layout()
    if "agg" in plt.get_backend().lower():
        plt.close(axes.figure)
    else:
        plt.show()


def load_array(
    data_arrays: Iterable[torch.Tensor],
    batch_size: int,
    is_train: bool = True,
) -> data.DataLoader:
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)


def build_features(series: torch.Tensor, tau: int) -> tuple[torch.Tensor, torch.Tensor]:
    if tau <= 0:
        raise ValueError("tau must be positive")
    if len(series) <= tau:
        raise ValueError("series length must be greater than tau")

    features = torch.zeros((len(series) - tau, tau), dtype=series.dtype)
    for i in range(tau):
        features[:, i] = series[i : len(series) - tau + i]
    labels = series[tau:].reshape((-1, 1))
    return features, labels


def init_weights(module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)


def get_net(tau: int = 4, hidden_dim: int = 10) -> nn.Module:
    net = nn.Sequential(
        nn.Linear(tau, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )
    net.apply(init_weights)
    return net


def evaluate_loss(net: nn.Module, data_iter, loss) -> float:
    was_training = net.training
    net.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for X, y in data_iter:
            batch_loss = loss(net(X), y)
            total_loss += batch_loss.sum().item()
            total_count += batch_loss.numel()

    if was_training:
        net.train()
    return total_loss / total_count


def train(net: nn.Module, train_iter, loss, epochs: int, lr: float) -> list[float]:
    trainer = torch.optim.Adam(net.parameters(), lr)
    history = []

    for epoch in range(epochs):
        net.train()
        for X, y in train_iter:
            trainer.zero_grad()
            batch_loss = loss(net(X), y)
            batch_loss.sum().backward()
            trainer.step()

        epoch_loss = evaluate_loss(net, train_iter, loss)
        history.append(epoch_loss)
        print(f"epoch {epoch + 1}, loss: {epoch_loss:f}")

    return history


def predict_onestep(net: nn.Module, features: torch.Tensor) -> torch.Tensor:
    return net(features)


def predict_multistep(
    net: nn.Module,
    series: torch.Tensor,
    tau: int,
    n_train: int,
) -> torch.Tensor:
    preds = torch.zeros_like(series)
    preds[: n_train + tau] = series[: n_train + tau]
    for i in range(n_train + tau, len(series)):
        preds[i] = net(preds[i - tau : i].reshape((1, -1))).reshape(())
    return preds


def predict_k_step(
    net: nn.Module,
    series: torch.Tensor,
    tau: int,
    max_steps: int,
) -> torch.Tensor:
    rows = len(series) - tau - max_steps + 1
    features = torch.zeros((rows, tau + max_steps), dtype=series.dtype)

    for i in range(tau):
        features[:, i] = series[i : i + rows]

    for i in range(tau, tau + max_steps):
        features[:, i] = net(features[:, i - tau : i]).reshape(-1)

    return features


def plot_data(time: torch.Tensor, series: torch.Tensor) -> None:
    plot(time, [series], "time", "x", xlim=[1, len(time)], figsize=(6, 3))


def plot_onestep_predictions(
    time: torch.Tensor,
    series: torch.Tensor,
    tau: int,
    onestep_preds: torch.Tensor,
) -> None:
    plot(
        [time, time[tau:]],
        [series, onestep_preds.reshape(-1)],
        "time",
        "x",
        legend=["data", "1-step preds"],
        xlim=[1, len(time)],
        figsize=(6, 3),
    )


def plot_multistep_predictions(
    time: torch.Tensor,
    series: torch.Tensor,
    tau: int,
    n_train: int,
    onestep_preds: torch.Tensor,
    multistep_preds: torch.Tensor,
) -> None:
    plot(
        [time, time[tau:], time[n_train + tau :]],
        [series, onestep_preds.reshape(-1), multistep_preds[n_train + tau :]],
        "time",
        "x",
        legend=["data", "1-step preds", "multistep preds"],
        xlim=[1, len(time)],
        figsize=(6, 3),
    )


def plot_k_step_predictions(
    time: torch.Tensor,
    forecast_features: torch.Tensor,
    tau: int,
    max_steps: int,
    steps: Sequence[int] = (1, 4, 16, 64),
) -> None:
    if any(step > max_steps for step in steps):
        raise ValueError("all steps must be less than or equal to max_steps")

    plot(
        [time[tau + step - 1 : len(time) - max_steps + step] for step in steps],
        [forecast_features[:, tau + step - 1] for step in steps],
        "time",
        "x",
        legend=[f"{step}-step preds" for step in steps],
        xlim=[tau + 1, len(time)],
        figsize=(6, 3),
    )


def run_sequence_model_demo(
    config: SequenceModelConfig | None = None,
) -> dict[str, object]:
    config = config or SequenceModelConfig()

    time, x = make_sequence(
        total_steps=config.total_steps,
        freq=config.freq,
        noise_std=config.noise_std,
    )
    plot_data(time, x)

    features, labels = build_features(x, config.tau)
    train_iter = load_array(
        (features[: config.n_train], labels[: config.n_train]),
        config.batch_size,
        is_train=True,
    )

    net = get_net(config.tau, config.hidden_dim)
    loss = nn.MSELoss(reduction="none")
    history = train(net, train_iter, loss, config.epochs, config.lr)

    onestep_preds = predict_onestep(net, features)
    plot_onestep_predictions(time, x, config.tau, onestep_preds)

    multistep_preds = predict_multistep(net, x, config.tau, config.n_train)
    plot_multistep_predictions(
        time,
        x,
        config.tau,
        config.n_train,
        onestep_preds,
        multistep_preds,
    )

    forecast_features = predict_k_step(net, x, config.tau, config.max_steps)
    plot_k_step_predictions(
        time,
        forecast_features,
        config.tau,
        config.max_steps,
    )

    return {
        "config": config,
        "time": time,
        "x": x,
        "features": features,
        "labels": labels,
        "train_iter": train_iter,
        "net": net,
        "loss": loss,
        "history": history,
        "onestep_preds": onestep_preds,
        "multistep_preds": multistep_preds,
        "forecast_features": forecast_features,
    }


__all__ = [
    "SequenceModelConfig",
    "build_features",
    "evaluate_loss",
    "get_net",
    "init_weights",
    "load_array",
    "make_sequence",
    "plot",
    "plot_data",
    "plot_k_step_predictions",
    "plot_multistep_predictions",
    "plot_onestep_predictions",
    "predict_k_step",
    "predict_multistep",
    "predict_onestep",
    "run_sequence_model_demo",
    "set_axes",
    "train",
]


if __name__ == "__main__":
    run_sequence_model_demo()
