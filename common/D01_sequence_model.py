"""Utilities and demo helpers for sequence modeling with a fixed Markov window."""

from dataclasses import dataclass
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils import data


@dataclass(frozen=True)
class SequenceDemoConfig:
    total_steps: int = 1000
    tau: int = 100
    batch_size: int = 16
    n_train: int = 600
    epochs: int = 5
    lr: float = 0.01
    freq: float = 0.01
    noise_std: float = 0.2
    hidden_dim: int = 10


def load_curve(
    total_steps: int = 1000,
    freq: float = 0.01,
    noise_std: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    time = torch.arange(1, total_steps + 1, dtype=torch.float32)
    x = torch.sin(freq * time) + torch.normal(0, noise_std, (total_steps,))
    return time, x


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


def _has_one_axis(values) -> bool:
    return (hasattr(values, "ndim") and values.ndim == 1) or (
        isinstance(values, list) and values and not hasattr(values[0], "__len__")
    )


def _to_plot_values(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return values


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
    figsize: tuple[int, int] = (10, 4),
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
    plt.show()


def load_array(
    data_arrays: Iterable[torch.Tensor],
    batch_size: int,
    is_train: bool = True,
) -> data.DataLoader:
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)


def build_markov_features(
    series: torch.Tensor,
    tau: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        nn.init.xavier_normal_(module.weight)


def get_net(input_dim: int, hidden_dim: int = 10) -> nn.Module:
    net = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )
    net.apply(init_weights)
    return net


def evaluate_loss(net, data_iter, loss) -> float:
    was_training = net.training
    net.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for X, y in data_iter:
            out = net(X)
            y = y.reshape(out.shape)
            batch_loss = loss(out, y)
            if batch_loss.ndim == 0:
                total_loss += batch_loss.item() * y.numel()
                total_count += y.numel()
            else:
                total_loss += batch_loss.sum().item()
                total_count += batch_loss.numel()

    if was_training:
        net.train()
    return total_loss / total_count


def train(
    net,
    train_iter,
    loss,
    epochs: int,
    lr: float,
) -> list[float]:
    trainer = torch.optim.Adam(net.parameters(), lr)
    history = []

    for epoch in range(epochs):
        net.train()
        for X, y in train_iter:
            trainer.zero_grad()
            batch_loss = loss(net(X), y)
            batch_loss.backward()
            trainer.step()

        epoch_loss = evaluate_loss(net, train_iter, loss)
        history.append(epoch_loss)
        print(f"epoch {epoch + 1}, loss: {epoch_loss:f}")

    return history


def predict_multistep(
    net: nn.Module,
    series: torch.Tensor,
    tau: int,
    warmup_steps: int,
) -> torch.Tensor:
    preds = torch.zeros_like(series)
    preds[:warmup_steps] = series[:warmup_steps]

    for i in range(warmup_steps, len(series)):
        window = preds[i - tau : i].reshape((1, -1))
        preds[i] = net(window).reshape(())

    return preds


def plot_one_step_predictions(
    time: torch.Tensor,
    series: torch.Tensor,
    tau: int,
    preds: torch.Tensor,
) -> None:
    plot(
        [time, time[tau:]],
        [series, preds.reshape(-1)],
        xlabel="time",
        ylabel="x",
        legend=["data", "1-step preds"],
        xlim=[1, len(time)],
    )


def plot_prediction_comparison(
    time: torch.Tensor,
    series: torch.Tensor,
    tau: int,
    n_train: int,
    one_step_preds: torch.Tensor,
    multistep_preds: torch.Tensor,
) -> None:
    plot(
        [time, time[tau:], time[n_train + tau :]],
        [series, one_step_preds.reshape(-1), multistep_preds[n_train + tau :]],
        xlabel="time",
        ylabel="x",
        legend=["data", "1-step preds", "multistep preds"],
        xlim=[1, len(time)],
    )


def run_sequence_model_demo(
    config: SequenceDemoConfig | None = None,
):
    config = config or SequenceDemoConfig()

    time, x = load_curve(
        total_steps=config.total_steps,
        freq=config.freq,
        noise_std=config.noise_std,
    )
    plot(time, x, xlabel="time", ylabel="x", legend=["x"])

    features, labels = build_markov_features(x, config.tau)
    train_iter = load_array(
        (features[: config.n_train], labels[: config.n_train]),
        batch_size=config.batch_size,
        is_train=True,
    )

    net = get_net(input_dim=config.tau, hidden_dim=config.hidden_dim)
    loss = nn.MSELoss()
    history = train(net, train_iter, loss, config.epochs, config.lr)

    one_step_preds = net(features)
    plot_one_step_predictions(time, x, config.tau, one_step_preds)

    multistep_preds = predict_multistep(
        net,
        x,
        tau=config.tau,
        warmup_steps=config.n_train + config.tau,
    )
    plot_prediction_comparison(
        time,
        x,
        tau=config.tau,
        n_train=config.n_train,
        one_step_preds=one_step_preds,
        multistep_preds=multistep_preds,
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
        "one_step_preds": one_step_preds,
        "multistep_preds": multistep_preds,
    }


__all__ = [
    "SequenceDemoConfig",
    "build_markov_features",
    "evaluate_loss",
    "get_net",
    "init_weights",
    "load_array",
    "load_curve",
    "plot",
    "plot_one_step_predictions",
    "plot_prediction_comparison",
    "predict_multistep",
    "run_sequence_model_demo",
    "set_axes",
    "train",
]


if __name__ == "__main__":
    run_sequence_model_demo()
