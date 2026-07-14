"""Day 7: Review tensors, gradients, and the steps of a training loop."""

import torch


def main():
    weight = torch.tensor(1.0, requires_grad=True)
    prediction = weight * torch.tensor([1.0, 2.0])
    loss = ((prediction - torch.tensor([2.0, 4.0])) ** 2).mean()
    loss.backward()
    review = {
        "tensor": {"shape": tuple(prediction.shape), "dtype": str(prediction.dtype)},
        "gradient": {"loss": round(loss.item(), 3), "d_loss/d_weight": round(weight.grad.item(), 3)},
        "training_loop": ["forward pass", "compute loss", "zero gradients", "backward", "optimizer step"],
    }
    for concept, note in review.items():
        print(f"{concept}: {note}")

if __name__ == "__main__":
    main()
