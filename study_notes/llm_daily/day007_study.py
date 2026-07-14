"""Day 7: Review tensors, gradients, and the steps of a training loop."""

import torch


def main():
    # A scalar weight is broadcast across two inputs, yielding a length-2 prediction tensor.
    weight = torch.tensor(1.0, requires_grad=True)
    prediction = weight * torch.tensor([1.0, 2.0])
    # Mean squared error averages the two per-example residual squares into the scalar backward needs.
    loss = ((prediction - torch.tensor([2.0, 4.0])) ** 2).mean()
    loss.backward()
    # Keep numeric values in the review so the printed summary connects concepts to this concrete forward pass.
    review = {
        "tensor": {"shape": tuple(prediction.shape), "dtype": str(prediction.dtype)},
        "gradient": {"loss": round(loss.item(), 3), "d_loss/d_weight": round(weight.grad.item(), 3)},
        "training_loop": ["forward pass", "compute loss", "zero gradients", "backward", "optimizer step"],
    }
    for concept, note in review.items():
        print(f"{concept}: {note}")

# The printed review separates forward prediction, loss measurement, gradient calculation, and parameter update.
if __name__ == "__main__":
    main()
