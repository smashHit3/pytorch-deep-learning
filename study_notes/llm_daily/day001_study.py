"""Day 1: Python and PyTorch basics."""

import torch


def tensor_mean(tensor):
    """Return the arithmetic mean of all tensor values."""
    return tensor.mean()


def main():
    name = "Learner"
    study_topics = ["variables", "lists", "loops", "functions", "tensors"]
    study_info = {"day": 1, "topic": "Python and PyTorch basics"}

    print(f"Hello, {name}!")
    print("Study info:", study_info)
    print("Today's topics:")
    for topic in study_topics:
        print("-", topic)

    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    z_add = torch.add(x, y)
    z_mul = torch.mul(x, y)

    print("\nx:", x)
    print("y:", y)
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("x dtype:", x.dtype)
    print("y dtype:", y.dtype)
    print("add:", z_add)
    print("mul:", z_mul)
    print("mean of x:", tensor_mean(x))

    matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print("\n2D tensor:", matrix)
    print("2D tensor shape:", matrix.shape)
    print("zeros:", torch.zeros((2, 3)))
    print("ones:", torch.ones((2, 3)))
    print("sum of matrix:", matrix.sum())
    print("max of matrix:", matrix.max())


if __name__ == "__main__":
    main()
