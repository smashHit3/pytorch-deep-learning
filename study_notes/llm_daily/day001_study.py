"""Day 1: Python and PyTorch basics."""

import torch


def tensor_mean(tensor):
    """Return the arithmetic mean of all tensor values."""
    return tensor.mean()


def main():
    # Lists preserve the lesson order, while the dictionary labels the same session metadata.
    name = "Learner"
    study_topics = ["variables", "lists", "loops", "functions", "tensors"]
    study_info = {"day": 1, "topic": "Python and PyTorch basics"}

    print(f"Hello, {name}!")
    print("Study info:", study_info)
    print("Today's topics:")
    for topic in study_topics:
        print("-", topic)

    # These equal-length float vectors make the elementwise add and multiply outputs easy to inspect.
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

    # A 2-by-3 matrix demonstrates that reductions such as sum and max consume every element.
    matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print("\n2D tensor:", matrix)
    print("2D tensor shape:", matrix.shape)
    print("zeros:", torch.zeros((2, 3)))
    print("ones:", torch.ones((2, 3)))
    print("sum of matrix:", matrix.sum())
    print("max of matrix:", matrix.max())


# Elementwise arithmetic preserves the tensor shape, while mean reduces every element to one scalar.
if __name__ == "__main__":
    main()
