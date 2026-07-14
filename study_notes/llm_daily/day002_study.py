"""Day 2: Python collections, classes, file I/O, and clean tensor code."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch


class StudySession:
    """Store a day's learning topics and describe the session."""

    def __init__(self, day, topics, completed):
        self.day = day
        self.topics = topics
        self.completed = completed

    def summary(self):
        status = "completed" if self.completed else "in progress"
        return f"Day {self.day} ({status}): {', '.join(self.topics)}"

    def topic_count(self):
        return len(self.topics)


def describe_tensor(name, tensor):
    """Print a tensor and its essential properties."""
    print(f"{name}: {tensor}")
    print(f"{name} shape: {tensor.shape}")
    print(f"{name} dtype: {tensor.dtype}")


def tensor_operations(x, y):
    """Return common element-wise operations for two tensors."""
    return {"add": torch.add(x, y), "multiply": torch.mul(x, y)}


def save_and_load_summary(summary):
    """Write a summary to a temporary file and return its contents."""
    # The temporary directory is removed after the block, so this I/O example leaves no lesson artifact.
    with TemporaryDirectory() as directory:
        path = Path(directory) / "day2_summary.txt"
        path.write_text(summary, encoding="utf-8")
        return path.read_text(encoding="utf-8")


def main():
    # The appended topic shows that the same list is passed into the session object by reference.
    topics = ["lists", "dictionaries", "classes", "file I/O", "PyTorch"]
    topics.append("clean code")
    session_data = {"day": 2, "focus": "Python organization", "completed": True}
    session = StudySession(
        day=session_data["day"],
        topics=topics,
        completed=session_data["completed"],
    )

    print("Today's topics:")
    for topic in session.topics:
        print("-", topic)
    print("Session data:", session_data)
    print("Topic count:", session.topic_count())
    print("Saved summary:", save_and_load_summary(session.summary()))

    # Matching vector shapes let tensor_operations return independent elementwise result tensors.
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    results = tensor_operations(x, y)

    print()
    describe_tensor("x", x)
    describe_tensor("y", y)
    print("addition:", results["add"])
    print("multiplication:", results["multiply"])
    print("mean of x:", x.mean())


# The direct-execution guard lets the reusable class and helpers be imported without producing lesson output.
if __name__ == "__main__":
    main()
