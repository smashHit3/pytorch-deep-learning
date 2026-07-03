# Day 1 Study Plan: Python and PyTorch Basics

## Goal
Build a first understanding of Python scripting and PyTorch tensors so you can read and write simple deep learning code.

## Today's Focus
- Python basics: variables, lists, loops, and functions
- PyTorch basics: tensors, shapes, dtypes, and simple math
- Running one Python script from start to finish

## Study Tasks

### 1. Python Basics
Learn or review:
- variables
- `list` and `dict`
- `for` loops
- function definition with `def`
- printing values with `print()`

### 2. PyTorch Basics
Learn or review:
- what a tensor is
- how to create a tensor
- tensor shape
- tensor dtype
- element-wise addition and multiplication
- mean of a tensor

### 3. Hands-on Exercise
Create a Python file and complete these tasks:
1. Import `torch`
2. Create two tensors
3. Add them together
4. Multiply them together
5. Print tensor values and shapes
6. Write a function that returns the mean of a tensor
7. Call the function and print the result

## Example Code

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

z_add = x + y
z_mul = x * y

def tensor_mean(t):
    return t.mean()

print("x:", x)
print("y:", y)
print("x shape:", x.shape)
print("y shape:", y.shape)
print("add:", z_add)
print("mul:", z_mul)
print("mean of x:", tensor_mean(x))
```

## What You Should Understand After Today
- A tensor is a data structure similar to an array, used heavily in deep learning.
- `shape` tells you the size of each dimension.
- `dtype` tells you the data type, such as float or integer.
- PyTorch can do tensor math directly without manual loops in many cases.

## Suggested Schedule
- 20 min: Python basics review
- 20 min: PyTorch tensor basics
- 30 min: Write and run the example code
- 20 min: Change values and test what happens
- 10 min: Write notes

## Mini Challenges
- Change the tensors from 1D to 2D
- Print the dtype of each tensor
- Try `torch.zeros()` and `torch.ones()`
- Compute the sum and max of a tensor

## Notes Template

```markdown
## Day 1 Notes

- What is a tensor?
- What is the difference between shape and dtype?
- What code worked well today?
- What confused me today?
- One thing I want to review tomorrow:
```

## Completion Checklist
- [ ] I wrote and ran one Python script
- [ ] I created and printed tensors
- [ ] I used addition and multiplication on tensors
- [ ] I wrote one Python function
- [ ] I understand shape and dtype at a basic level
- [ ] I wrote short notes about today's study
