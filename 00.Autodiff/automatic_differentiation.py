import torch

# y = 2x^Tx
x = torch.arange(4.0)
print(x)

x.requires_grad_(True)
print(x.grad)

y = 2 * torch.dot(x, x)
print(y)

y.backward()
print(x.grad)
print(x.grad == 4 * x)  # dy/dx = 4x

# Clear the previous gradient
x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

# Vector-Jacobian product
x.grad.zero_()
y = x * x
print(y)
# y = [x0^2, x1^2, x2^2, x3^2]
y.sum().backward()
print(x.grad)