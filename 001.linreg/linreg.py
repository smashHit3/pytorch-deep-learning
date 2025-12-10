import os
import random
import torch
import matplotlib
# If no X display is available (headless), use the non-interactive Agg backend.
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

def synthetic_data(w, b, num_examples):  #@save
    """Generate y = Xw + b + noise."""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # Shuffle the data
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def linreg(X, w, b):  #@save
    """The linear regression model."""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):  #@save
    """Mean squared error loss."""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):  #@save    
    """Minibatch stochastic gradient descent."""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # Compute loss
        l.sum().backward()          # Backpropagate
        sgd([w, b], lr, batch_size) # Update parameters
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print('true_w:', true_w, '\nestimated_w:', w.reshape(true_w.shape))
print('true_b:', true_b, '\nestimated_b:', b)

print('features:', features[0],"\nlabel:", labels[0])
# Ensure tensors are on CPU and detached before converting to NumPy
x = features[:, 1].cpu().detach().numpy()
y = labels.cpu().detach().numpy()

plt.figure(figsize=(16, 9))
plt.scatter(x, y, s=1)
# If headless, save to file; otherwise show interactively
if os.environ.get('DISPLAY', '') == '':
    out = 'scatter.png'
    plt.savefig(out, dpi=200)
    print(f'Saved {out}')
else:
    plt.show()
