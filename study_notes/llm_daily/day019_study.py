"""Day 19: Study overfitting, validation, and train/test split.

A dependency-light, local demonstration for the Day 19 LLM roadmap topic.
It uses a toy-sized example and labels production-scale limitations.
"""

import torch

def main():
    # The first seven ordered points train the model; the final three remain held out for validation.
    x = torch.arange(10., dtype=torch.float32).unsqueeze(1); y = 2*x + 1
    train_x, valid_x, train_y, valid_y = x[:7], x[7:], y[:7], y[7:]
    model = torch.nn.Linear(1, 1); optimizer = torch.optim.SGD(model.parameters(), lr=.03)
    # Only train_x and train_y participate in optimizer updates, preserving the validation comparison.
    for _ in range(100):
        loss = torch.nn.functional.mse_loss(model(train_x), train_y); optimizer.zero_grad(); loss.backward(); optimizer.step()
    # Both reported MSE values use the final parameters, but only the first influenced those parameters.
    print("train MSE:", round(loss.item(), 5), "validation MSE:", round(torch.nn.functional.mse_loss(model(valid_x), valid_y).item(), 5))
    print("The validation set is never used for fitting these parameters.")

# Comparing training and validation behavior helps reveal memorization that a training loss alone can hide.
if __name__ == "__main__":
    main()
