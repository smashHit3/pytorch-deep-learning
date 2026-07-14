"""Day 13: Compare SGD and Adam in a toy example.

A dependency-light, local demonstration for the Day 13 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

def step_sgd(x, grad, lr): return x - lr * grad

def main():
    # Both optimizers start at the same point so their ten-step trajectories can be compared fairly.
    x_sgd = x_adam = 5.0
    m = v = 0.0
    # The time index supplies Adam's bias-correction exponents as its moving averages warm up from zero.
    for t in range(1, 11):
        grad = 2 * x_sgd; x_sgd = step_sgd(x_sgd, grad, 0.1)
        # Adam accumulates first and second gradient moments before bias-correcting its adaptive step.
        grad = 2 * x_adam; m = .9*m + .1*grad; v = .999*v + .001*grad*grad
        x_adam -= .1 * (m/(1-.9**t)) / ((v/(1-.999**t))**.5 + 1e-8)
    print("SGD x:", round(x_sgd, 4), "Adam-style x:", round(x_adam, 4))

# Adam rescales updates using running gradient statistics, unlike SGD's direct fixed-rate gradient step.
if __name__ == "__main__":
    main()
