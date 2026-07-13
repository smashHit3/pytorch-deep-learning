"""Day 54: Train a tiny causal-attention GPT-style language model locally."""

import torch


class MiniGPT(torch.nn.Module):
    """Token embeddings, causal attention, and a vocabulary prediction head."""

    def __init__(self, vocabulary_size, width=6):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocabulary_size, width)
        self.attention = torch.nn.MultiheadAttention(width, 2, batch_first=True)
        self.head = torch.nn.Linear(width, vocabulary_size)

    def forward(self, ids):
        hidden = self.embedding(ids)
        length = ids.size(1)
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
        attended, _ = self.attention(hidden, hidden, hidden, attn_mask=mask, need_weights=False)
        return self.head(hidden + attended)


def main():
    torch.manual_seed(0)
    data = torch.tensor([[0, 1, 2, 0, 1, 2]])
    model = MiniGPT(vocabulary_size=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
    for epoch in range(3):
        loss = torch.nn.functional.cross_entropy(model(data[:, :-1]).reshape(-1, 3), data[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("epoch", epoch + 1, "next-token loss", round(loss.item(), 4))
    print("The causal mask prevents each position from reading later target tokens.")

if __name__ == "__main__":
    main()
