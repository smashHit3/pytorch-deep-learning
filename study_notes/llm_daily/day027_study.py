"""Day 27: Train a tiny character-level next-character sequence model."""

import torch


def main():
    torch.manual_seed(0)
    text = "abababa"
    vocabulary = {character: index for index, character in enumerate(sorted(set(text)))}
    token_ids = torch.tensor([vocabulary[character] for character in text])
    model = torch.nn.Sequential(torch.nn.Embedding(len(vocabulary), 4), torch.nn.Linear(4, len(vocabulary)))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
    for _ in range(30):
        loss = torch.nn.functional.cross_entropy(model(token_ids[:-1]), token_ids[1:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    predictions = model(token_ids[:-1]).argmax(dim=-1)
    decoded = "".join(sorted(vocabulary, key=vocabulary.get)[index] for index in predictions.tolist())
    print("characters:", vocabulary, "final loss:", round(loss.item(), 4))
    print("input:", text[:-1], "predicted next characters:", decoded)

if __name__ == "__main__":
    main()
