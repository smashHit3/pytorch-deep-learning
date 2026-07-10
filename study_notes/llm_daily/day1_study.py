import torch

a = torch.tensor([1])
b = torch.tensor([[1, 2, 3]])

print(a, a.shape)
print(b, b.shape)

person = {
    "name": "xxx",
    "age": 24
}

print(person["name"])
print(person["age"])

print(person.keys())
print(person.values())

del person["name"]
person["city"] = "shenzhen"

print(person.keys())
print(person.values())