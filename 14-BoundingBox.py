import torch
import shared as d2l
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
img = plt.imread('img/catdog.jpg')
# plt.imshow(img)
# plt.show()

def box_corner_to_center(boxes):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack((cx, cy, w, h), axis=-1)

def box_center_to_corner(boxes):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    return torch.stack((x1, y1, x2, y2), axis=-1)

dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
boxes = torch.tensor((dog_bbox, cat_bbox))
print(box_center_to_corner(box_corner_to_center(boxes)) == boxes)

def bbox_to_rect(bbox, color):
    x1, y1, x2, y2 = bbox
    return plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                         fill=False, edgecolor=color, linewidth=2)

plt.imshow(img)
dog_rect = bbox_to_rect(dog_bbox, 'blue')
cat_rect = bbox_to_rect(cat_bbox, 'red')
plt.gca().add_patch(dog_rect)
plt.gca().add_patch(cat_rect)
plt.show()