import torch
from itertools import groupby
import numpy as np
import string
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from YOLOLayer import YOLOLayer
import streamlit as st

def iou(box1, box2):
  box1 = [box1[0] - box1[2]/2, box1[0] + box1[2]/2, box1[1] -
          box1[3]/2, box1[1] + box1[3]/2]
  box2 = [box2[0] - box2[2]/2, box2[0] + box2[2] /
          2, box2[1] - box2[3]/2, box2[1] + box2[3]/2]

  inter_x1 = max(box1[0], box2[0])
  inter_y1 = max(box1[2], box2[2])
  inter_x2 = min(box1[1], box2[1])
  inter_y2 = min(box1[3], box2[3])

  if inter_x2 < inter_x1 or inter_y2 < inter_y1:
    return 0

  inter_area = (inter_x2 - inter_x1)*(inter_y2 - inter_y1)
  tot_area = (box1[1] - box1[0])*(box1[3] - box1[2]) + \
      (box2[1] - box2[0])*(box2[3] - box2[2]) - inter_area

  return (inter_area/tot_area).item()


# Non Max suppression
def nms(out):
  predictions = out.view(-1, 5).contiguous()

  obj_mask = predictions[:, 0] > 0.5
  boxes = predictions[obj_mask, 1:5]
  obj_conf_score = predictions[obj_mask, 0]

  sort_index = torch.argsort(obj_conf_score)
  boxes = boxes[sort_index, :]

  nms_detected_boxes = []
  while boxes.shape[0] > 0:
    iou_ = []
    box = boxes[0, :]
    nms_detected_boxes.append(box)
    boxes = boxes[1:, :]
    for i in range(boxes.shape[0]):
      iou_.append(iou(box, boxes[i, :]))
    if len(iou_) > 0:
      iou_ = torch.Tensor(iou_)
      keep_boxes = iou_ < 0.5
      boxes = boxes[keep_boxes, :]

  return nms_detected_boxes


def bottom_left(boxes):
    boxes[:, 0] = boxes[:, 0] - boxes[:, 2]/2
    boxes[:, 1] = boxes[:, 1] - boxes[:, 3]/2
    return boxes

def get_bounding_boxes(model, img, input_img_size, anchors, device):
    test_transform = transforms.Compose([
                               transforms.Resize(input_img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.4666, 0.4546, 0.4374), (0.2268, 0.2224, 0.2321))
    ])

    img = img.resize(input_img_size)
    test_img = test_transform(img)
    test_img = test_img.unsqueeze(0).to(device)

    boxes = []
    with torch.no_grad():
        model.eval()
        out = model(test_img)
        out = YOLOLayer(anchors, input_img_size[0], device)(out)
        boxes = nms(out)
        if len(boxes) == 0:
            return None
        boxes = [box.unsqueeze(0) for box in boxes]
        boxes = torch.cat(boxes)
        boxes = bottom_left(boxes)
    return boxes

def draw_bounding_box(boxes, img, input_img_size):
    img = img.resize(input_img_size)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)
    ax.axis('off')

    for i in boxes:
      rect = patches.Rectangle((i[0], i[1]), i[2], i[3], linewidth=2, edgecolor='r', fill=False)
      ax.add_patch(rect)
    st.pyplot(fig)

def ctc_decoder(predictions, ind2alphanum):
    text_list = []

    pred_indcies = np.argmax(predictions, axis=2)

    for i in range(pred_indcies.shape[0]):
        ans = ""

        ## merge repeats
        merged_list = [k for k,_ in groupby(pred_indcies[i])]

        ## remove blanks
        for p in merged_list:
            if p != 0:
                ans += ind2alphanum[int(p)]

        text_list.append(ans)

    return text_list

def get_encodings():
    chars = string.ascii_letters + string.digits
    alphanum2ind = {char: i+1 for i, char in enumerate(list(chars))}
    alphanum2ind['-'] = 0
    ind2alphanum = {v: k for k, v in alphanum2ind.items()}
    return alphanum2ind, ind2alphanum

def get_cropped_images(img, boxes, input_img_size):
    img = img.resize(input_img_size)
    cropped_imgs = []
    for i in boxes.tolist():
        cropped_imgs.append(img.crop((i[0], i[1], i[0]+i[2], i[1]+i[3])))
    return cropped_imgs

def get_predicted_texts(cropped_imgs, img_size, model, ind2alphanum):
    tranformer = transforms.Compose([transforms.Grayscale(),
                                transforms.ToTensor(),
                                transforms.Resize(img_size)])
    images = []
    for i in cropped_imgs:
        img = tranformer(i)
        images.append(img)
    x = torch.stack(images, dim=0)

    model.eval()
    with torch.no_grad():
        outputs = model(x).cpu()
        preds = ctc_decoder(outputs, ind2alphanum)
    return preds
        
