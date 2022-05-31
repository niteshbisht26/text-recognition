import torch.nn as nn
import torch

class YOLOLayer(nn.Module):

  def __init__(self, anchors, img_size, device):
    super(YOLOLayer, self).__init__()
    self.img_size = img_size
    self.anchors = anchors
    self.device = device
    self.num_anchors = self.anchors.shape[0]
    self.mse = nn.MSELoss(reduction='sum')
    self.bce = nn.BCELoss(reduction='sum')
    self.obj_lambda = 1
    self.no_obj_lambda = 0.5
    self.coord_lambda = 5
    self.grid_size = 0

  def utils(self, grid_size):
    self.grid_size = grid_size
    self.stride = self.img_size/self.grid_size          # stride = 418/19 -> 22
    #Computing offsets for each grid in matrix fashion
    grid_nums = torch.arange(0, self.grid_size).type(torch.FloatTensor)
    self.grid_y, self.grid_x = torch.meshgrid(grid_nums, grid_nums)         #shape -> (19,19)
    self.grid_y, self.grid_x = self.grid_y.to(self.device), self.grid_x.to(self.device)

    self.scaled_anchors = (self.anchors/(self.stride)).to(self.device)

    self.anchor_w = self.scaled_anchors[:,0].view(1, self.num_anchors, 1, 1).to(self.device)  # shape -> (1,4,1,1)
    self.anchor_h = self.scaled_anchors[:,1].view(1, self.num_anchors, 1, 1).to(self.device)

  def forward(self, y_pred, targets=None, evaluation=False):  # y_pred -> (b,19,19,5x4)   

    self.batch_size = y_pred.shape[0]
    grid_size = y_pred.shape[1]
    predictions = y_pred.view(self.batch_size, grid_size, grid_size, self.num_anchors, 5).contiguous().permute(0, 3, 1, 2, 4).contiguous()              # shape -> (b,4,19,19,5)

    pred_conf = torch.sigmoid(predictions[...,0])
    x = torch.sigmoid(predictions[...,1])
    y = torch.sigmoid(predictions[...,2])
    w = predictions[...,3]                         # shape -> (b,4,19,19)
    h = predictions[...,4]

    if self.grid_size != grid_size:
      self.utils(grid_size)

    if not targets:
      pred_box = torch.zeros(self.batch_size, self.num_anchors, self.grid_size, self.grid_size, 5)
      pred_box[..., 0] = pred_conf
      pred_box[..., 1] = (x + self.grid_x)*self.stride                  # de-scaling
      pred_box[..., 2] = (y + self.grid_y)*self.stride
      pred_box[..., 3] = (torch.exp(w)*self.anchor_w)*self.stride
      pred_box[..., 4] = (torch.exp(h)*self.anchor_h)*self.stride
      return pred_box
    
    t_conf, tx, ty, tw, th = self.build_target(targets[0], targets[1])    # shape -> (b,4,19,19)

    if evaluation:
      pred_box = torch.zeros(self.batch_size, self.num_anchors, self.grid_size, self.grid_size, 5)
      pred_box[..., 0] = pred_conf
      pred_box[..., 1] = (x + self.grid_x)*self.stride                  # de-scaling
      pred_box[..., 2] = (y + self.grid_y)*self.stride
      pred_box[..., 3] = (torch.exp(w)*self.anchor_w)*self.stride
      pred_box[..., 4] = (torch.exp(h)*self.anchor_h)*self.stride
      return pred_box, t_conf
    
    bool_obj_mask = t_conf.type(torch.BoolTensor)
    no_obj_mask = (t_conf*-1) + 1
    bool_noobj_mask = no_obj_mask.type(torch.BoolTensor)

    loss_x = self.coord_lambda*self.mse(x[bool_obj_mask], tx[bool_obj_mask])
    loss_y = self.coord_lambda*self.mse(y[bool_obj_mask], ty[bool_obj_mask])
    loss_w = self.coord_lambda*self.mse(w[bool_obj_mask], tw[bool_obj_mask])
    loss_h = self.coord_lambda*self.mse(h[bool_obj_mask], th[bool_obj_mask])
    loss_obj = self.obj_lambda*self.bce(pred_conf[bool_obj_mask], t_conf[bool_obj_mask])
    loss_noobj = self.no_obj_lambda*self.bce(pred_conf[bool_noobj_mask], t_conf[bool_noobj_mask])

    total_loss = loss_x + loss_y + loss_w + loss_h + loss_obj + loss_noobj
    return total_loss
  
  def build_target(self, ground_bboxes, bbox_len):

    ground_bboxes /= self.stride      # shape -> (total boxes in a batch, 5) -> eg. (12, 4)

    gx, gy = ground_bboxes[:,0:2].t()     # shape -> (1, 12)
    gw, gh = ground_bboxes[:,2:4].t()
    gi, gj = gx.long(), gy.long()

    t_conf = torch.FloatTensor(self.batch_size, self.num_anchors, self.grid_size, self.grid_size).fill_(0).to(self.device)
    tx = torch.FloatTensor(self.batch_size, self.num_anchors, self.grid_size, self.grid_size).fill_(0).to(self.device)          # Shape -> (b,4,13,13)
    ty = torch.FloatTensor(self.batch_size, self.num_anchors, self.grid_size, self.grid_size).fill_(0).to(self.device)
    tw = torch.FloatTensor(self.batch_size, self.num_anchors, self.grid_size, self.grid_size).fill_(0).to(self.device)
    th = torch.FloatTensor(self.batch_size, self.num_anchors, self.grid_size, self.grid_size).fill_(0).to(self.device)

    ious = torch.cat([self.bbox_wh_iou(self.scaled_anchors, box[2:]).view(1, self.num_anchors) for box in ground_bboxes])  # shape -> (12, 4)
    best_ious, best_ious_index = ious.max(1)                                                   # shape -> (12,)

    batch_idx = []
    for i, num_box in enumerate(bbox_len):
      batch_idx.extend([i]*num_box)

    t_conf[batch_idx, best_ious_index, gj, gi] = 1
    tx[batch_idx, best_ious_index, gj, gi] = gx - gi
    ty[batch_idx, best_ious_index, gj, gi] = gy - gj
    tw[batch_idx, best_ious_index, gj, gi] = torch.log(gw/self.scaled_anchors[best_ious_index, 0])
    th[batch_idx, best_ious_index, gj, gi] = torch.log(gh/self.scaled_anchors[best_ious_index, 1])

    return [t_conf, tx, ty, tw, th]

  def bbox_wh_iou(self, anchors, target_box):

    inter_w = torch.min(anchors[:,0], target_box[0].repeat(anchors.shape[0]))
    inter_h = torch.min(anchors[:,1], target_box[1].repeat(anchors.shape[0]))

    inter_area = inter_w * inter_h

    anchors_area = anchors[:,0] * anchors[:,1]
    target_box_area = target_box[0] * target_box[1]

    total_area = anchors_area + target_box_area - inter_area

    return inter_area/total_area