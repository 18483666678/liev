import numpy as np


def iou(box, boxes, isMin=False):
    # [x1,y1,x2,y2,c]
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)
    inter = w * h

    if isMin:
        ovr = inter / np.minimum(boxes, area)
        # ovr = np.true_divide(inter, np.minimum(box_area, area))
    else:
        ovr = inter / (box_area + area - inter)
        # ovr = np.true_divide(inter, (box_area + area - inter))
    return ovr


def nms(boxes,thresh=0.3,isMin=False):

    if boxes.shape[0] == 0:
        return np.array([])

    #根据置信度从大到小排序 ,返回索引
    _boxes = boxes[(-boxes[:,4]).argsort()]
    r_boxes = [] #留下的框保留

    while _boxes.shape[0] > 1:
        a_box = _boxes[0]  #第一个值
        b_boxes = _boxes[1:]

        r_boxes.append(a_box) #留下第一个

        index = np.where(iou(a_box,b_boxes,isMin) < thresh)
        _boxes = b_boxes[index]

    if _boxes.shape[0] > 0:
        r_boxes.append(_boxes[0])

    return np.stack(r_boxes)

