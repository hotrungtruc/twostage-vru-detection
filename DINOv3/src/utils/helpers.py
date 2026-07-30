import numpy as np

def calculate_iou(box1, box2):
    """
    Calculate IoU (Intersection over Union) between two bounding boxes.
    Box structure: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return inter / (area1 + area2 - inter + 1e-6)


def calculate_ioa(box1, box2):
    """
    Calculate IoA (Intersection over Area of box1) between two bounding boxes.
    This is useful for checking if box1 is completely inside box2.
    Box structure: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    
    return inter / (area1 + 1e-6)


def crop_with_padding(frame, box, pad_ratio):
    """
    Crop an image region from the frame with an optional padding ratio.
    """
    x1, y1, x2, y2 = map(int, box)
    h, w = y2 - y1, x2 - x1
    
    pad_w = int(w * pad_ratio)
    pad_h = int(h * pad_ratio)
    
    cx1 = max(0, x1 - pad_w)
    cy1 = max(0, y1 - pad_h)
    cx2 = min(frame.shape[1], x2 + pad_w)
    cy2 = min(frame.shape[0], y2 + pad_h)
    
    return frame[cy1:cy2, cx1:cx2]


def get_padded_box(box, pad_ratio, frame_w, frame_h):
    """
    Calculate new bounding box coordinates after applying a padding ratio,
    ensuring it does not exceed the frame dimensions (frame_w, frame_h).
    """
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    
    pad_w = int(w * pad_ratio)
    pad_h = int(h * pad_ratio)
    
    cx1 = max(0, x1 - pad_w)
    cy1 = max(0, y1 - pad_h)
    cx2 = min(frame_w, x2 + pad_w)
    cy2 = min(frame_h, y2 + pad_h)
    
    return [cx1, cy1, cx2, cy2]
