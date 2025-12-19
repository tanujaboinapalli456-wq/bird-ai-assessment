import math

def relative_weight_index(bbox):
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    return round(math.sqrt(area), 2)
