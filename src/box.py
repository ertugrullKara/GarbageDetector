import numpy as np


class BoundBox:
    """
        Adopted from https://github.com/thtrieu/darkflow/blob/master/darkflow/utils/box.py
    """

    def __init__(self, obj_prob, probs=None, box_coord=[float() for i in range(4)]):
        self.x, self.y = float(box_coord[0]), float(box_coord[1])
        self.w, self.h = float(box_coord[2]), float(box_coord[3])
        self.c = 0.
        self.obj_prob = obj_prob
        self.class_probs = None if probs is None else np.array(probs)

    def get_score(self):
        return max(self.class_probs)

    def get_classindex(self):
        return np.argmax(self.class_probs)  # class_index = np.argmax(box.classes)

    def get_coordinates(self):
        return self.x, self.y, self.w, self.h


def overlap(x1, w1, x2, w2):
    l1 = x1 - w1 / 2.
    l2 = x2 - w2 / 2.
    left = max(l1, l2)
    r1 = x1 + w1 / 2.
    r2 = x2 + w2 / 2.
    right = min(r1, r2)
    return right - left


def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0: return 0;
    area = w * h
    return area


def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u


def box_iou(a, b):
    # Box intersect over union.
    return box_intersection(a, b) / box_union(a, b)


def prob_compare(box):
    return box.probs[box.class_num]


def prob_compare2(boxa, boxb):
    if (boxa.pi < boxb.pi):
        return 1
    elif (boxa.pi == boxb.pi):
        return 0
    else:
        return -1
