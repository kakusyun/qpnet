import cv2
import numpy as np
import shapely
from shapely.geometry import Polygon
from qpnet.utils import logging

logger = logging.get_logger(__name__)


def calculate_polygon_IoM(grid, polygon_bbox):
    a = np.array(grid).reshape(4, 2)
    poly1 = Polygon(
        a).convex_hull
    b = np.array(polygon_bbox).reshape(4, 2)
    poly2 = Polygon(b).convex_hull
    try:
        inter_area = poly1.intersection(poly2).area
        min_area = min(poly1.area, poly2.area)
        if min_area == 0:
            iom = 0
        else:
            iom = float(inter_area) / min_area
    except shapely.geos.TopologicalError:
        logger.info('shapely.geos.TopologicalError occured, iou set to 0')
        iom = 0
    return iom


def calculate_polygon_IoU(grid, polygon_bbox):
    a = np.array(grid).reshape(4, 2)
    poly1 = Polygon(
        a).convex_hull
    b = np.array(polygon_bbox).reshape(4, 2)
    poly2 = Polygon(b).convex_hull
    try:
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.area + poly2.area - inter_area
        if union_area == 0:
            iou = 0
        else:
            iou = float(inter_area) / union_area
    except shapely.geos.TopologicalError:
        logger.info('shapely.geos.TopologicalError occured, iou set to 0')
        iou = 0
    return iou


def calculate_polygon_area(polygon_bbox):
    a = np.array(polygon_bbox).reshape(4, 2)
    poly1 = Polygon(a).convex_hull
    return poly1.area


def calculate_polygon_IoM_slow(grid, polygon_bbox, h, w):
    a = np.array(grid).reshape(4, 2)
    b = np.array(polygon_bbox).reshape(4, 2)
    img1 = np.zeros((h, w), np.uint8)
    img2 = np.zeros((h, w), np.uint8)
    cv2.fillPoly(img1, [a], 1)
    cv2.fillPoly(img2, [b], 1)
    return (img1 * img2).sum() / min(img1.sum(), img2.sum())
