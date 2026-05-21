import numpy as np
from scipy.ndimage import center_of_mass

def s_region(prediction, gt):
    """
    S_region computes the region similarity between the foreground map and
    ground truth
    
    Args:
        prediction: Binary/Non binary foreground map with values in the range
                   [0, 1]. Type: numpy.ndarray (float).
        gt: Binary ground truth. Type: numpy.ndarray (bool).
        
    Returns:
        Q: The region similarity score (float).
    """
    
    # find the centroid of the GT
    x, y = centroid(gt)
    
    # divide GT into 4 regions
    gt_1, gt_2, gt_3, gt_4, w1, w2, w3, w4 = divide_gt(gt, x, y)
    
    # divide prediction into 4 regions
    pred_1, pred_2, pred_3, pred_4 = divide_prediction(prediction, x, y)
    
    # compute the ssim score for each region
    q1 = ssim(pred_1, gt_1)
    q2 = ssim(pred_2, gt_2)
    q3 = ssim(pred_3, gt_3)
    q4 = ssim(pred_4, gt_4)
    
    # sum the 4 scores
    q = w1 * q1 + w2 * q2 + w3 * q3 + w4 * q4
    
    return q


def centroid(gt):
    """
    Compute the centroid of the GT.
    
    Args:
        gt: Binary ground truth. Type: numpy.ndarray (bool).
        
    Returns:
        x, y: The coordinates of centroid (int, int).
    """
    rows, cols = gt.shape
    
    if np.sum(gt) == 0:
        x = round(cols / 2)
        y = round(rows / 2)
    else:
        total = np.sum(gt)
        i = np.arange(1, cols + 1)
        j = np.arange(1, rows + 1)
        x = round(np.sum(np.sum(gt, axis=0) * i) / total)
        y = round(np.sum(np.sum(gt, axis=1) * j) / total)
    
    return x, y


def divide_gt(gt, x, y):
    """
    Divide the GT into 4 regions according to the centroid of the GT and return the weights.
    
    Args:
        gt: Binary ground truth.
        x, y: Centroid coordinates.
        
    Returns:
        lt, rt, lb, rb: Four regions (left-top, right-top, left-bottom, right-bottom).
        w1, w2, w3, w4: Weights for each region.
    """
    # width and height of the GT
    hei, wid = gt.shape
    area = wid * hei
    
    # copy the 4 regions (adjusting for 0-based indexing)
    lt = gt[:y, :x]
    rt = gt[:y, x:wid]
    lb = gt[y:hei, :x]
    rb = gt[y:hei, x:wid]
    
    # the different weight (each block proportional to the GT foreground region)
    w1 = (x * y) / area
    w2 = ((wid - x) * y) / area
    w3 = (x * (hei - y)) / area
    w4 = 1.0 - w1 - w2 - w3
    
    return lt, rt, lb, rb, w1, w2, w3, w4


def divide_prediction(prediction, x, y):
    """
    Divide the prediction into 4 regions according to the centroid of the GT.
    
    Args:
        prediction: Prediction map.
        x, y: Centroid coordinates.
        
    Returns:
        lt, rt, lb, rb: Four regions.
    """
    # width and height of the prediction
    hei, wid = prediction.shape
    
    # copy the 4 regions (adjusting for 0-based indexing)
    lt = prediction[:y, :x]
    rt = prediction[:y, x:wid]
    lb = prediction[y:hei, :x]
    rb = prediction[y:hei, x:wid]
    
    return lt, rt, lb, rb


def ssim(prediction, gt):
    """
    SSIM computes the region similarity between foreground maps and ground
    truth
    
    Args:
        prediction: Binary/Non binary foreground map with values in the range
                   [0, 1]. Type: numpy.ndarray (float).
        gt: Binary ground truth. Type: numpy.ndarray (bool).
        
    Returns:
        Q: The region similarity score (float).
    """
    
    d_gt = gt.astype(np.float64)
    
    hei, wid = prediction.shape
    n = wid * hei
    
    # compute the mean of prediction, GT
    x = np.mean(prediction)
    y = np.mean(d_gt)
    
    # compute the variance of prediction, GT
    sigma_x2 = np.sum((prediction - x)**2) / (n - 1 + np.finfo(float).eps)
    sigma_y2 = np.sum((d_gt - y)**2) / (n - 1 + np.finfo(float).eps)
    
    # compute the covariance between prediction and GT
    sigma_xy = np.sum((prediction - x) * (d_gt - y)) / (n - 1 + np.finfo(float).eps)
    
    alpha = 4 * x * y * sigma_xy
    beta = (x**2 + y**2) * (sigma_x2 + sigma_y2)
    
    if alpha != 0:
        q = alpha / (beta + np.finfo(float).eps)
    elif alpha == 0 and beta == 0:
        q = 1.0
    else:
        q = 0
    
    return q