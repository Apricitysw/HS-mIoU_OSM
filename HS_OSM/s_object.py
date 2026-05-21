import numpy as np

def s_object(prediction, gt):
    """
    S_object computes the object similarity between foreground maps and ground
    truth 
    
    Args:
        prediction: Binary/Non binary foreground map with values in the range
                   [0, 1]. Type: numpy.ndarray (float).
        gt: Binary ground truth. Type: numpy.ndarray (bool).
        
    Returns:
        Q: The object similarity score (float).
    """
    
    # compute the similarity of the foreground in the object level
    prediction_fg = prediction.copy()
    prediction_fg[~gt] = 0
    o_fg = object_similarity(prediction_fg, gt)
    
    # compute the similarity of the background
    prediction_bg = 1.0 - prediction
    prediction_bg[gt] = 0
    o_bg = object_similarity(prediction_bg, ~gt)
    
    # combine the foreground measure and background measure together
    u = np.mean(gt)
    q = u * o_fg + (1 - u) * o_bg
    
    return q


def object_similarity(prediction, gt):
    """
    Helper function to compute object similarity.
    
    Args:
        prediction: Foreground/background prediction map.
        gt: Ground truth mask.
        
    Returns:
        score: Object similarity score.
    """
    
    # check the input
    if prediction.size == 0:
        return 0
    
    if not isinstance(prediction, np.ndarray):
        prediction = np.array(prediction)
    
    if prediction.dtype != np.float64:
        prediction = prediction.astype(np.float64)
    
    if prediction.max() > 1 or prediction.min() < 0:
        raise ValueError('prediction should be in the range of [0 1]')
    
    if not isinstance(gt, np.ndarray) or gt.dtype != bool:
        raise ValueError('GT should be of type: bool')
    
    # compute the mean of the foreground or background in prediction
    if np.sum(gt) == 0:
        return 0
    
    x = np.mean(prediction[gt])
    
    # compute the standard deviations of the foreground or background in prediction
    sigma_x = np.std(prediction[gt])
    
    score = 2.0 * x / (x**2 + 1.0 + sigma_x + np.finfo(float).eps)
    
    return score