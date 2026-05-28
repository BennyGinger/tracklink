import numpy as np



def temporal_mean(stack: np.ndarray, window: int = 5) -> np.ndarray:
    half = window // 2
    out = np.empty_like(stack, dtype=np.float32)

    for i in range(len(stack)):
        start = max(0, i - half)
        stop = min(len(stack), i + half + 1)
        out[i] = stack[start:stop].mean(axis=0)

    return out


def temporal_variance(stack: np.ndarray, window: int = 5) -> np.ndarray:
    half = window // 2
    out = np.empty_like(stack, dtype=np.float32)

    for i in range(len(stack)):
        start = max(0, i - half)
        stop = min(len(stack), i + half + 1)
        out[i] = stack[start:stop].std(axis=0)

    return out


def frame_difference(stack: np.ndarray) -> np.ndarray:
    out = np.zeros_like(stack, dtype=np.float32)
    out[1:] = np.abs(stack[1:].astype(np.float32) - stack[:-1].astype(np.float32))
    return out

def normalize(img):
    p1 = np.percentile(img, 1)
    p99 = np.percentile(img, 99)

    img = np.clip(img, p1, p99)
    img = (img - p1) / (p99 - p1)

    return img

def normalize01(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    img = img.astype(np.float32)
    lo = img.min()
    hi = img.max()
    return (img - lo) / (hi - lo + eps)


def enhance_with_variance(
    raw_stack: np.ndarray,
    window: int = 5,
    alpha: float = 0.75,
) -> np.ndarray:
    var_stack = temporal_variance(raw_stack, window=window)

    enhanced = []
    for raw, var in zip(raw_stack, var_stack):
        raw_n = normalize01(raw)
        var_n = normalize01(var)
        enh = raw_n + alpha * var_n
        enhanced.append(normalize01(enh))

    return np.stack(enhanced, axis=0)

def optical_flow_motion_stack(stack: np.ndarray) -> np.ndarray:
    import cv2
    import numpy as np

    stack = stack.astype(np.float32)
    out = np.zeros_like(stack, dtype=np.float32)

    for t in range(1, len(stack)):
        prev = stack[t - 1]
        curr = stack[t]

        flow = cv2.calcOpticalFlowFarneback(
            prev,
            curr,
            None, # type: ignore
            pyr_scale=0.5,
            levels=1,
            winsize=5,
            iterations=3,
            poly_n=3,
            poly_sigma=1.1,
            flags=0,
        )

        dx = flow[..., 0]
        dy = flow[..., 1]

        out[t] = np.sqrt(dx**2 + dy**2)

    return out

def temporal_median_background(stack: np.ndarray, window: int = 15) -> np.ndarray:
    half = window // 2
    out = np.empty_like(stack, dtype=np.float32)
    x = stack.astype(np.float32)

    for i in range(len(x)):
        start = max(0, i - half)
        stop = min(len(x), i + half + 1)
        out[i] = np.median(x[start:stop], axis=0)

    return out



if __name__ == "__main__":
    from fits_io import FitsIO
    from tifffile import imwrite
    from pathlib import Path
    from scipy.ndimage import gaussian_filter
    import cv2
    import numpy as np
    
    path = Path("/media/ben/Analysis/Python/Images/NeutrophilTrackingTest/dia/c1133-MaxIP_s9/fits_array.tif")
    
    reader = FitsIO.from_path(path)
    array = reader.get_array()
    if isinstance(array, list):
        array = array[0]
    print(f"Original array shape: {array.shape}, dtype: {array.dtype} and axes: {reader.axes}")
    
    red_channel = array[:, 1, :, :]
    print(f"Red channel shape: {red_channel.shape}, dtype: {red_channel.dtype}")
    
    smoothed = gaussian_filter(red_channel, sigma=(0, 1, 1))
    temp_var = temporal_variance(smoothed, window=5)
    imwrite(path.with_name(path.stem + "_temporal_variance.tif"), temp_var)
    
    # flow = optical_flow_motion_stack(red_channel)
    # print(f"Optical flow shape: {flow.shape}, dtype: {flow.dtype}")
    # imwrite(path.with_name(path.stem + "_optical_flow.tif"), flow)
    
    # motion = temp_var * flow
    # print(f"Motion stack shape: {motion.shape}, dtype: {motion.dtype}")
    # imwrite(path.with_name(path.stem + "_motion_stack.tif"), motion)
    
    norm = temp_var / (temp_var.max())*100000
    norm = norm.astype(np.uint16)
    norm[norm < 200] = 0
    # norm = np.clip(norm, 0, 65535)
    imwrite(path.with_name(path.stem + "_normalized.tif"), norm)
    
    
    background = temporal_median_background(red_channel, window=15)
    clean = red_channel.astype(np.float32) - background
    clean[clean < 0] = 0
    imwrite(path.with_name(path.stem + "_background_subtracted.tif"), clean.astype(np.uint16))
    