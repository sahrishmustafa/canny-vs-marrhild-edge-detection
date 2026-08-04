import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import scipy.io

# ===============================
# Utility Functions
# ===============================

def load_and_preprocess(image_path):
    img = Image.open(image_path).convert('L')
    img = np.array(img, dtype=np.float32)
    img /= 255.0
    return img

def load_ground_truth_mat(mat_path):
    """
    Loads and averages edge maps from a BSDS-style .mat ground truth file.
    Each .mat file contains several annotators with 'Boundaries' maps.
    Returns a normalized 2D numpy array (float32, range [0,1]).
    """
    data = scipy.io.loadmat(mat_path)
    if 'groundTruth' not in data:
        raise ValueError(f"'groundTruth' not found in {mat_path}. Keys: {list(data.keys())}")

    gt_structs = data['groundTruth']
    num_annotators = gt_structs.shape[1]

    edge_maps = []
    for i in range(num_annotators):
        entry = gt_structs[0, i]
        if isinstance(entry, np.ndarray):
            entry = entry[0, 0]  # unwrap nested structure

        # Extract the 'Boundaries' field
        if 'Boundaries' in entry.dtype.names:
            boundaries = entry['Boundaries'][0, 0]
            edge_maps.append(boundaries.astype(np.float32))
        else:
            raise ValueError(f"'Boundaries' field missing in annotator {i} of {mat_path}")

    # Average across all annotators
    avg_edge_map = np.mean(edge_maps, axis=0)

    # Normalize to [0, 1]
    if avg_edge_map.max() > 0:
        avg_edge_map /= avg_edge_map.max()

    return avg_edge_map


def save_edge_map(output_path, edge_map):
    plt.imsave(output_path, edge_map, cmap='gray')

def gaussian_kernel(size=5, sigma=1):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

def convolve_2d(image, kernel):
    """  
    Parameters:
        image: 2D numpy array
        kernel: 2D numpy array (convolution kernel)
    
    Returns:
        convolved: 2D numpy array (same size as image)
    """
    # Flip the kernel (for true convolution)
    kernel = np.flipud(np.fliplr(kernel))

    # Get kernel and image dimensions
    k_h, k_w = kernel.shape
    i_h, i_w = image.shape

    pad_h = k_h // 2
    pad_w = k_w // 2

    # Reflect padding (same as scipy.ndimage.convolve default)
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

    # Create output array
    output = np.zeros_like(image, dtype=np.float32)

    # Perform convolution
    for i in range(i_h):
        for j in range(i_w):
            region = padded[i:i+k_h, j:j+k_w]
            output[i, j] = np.sum(region * kernel)

    return output

def preprocess_for_edges(img):
    """Strong noise reduction before edge detection."""
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Strong Gaussian blur (large sigma)
    kernel = gaussian_kernel(size=9, sigma=2.0)
    blurred = convolve_2d(img, kernel)

    # Compute local mean and variance to suppress flat noise regions
    mean_kernel = np.ones((7,7)) / 49.0
    local_mean = convolve_2d(blurred, mean_kernel)
    diff = blurred - local_mean
    local_var = convolve_2d(diff**2, mean_kernel)
    mask = np.exp(-local_var / 0.02)  # smaller var -> heavier smoothing
    smoothed = mask * local_mean + (1 - mask) * blurred

    return smoothed

def clean_edges(edge_map, min_size=50):
    from scipy.ndimage import label, binary_erosion, binary_opening

    # Morphological open to remove small blobs
    opened = binary_opening(edge_map > 0.5)

    # Remove very small regions
    labeled, n = label(opened)
    sizes = np.bincount(labeled.ravel())
    mask = sizes >= min_size
    mask[0] = 0
    cleaned = mask[labeled]

    # Optional edge thinning
    thinned = binary_erosion(cleaned)
    return thinned.astype(np.float32)



# ===============================
# Manual Canny Implementation
# ===============================
# ===============================
# Heath–Sarkar Inspired Canny Implementation
# ===============================

def gaussian_smooth(img, sigma):
    """Apply separable Gaussian smoothing with reduced intensity (BOOSTBLURFACTOR)."""
    size = int(1 + 2 * np.ceil(2.5 * sigma))
    ax = np.arange(size) - size // 2
    kernel = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel /= kernel.sum()

    # Separable convolution: horizontal + vertical
    temp = convolve_2d(img, kernel.reshape(1, -1))
    smoothed = convolve_2d(temp, kernel.reshape(-1, 1))

    # Scale intensity (lower boost for normalized float images)
    BOOSTBLURFACTOR = 10.0
    return smoothed * BOOSTBLURFACTOR


def compute_gradients(img):
    """Compute gradients using central differences."""
    dx = np.zeros_like(img)
    dy = np.zeros_like(img)
    dx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    dy[1:-1, :] = img[2:, :] - img[:-2]
    mag = np.sqrt(dx**2 + dy**2)
    return dx, dy, mag


def non_max_suppression_subpixel(mag, dx, dy):
    """Perform non-maximum suppression along gradient direction."""
    nms = np.zeros_like(mag, dtype=np.uint8)
    rows, cols = mag.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            gx, gy = dx[i, j], dy[i, j]
            m = mag[i, j]
            if m == 0:
                continue

            angle = np.arctan2(gy, gx) * 180.0 / np.pi
            angle = angle % 180  # normalize to [0,180)

            # Direction-based neighbor comparison
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                before, after = mag[i, j-1], mag[i, j+1]
            elif (22.5 <= angle < 67.5):
                before, after = mag[i-1, j+1], mag[i+1, j-1]
            elif (67.5 <= angle < 112.5):
                before, after = mag[i-1, j], mag[i+1, j]
            else:
                before, after = mag[i-1, j-1], mag[i+1, j+1]

            if m >= before and m >= after:
                nms[i, j] = 128  # possible edge

    return nms


def hysteresis(mag, nms, tlow=0.2, thigh=0.6):
    """Perform double thresholding and recursive edge tracking."""
    mag = mag / (mag.max() + 1e-8)
    possible = (nms == 128)
    edge = np.zeros_like(nms, dtype=np.uint8)

    high = np.percentile(mag[possible], thigh * 100) if np.any(possible) else 0.3
    low = np.percentile(mag[possible], tlow * 100) if np.any(possible) else 0.1

    visited = np.zeros_like(edge, dtype=bool)
    rows, cols = mag.shape

    def follow_edges(i, j):
        if visited[i, j]:
            return
        visited[i, j] = True
        edge[i, j] = 255
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    if possible[ni, nj] and not visited[ni, nj] and mag[ni, nj] >= low:
                        follow_edges(ni, nj)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if possible[i, j] and mag[i, j] >= high and not visited[i, j]:
                follow_edges(i, j)

    return edge


def canny_edge_detection(img, sigma=1.2, tlow=0.25, thigh=0.7):
    """Improved Heath–Sarkar Canny tuned for normalized grayscale images."""
    smoothed = gaussian_smooth(img, sigma)
    dx, dy, mag = compute_gradients(smoothed)
    nms = non_max_suppression_subpixel(mag, dx, dy)
    edge = hysteresis(mag, nms, tlow, thigh)
    return edge / 255.0



# ===============================
# Marr–Hildreth (LoG)
# ===============================

def marr_hildreth_edge_detection(img, sigma=2.5, mag_thresh=0.015):
    # 1. Pre-smooth
    pre_kernel = gaussian_kernel(size=11, sigma=3.0)
    img = convolve_2d(img, pre_kernel)

    # 2. Construct LoG filter
    size = int(6 * sigma + 1)
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    LoG = ((xx**2 + yy**2 - 2*sigma**2) / sigma**4) * np.exp(-(xx**2 + yy**2) / (2*sigma**2))
    LoG -= LoG.mean()

    # 3. Apply LoG
    log_img = convolve_2d(img, LoG)
    log_img = log_img / (np.max(np.abs(log_img)) + 1e-8)

    # 4. Zero-crossing detection with magnitude + gradient constraint
    zero_cross = np.zeros_like(log_img)
    grad_kernel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])  # Sobel
    Gx = convolve_2d(img, grad_kernel)
    Gy = convolve_2d(img, grad_kernel.T)
    grad_mag = np.hypot(Gx, Gy)
    grad_mag /= grad_mag.max() + 1e-8

    for i in range(1, log_img.shape[0]-1):
        for j in range(1, log_img.shape[1]-1):
            patch = log_img[i-1:i+2, j-1:j+2]
            local_max = np.max(patch)
            local_min = np.min(patch)
            if (local_max * local_min < 0):  # zero-cross
                diff = local_max - local_min
                if diff > mag_thresh and grad_mag[i, j] > 0.2:
                    zero_cross[i, j] = 1.0

    # # 5. Optional edge thinning (suppress dense blobs)
    # blur_small = gaussian_kernel(size=3, sigma=0.7)
    # density = convolve_2d(zero_cross, blur_small)
    # zero_cross[density > 0.4] = 0  # remove clustered blob pixels

    return zero_cross



# ===============================
# Evaluation Metrics
# ===============================

# def evaluate_edges(pred, gt, tolerance=1):
#     """
#     Compare predicted edges with ground truth edge maps allowing spatial tolerance.
#     Args:
#         pred (np.ndarray): predicted edge map, float [0,1]
#         gt   (np.ndarray): ground truth edge map, float [0,1]
#         tolerance (int): number of pixels to tolerate in localization mismatch
#     Returns:
#         (precision, recall, f1)
#     """
#     # Normalize
#     pred = pred / (pred.max() + 1e-8)
#     gt = gt / (gt.max() + 1e-8)

#     # Binarize
#     pred_bin = pred > 0.5
#     gt_bin = gt > 0.2  # more lenient

#     # Allow ±tolerance pixels mismatch
#     struct = np.ones((2*tolerance+1, 2*tolerance+1), dtype=bool)
#     pred_dil = binary_dilation(pred_bin, structure=struct)
#     gt_dil = binary_dilation(gt_bin, structure=struct)

#     TP = np.sum(pred_bin & gt_dil)
#     FP = np.sum(pred_bin & ~gt_dil)
#     FN = np.sum(gt_bin & ~pred_dil)

#     precision = TP / (TP + FP + 1e-10)
#     recall = TP / (TP + FN + 1e-10)
#     f1 = 2 * precision * recall / (precision + recall + 1e-10)

#     return precision, recall, f1

def evaluate_edges(pred, gt):
    pred_bin = pred > 0.3
    gt_bin = gt > 0.2
    TP = np.sum(pred_bin & gt_bin)
    FP = np.sum(pred_bin & ~gt_bin)
    FN = np.sum(~pred_bin & gt_bin)

    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return precision, recall, f1


# ===============================
# Main
# ===============================

def main():
    image_dir = "images"
    gt_dir = "ground_truth"
    output_dir = "output/alter"
    os.makedirs(output_dir + "/canny", exist_ok=True)
    os.makedirs(output_dir + "/marr_hildreth", exist_ok=True)

    results = []
    count = 0

    for filename in os.listdir(image_dir):
        count+=1
        if count == 3:
            break

        if not filename.endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(image_dir, filename)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        if base_name.endswith("img"):
          base_name = base_name[:-3]

        gt_path = os.path.join(gt_dir, base_name + ".mat")

        img = load_and_preprocess(img_path)
        if gt_path.endswith('.mat'):
            gt = load_ground_truth_mat(gt_path)
        else:
            gt = load_and_preprocess(gt_path)


        # img_proc = preprocess_for_edges(img)
        # canny_edges = canny_edge_detection(img)
        canny_edges = canny_edge_detection(img, sigma=1.8, tlow=0.3, thigh=0.8)
        marr_edges = marr_hildreth_edge_detection(img, sigma=3.5, mag_thresh=0.02)


        # canny_edges = clean_edges(canny_edges)
        # marr_edges = clean_edges(marr_edges)

        save_edge_map(f"{output_dir}/canny/{filename}", canny_edges)
        save_edge_map(f"{output_dir}/marr_hildreth/{filename}", marr_edges)

        p1, r1, f1 = evaluate_edges(canny_edges, gt)
        p2, r2, f2 = evaluate_edges(marr_edges, gt)

        results.append([filename, "Canny", p1, r1, f1])
        results.append([filename, "Marr–Hildreth", p2, r2, f2])

    print("\n=== Edge Detection Evaluation ===")
    print("Image\tMethod\tPrecision\tRecall\tF1 Score")
    for row in results:
        print(f"{row[0]}\t{row[1]}\t{row[2]:.3f}\t{row[3]:.3f}\t{row[4]:.3f}")

if __name__ == "__main__":
    main()