import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, zoom
import scipy.io

###############################################################################################################################################################
# Helper Function
###############################################################################################################################################################

COUNT = 10

def load_and_preprocess(image_path):
    img = Image.open(image_path).convert('L')
    img = np.array(img, dtype=np.float32)
    img /= 255.0
    return img

def load_ground_truth_mat(mat_path):
    data = scipy.io.loadmat(mat_path)
    if 'groundTruth' not in data:
        raise ValueError(f"'groundTruth' not found in {mat_path}. Keys: {list(data.keys())}")

    gt_structs = data['groundTruth']
    edge_maps = []

    for i in range(gt_structs.shape[1]):
        entry = gt_structs[0, i]
        # Access the 'Boundaries' field
        boundaries = entry['Boundaries'][0, 0]
        edge_maps.append(boundaries.astype(np.float32))

    avg_edge_map = np.mean(edge_maps, axis=0)
    avg_edge_map /= (avg_edge_map.max() + 1e-8)

    # Force it to 2D
    if avg_edge_map.ndim > 2:
        avg_edge_map = avg_edge_map.squeeze()
    elif avg_edge_map.ndim < 2:
        raise ValueError(f"Invalid GT shape from {mat_path}: {avg_edge_map.shape}")

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

    # Reflect padding
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
    kernel = gaussian_kernel(size=11, sigma=3.0)
    blurred = convolve_2d(img, kernel)

    # Compute local mean and variance to suppress flat noise regions
    mean_kernel = np.ones((7,7)) / 49.0
    local_mean = convolve_2d(blurred, mean_kernel)
    diff = blurred - local_mean
    local_var = convolve_2d(diff**2, mean_kernel)
    mask = np.exp(-local_var / 0.01)  # smaller var -> heavier smoothing
    smoothed = mask * local_mean + (1 - mask) * blurred

    return smoothed
###############################################################################################################################################################
# Manual Canny Implementation
###############################################################################################################################################################

def canny_edge_detection(img, low_ratio=0.4, high_percentile=95):
    # 1. Gaussian smoothing
    # In preprocess_for_edges
    kernel = gaussian_kernel(size=11, sigma=3.0)
    smoothed = convolve_2d(img, kernel)

    # 2. Sobel gradients
    Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    Ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    Ix = convolve_2d(smoothed, Kx) # grad vector - x
    Iy = convolve_2d(smoothed, Ky) # grad vector - y

    mag = np.hypot(Ix, Iy)
    mag = mag / (mag.max() + 1e-8) # edge strength 
    theta = np.arctan2(Iy, Ix) # alpha

    # 3. Non-maximum suppression
    Z = np.zeros_like(mag) # all zeros 
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, mag.shape[0]-1):
        for j in range(1, mag.shape[1]-1):
            q = 255
            r = 255

            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = mag[i, j+1] # right neighbour
                r = mag[i, j-1] # left neighbour
            elif (22.5 <= angle[i,j] < 67.5):
                q = mag[i+1, j-1] # top right neighbour
                r = mag[i-1, j+1] # bottom right neighbour
            elif (67.5 <= angle[i,j] < 112.5):
                q = mag[i+1, j] # above neighbour
                r = mag[i-1, j] # below neighbour
            elif (112.5 <= angle[i,j] < 157.5):
                q = mag[i-1, j-1] # top left neighbour
                r = mag[i+1, j+1] # bottom left neighbour

            if (mag[i,j] >= q) and (mag[i,j] >= r) and (mag[i,j] > 0.08):
                Z[i,j] = mag[i,j] # retain

    # 4. Adaptive double threshold
    high = np.percentile(Z, high_percentile)
    low = high * low_ratio
    strong, weak = 1.0, 0.3
    res = np.zeros_like(Z)
    strong_i, strong_j = np.where(Z >= high)
    weak_i, weak_j = np.where((Z >= low) & (Z < high))
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    # 5. Hysteresis 
    mean_mag = np.mean(Z)
    for i in range(1, res.shape[0]-1):
        for j in range(1, res.shape[1]-1):
            if res[i,j] == weak:
                if np.any(res[i-1:i+2, j-1:j+2] == strong) and Z[i,j] > (1.2 * mean_mag):
                    res[i,j] = strong
                else:
                    res[i,j] = 0

    res = res * (mag > np.percentile(mag, 70))
    return res


###############################################################################################################################################################
# Marr–Hildreth (LoG)
###############################################################################################################################################################
def marr_hildreth_edge_detection(img, sigma=2.5, mag_thresh=0.015):
    # 1. Pre-smooth
    pre_kernel = gaussian_kernel(size=13, sigma=5.0)
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

    return zero_cross

###############################################################################################################################################################
# evaluate edges
###############################################################################################################################################################

def evaluate_edges(pred, gt, tolerance=1):
    """
    Compare predicted edges with BSDS-style ground truth (possibly multiple annotators).
    Uses spatial tolerance to handle slight misalignments.
    """

    if gt.shape != pred.shape:
        gt = zoom(gt, (
            pred.shape[0] / gt.shape[0],
            pred.shape[1] / gt.shape[1]
        ), order=1)

    # Normalize and binarize
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)

    pred_bin = pred > 0.3
    gt_bin = gt > 0.2

    # Apply spatial tolerance (so small shifts don’t kill TP)
    struct = np.ones((2*tolerance+1, 2*tolerance+1), dtype=bool)
    pred_dil = binary_dilation(pred_bin, structure=struct)
    gt_dil = binary_dilation(gt_bin, structure=struct)

    TP = np.sum(pred_bin & gt_dil)
    FP = np.sum(pred_bin & ~gt_dil)
    FN = np.sum(gt_bin & ~pred_dil)

    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return precision, recall, f1

###############################################################################################################################################################
# main
###############################################################################################################################################################

def main():
    import pandas as pd
    import numpy as np

    image_dir = "images"
    gt_dir = "ground_truth"
    output_dir = "output"
    os.makedirs(output_dir + "/canny", exist_ok=True)
    os.makedirs(output_dir + "/marr_hildreth", exist_ok=True)

    # Metric containers
    canny_precisions, canny_recalls, canny_f1s = [], [], []
    marr_precisions, marr_recalls, marr_f1s = [], [], []
    image_names = []

    results = []
    count = 0

    for filename in sorted(os.listdir(image_dir)):
        count += 1
        
        # if count > COUNT: break

        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(image_dir, filename)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        if base_name.endswith("img"):
            base_name = base_name[:-3]
        gt_path = os.path.join(gt_dir, base_name + ".mat")

        # === Load image + ground truth ===
        img = load_and_preprocess(img_path)
        if gt_path.endswith('.mat'):
            gt = load_ground_truth_mat(gt_path)
        else:
            gt = load_and_preprocess(gt_path)

        # === Preprocess for noise suppression ===
        img_proc = preprocess_for_edges(img)

        # === Run Canny ===
        canny_edges = canny_edge_detection(img_proc, high_percentile=97.5, low_ratio=0.25)
        save_edge_map(f"{output_dir}/canny/{filename}", canny_edges)
        p1, r1, f1 = evaluate_edges(canny_edges, gt, tolerance=2)
        canny_precisions.append(p1)
        canny_recalls.append(r1)
        canny_f1s.append(f1)

        # === Run Marr–Hildreth ===
        marr_edges = marr_hildreth_edge_detection(img, sigma=3.5, mag_thresh=0.02)
        save_edge_map(f"{output_dir}/marr_hildreth/{filename}", marr_edges)
        p2, r2, f2 = evaluate_edges(marr_edges, gt, tolerance=2)
        marr_precisions.append(p2)
        marr_recalls.append(r2)
        marr_f1s.append(f2)

        image_names.append(filename)

        results.append([filename, "Canny", p1, r1, f1])
        results.append([filename, "Marr–Hildreth", p2, r2, f2])

        print(f"{filename}")
        print(f"  Canny        → P={p1:.3f}, R={r1:.3f}, F1={f1:.3f}")
        print(f"  Marr–Hildreth → P={p2:.3f}, R={r2:.3f}, F1={f2:.3f}")

    def summarize_metrics(prec, rec, f1, name):
        prec, rec, f1 = map(np.array, [prec, rec, f1])
        print(f"\n=== {name} Overall Evaluation ===")
        print(f"Average Precision: {np.mean(prec):.3f} ± {np.std(prec):.3f}")
        print(f"Average Recall:    {np.mean(rec):.3f} ± {np.std(rec):.3f}")
        print(f"Average F1-score:  {np.mean(f1):.3f} ± {np.std(f1):.3f}")
        return np.mean(prec), np.mean(rec), np.mean(f1)

    mean_c_p, mean_c_r, mean_c_f = summarize_metrics(canny_precisions, canny_recalls, canny_f1s, "Canny")
    mean_m_p, mean_m_r, mean_m_f = summarize_metrics(marr_precisions, marr_recalls, marr_f1s, "Marr–Hildreth")

    df = pd.DataFrame({
        "image": image_names,
        "canny_precision": canny_precisions,
        "canny_recall": canny_recalls,
        "canny_f1": canny_f1s,
        "marr_precision": marr_precisions,
        "marr_recall": marr_recalls,
        "marr_f1": marr_f1s
    })
    df.to_csv("edge_metrics_comparison.csv", index=False)
    print("\nSaved all metrics to edge_metrics_comparison.csv")

    print("\nOverall Comparison Summary")
    print(f"{'Method':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print(f"{'Canny':<15} {mean_c_p:.3f}        {mean_c_r:.3f}        {mean_c_f:.3f}")
    print(f"{'Marr–Hildreth':<15} {mean_m_p:.3f}        {mean_m_r:.3f}        {mean_m_f:.3f}")


if __name__ == "__main__":
    main()
