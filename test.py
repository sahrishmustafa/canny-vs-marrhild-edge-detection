# import scipy.io
# from edge_detection_manual import load_ground_truth_mat

# data = scipy.io.loadmat("ground_truth/2018.mat")
# print(data.keys())

# gt = data['groundTruth']
# print(type(gt))
# print(gt.shape)

# gt = load_ground_truth_mat("ground_truth/2018.mat")
# print(gt.shape, gt.dtype, gt.min(), gt.max())

# import matplotlib.pyplot as plt
# plt.imshow(gt, cmap='gray')
# plt.title("Averaged Ground Truth Edge Map")
# plt.show()

import os
import scipy.io
import matplotlib.pyplot as plt

# Path to .mat file
mat_path = "ground_truth/2018.mat"

# Load .mat data
data = scipy.io.loadmat(mat_path)
gt = data['groundTruth']

# Extract the first annotator's boundary map
boundaries = gt[0, 0]['Boundaries'][0, 0]

print("Boundaries shape:", boundaries.shape, "dtype:", boundaries.dtype)
print("Min:", boundaries.min(), "Max:", boundaries.max())

# Create folder if it doesn't exist
save_dir = "examples/groundtruth"
os.makedirs(save_dir, exist_ok=True)

# Build output path (e.g. examples_groundtruth/100007_gt.png)
base_name = os.path.splitext(os.path.basename(mat_path))[0]
save_path = os.path.join(save_dir, f"{base_name}_gt.png")

# Save image
plt.imsave(save_path, boundaries, cmap='gray')
print(f"✅ Saved ground truth edge map to: {save_path}")

# Optional: Display it too
plt.imshow(boundaries, cmap='gray')
plt.title("Ground Truth Edge Map (Annotator 1)")
plt.axis('off')
plt.show()
