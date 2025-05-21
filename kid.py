import os
import torch
import numpy as np
from torchvision.models import inception_v3
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

# Preprocessing images for InceptionV3
def preprocess_images(images, image_size=299):
    transform = Compose([
        Resize((image_size, image_size)),
        CenterCrop(image_size),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return torch.stack([transform(img) for img in images])

# Extract features using InceptionV3
def extract_features(images, model, device):
    model.eval()
    images = images.to(device)
    with torch.no_grad():
        features = model(images)
    return features.cpu().numpy()

# Polynomial kernel for MMD
def polynomial_kernel(x, y, degree=3, coef0=1, gamma=None):
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    return (gamma * np.dot(x, y.T) + coef0) ** degree

# Compute KID with random sampling buckets
def compute_kid_with_buckets(x, y, kernel_func=polynomial_kernel, bucket_size=20, num_buckets=1):
    n_x, n_y = len(x), len(y)
    if n_x < bucket_size or n_y < bucket_size:
        raise ValueError("Bucket size must be smaller than the dataset size.")

    kid_scores = []
    for _ in range(num_buckets):
        x_sample = x[np.random.choice(n_x, bucket_size, replace=False)]
        y_sample = y[np.random.choice(n_y, bucket_size, replace=False)]

        kxx = kernel_func(x_sample, x_sample).mean()
        kyy = kernel_func(y_sample, y_sample).mean()
        kxy = kernel_func(x_sample, y_sample).mean()
        kid_scores.append(kxx + kyy - 2 * kxy)

    return np.mean(kid_scores), np.std(kid_scores)

# Load all .png images from a folder
def load_images_from_folder(folder_path):
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]
    return [Image.open(f).convert("RGB") for f in image_files]

# Run KID across multiple subfolders (buckets)
def evaluate_kid_for_folders(path_a, path_b, device='cuda', bucket_size=20, num_buckets=1):
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.fc = torch.nn.Identity()  # Use penultimate layer

    results = {}
    buckets = sorted(os.listdir(path_a))  # Automatically detect bucket folders

    for bucket in buckets:
        a_folder = os.path.join(path_a, bucket, "subset")
        b_folder = os.path.join(path_b, bucket, "subset")

        if not os.path.exists(a_folder) or not os.path.exists(b_folder):
            print(f"Skipping {bucket}: missing folder(s)")
            continue

        images_a = load_images_from_folder(a_folder)
        images_b = load_images_from_folder(b_folder)

        if len(images_a) < bucket_size or len(images_b) < bucket_size:
            print(f"Skipping {bucket}: insufficient images (<{bucket_size})")
            continue

        images_a = preprocess_images(images_a)
        images_b = preprocess_images(images_b)

        features_a = extract_features(images_a, model, device)
        features_b = extract_features(images_b, model, device)

        kid_mean, kid_std = compute_kid_with_buckets(
            features_a, features_b, bucket_size=bucket_size, num_buckets=num_buckets
        )

        results[bucket] = {"mean": kid_mean, "std": kid_std}
        print(f"{bucket}: Score Mean = {kid_mean:.5f}, Std = {kid_std:.5f}")

    return results

# Aggregate scores across all buckets
def aggregate_kid_results(results):
    means = [v["mean"] for v in results.values()]
    stds = [v["std"] for v in results.values()]
    avg_mean = np.mean(means)
    avg_std = np.sqrt(np.mean(np.square(stds)))  # RMS of stds
    return avg_mean, avg_std

if __name__ == "__main__":

    path_before = "set_a"  # Before intervention
    path_after = "set_b"   # After intervention

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = evaluate_kid_for_folders(
        path_before, path_after,
        device=device,
        bucket_size=20,
        num_buckets=5
    )

    print("\nFinal Summary:")
    for bucket, stats in results.items():
        print(f"{bucket}: Mean = {stats['mean']:.5f}, Std = {stats['std']:.5f}")

    avg_mean, avg_std = aggregate_kid_results(results)
    print(f"\nAggregate Score: Mean = {avg_mean:.5f}, Std = {avg_std:.5f}")
