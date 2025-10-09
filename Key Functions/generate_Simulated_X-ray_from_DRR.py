import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter
from typing import Dict, Tuple

# =============================================================================
# --- CONFIGURATION ---
# =============================================================================

# --- I/O Directories ---
# Anonymized: Please replace with your actual paths.
INPUT_DIR = Path('/path/to/your/image_data/drr/class_0')
OUTPUT_DIR = Path('/path/to/your/image_data/drr/simulated_X_20251004')

# --- Model & Noise Parameters ---
V_MAX = 255.0  # Theoretical maximum pixel value.
NOISE_BLUR_SIGMA = 1.0  # Sigma for Gaussian filter to diffuse the noise.

# --- Polynomial Coefficients (Degree 10) for Content Dependency ---
# These coefficients model the probability and magnitude of noise based on pixel intensity.
PROB_COEFFS = np.array([
    -4.16140640e+00, 5.49064559e-01, -3.06594141e-02, 9.49310219e-04,
    -1.78007921e-05, 2.12708277e-07, -1.65394930e-09, 8.32232741e-12,
    -2.60897803e-14, 4.62481576e-17, -3.53588299e-20
])
MAG_COEFFS = np.array([
    -4.29686757e+01, 5.71962084e+00, -3.11870511e-01, 9.45399391e-03,
    -1.75439648e-04, 2.09435941e-06, -1.63778491e-08, 8.32438135e-11,
    -2.64339500e-13, 4.75459543e-16, -3.69212867e-19
])

# --- Parameter Ranges for Stochastic Degradation ---
# Defines the uniform sampling range for each noise parameter.
RANDOMIZATION_RANGES = {
    'Xc_std_range': (0, 14.2),
    'p_sigma_d': (103, 113),
    'p_C': (0.38, 0.42), 'p_B': (0.35, 0.37),
    'm_sigma_d': (103, 113),
    'm_C': (5.0, 6.0), 'm_B': (1.8, 2.0),
}


# =============================================================================
# --- CORE MATHEMATICAL MODELS ---
# =============================================================================

def polynomial_func(V: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Applies a 10th-degree polynomial function to the input array."""
    y = np.zeros_like(V, dtype=np.float64)
    for i, a in enumerate(coeffs):
        y += a * (V ** i)
    return y

def content_dependency_prob(V: np.ndarray) -> np.ndarray:
    """Calculates the content-dependent noise probability, clipped to [0, 1]."""
    raw_output = polynomial_func(V, PROB_COEFFS)
    return np.clip(raw_output, 0.0, 1.0)

def content_dependency_mag(V: np.ndarray) -> np.ndarray:
    """Calculates the content-dependent noise magnitude."""
    return polynomial_func(V, MAG_COEFFS)

def core_radial_modification(d: np.ndarray, sigma_d: float) -> np.ndarray:
    """
    Calculates the radial modification term based on distance from the beam center.
    This creates a non-uniform spatial noise distribution.
    """
    sigma_d = max(sigma_d, 1e-6)  # Avoid division by zero
    enhancement_term = 1.0 - np.exp(-d ** 2 / (2.0 * sigma_d ** 2))
    return enhancement_term


# =============================================================================
# --- NOISE GENERATION & APPLICATION ---
# =============================================================================

def add_masked_gaussian_noise(
    image: np.ndarray, noise_mask: np.ndarray, mean: float = 8, std_dev: float = 7
) -> np.ndarray:
    """
    Adds Gaussian noise to an image only at locations specified by a mask.

    Args:
        image: The input image array.
        noise_mask: A boolean array where True indicates where to add noise.
        mean: The mean of the Gaussian noise.
        std_dev: The standard deviation of the Gaussian noise.

    Returns:
        The noisy image, clipped to [0, 255] and converted to uint8.
    """
    noisy_image = image.astype(np.float32)
    noise = np.random.normal(mean, std_dev, image.shape)
    noisy_image[noise_mask] += noise[noise_mask]
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def calculate_noise_map(
    image_array: np.ndarray, params: Dict, is_prob_map: bool = True
) -> np.ndarray:
    """
    Calculates the noise probability (Np) or magnitude (Nm) map.

    Args:
        image_array: The input DRR image as a NumPy array.
        params: A dictionary of sampled stochastic parameters.
        is_prob_map: If True, calculates the probability map; otherwise, the magnitude map.

    Returns:
        The calculated noise map, scaled to its target range.
    """
    H, W = image_array.shape
    V = image_array.astype(np.float64)
    prefix = 'p_' if is_prob_map else 'm_'

    # Calculate radial distance 'd' from the shifted beam center
    center_y, center_x = H / 2.0, W / 2.0
    Y, X = np.indices((H, W))
    beam_center_x = center_x + params['X_c'][0]
    beam_center_y = center_y + params['X_c'][1]
    d = np.sqrt((X - beam_center_x) ** 2 + (Y - beam_center_y) ** 2)

    # Calculate spatial and content dependent terms
    F_d = core_radial_modification(d, params[prefix + 'sigma_d'])
    f_s = params[prefix + 'B'] + F_d

    if is_prob_map:
        f_v = content_dependency_prob(V)
        target_min, target_max = params['p_B'], 0.92
    else:
        f_v = content_dependency_mag(V)
        target_min, target_max = params['m_B'], 16.0

    # Combine terms and scale to the final target range
    n_map_unscaled = f_v * np.clip(f_s, 0.0, 1.0)
    min_val, max_val = np.min(n_map_unscaled), np.max(n_map_unscaled)
    
    # Avoid division by zero if the map is flat
    if (max_val - min_val) < 1e-6:
        return np.full(image_array.shape, target_min)
        
    scaled_n_map = (n_map_unscaled - min_val) / (max_val - min_val)  # Normalize to 0-1
    final_n_map = scaled_n_map * (target_max - target_min) + target_min
    return final_n_map


def apply_noise_degradation(image_path: Path) -> Image.Image | None:
    """
    Applies a full stochastic noise degradation process to a single image.

    The process includes:
    1. Randomly sampling noise parameters.
    2. Generating noise probability and magnitude maps.
    3. Applying initial scattered noise based on the maps.
    4. Diffusing the noise with a Gaussian blur.
    5. Adding a second layer of independent Gaussian noise for texture.

    Args:
        image_path: The path to the input DRR image.

    Returns:
        A PIL Image of the degraded image, or None if an error occurs.
    """
    try:
        img = Image.open(image_path).convert('L')
        img_array = np.array(img, dtype=np.float64)
    except Exception as e:
        print(f"Error loading {image_path.name}: {e}")
        return None

    # 1. Randomly sample parameters for this image
    params = {}
    Xc_std = np.random.uniform(*RANDOMIZATION_RANGES['Xc_std_range'])
    params['X_c'] = (
        int(np.round(np.random.normal(0, Xc_std))),
        int(np.round(np.random.normal(0, Xc_std)))
    )
    for key, range_tuple in RANDOMIZATION_RANGES.items():
        if key != 'Xc_std_range':
            params[key] = np.random.uniform(*range_tuple)

    # 2. Calculate noise probability and magnitude maps
    N_p_map = calculate_noise_map(img_array, params, is_prob_map=True)
    N_m_map = calculate_noise_map(img_array, params, is_prob_map=False)
    
    prob_map = np.clip(N_p_map, 0.0, 1.0)

    # 3. Apply initial scattered noise
    noise_mask_1 = np.random.rand(*img_array.shape) < prob_map
    safe_N_m_map = np.clip(N_m_map, 0.01, None)
    random_noise_values = np.random.normal(7, safe_N_m_map)
    
    initial_noise_array = np.zeros_like(img_array)
    initial_noise_array[noise_mask_1] = random_noise_values[noise_mask_1]

    # 4. Diffuse the noise with a Gaussian filter
    blurred_noise = gaussian_filter(
        initial_noise_array, sigma=NOISE_BLUR_SIGMA, mode='constant', cval=0.0
    )
    degraded_array = img_array + blurred_noise

    # 5. Add a second, independent layer of Gaussian noise for fine texture
    # This uses the same probability map but a new random roll, creating a
    # different mask that adds realism.
    noise_mask_2 = np.random.rand(*img_array.shape) < prob_map
    final_degraded_array = add_masked_gaussian_noise(
        degraded_array, noise_mask_2, mean=6, std_dev=8
    )

    return Image.fromarray(final_degraded_array)


# =============================================================================
# --- MAIN WORKFLOW ---
# =============================================================================

def main() -> None:
    """
    Main function to run the degradation process on all images in the input directory.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory is set to: {OUTPUT_DIR}")

    image_files = sorted(list(INPUT_DIR.glob('*.png')))
    if not image_files:
        print(f"Error: No PNG images found in {INPUT_DIR}. Please check the path.")
        return

    print(f"Found {len(image_files)} images. Starting degradation process...")
    for i, file_path in enumerate(image_files):
        # Progress indicator
        if (i + 1) % 100 == 0 or (i + 1) == len(image_files):
            print(f"Processing {i + 1}/{len(image_files)}: {file_path.name}")

        degraded_img = apply_noise_degradation(file_path)
        if degraded_img:
            output_path = OUTPUT_DIR / file_path.name
            degraded_img.save(output_path)

    print("\nDegradation process completed successfully.")
    print(f"All degraded images have been saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
