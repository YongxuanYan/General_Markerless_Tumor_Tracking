import os
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def create_template(image_path):
    """This is for background standardization, Create a binary mask from the input image where non-zero pixels become 1."""
    image = Image.open(image_path).convert('L')
    image_array = np.array(image)
    template = np.where(image_array == 0, 0, 1)
    return template


template_image_path = 'path/to/template/image'
st_mask = create_template(template_image_path)


class RealXRayGenerator:
    """Generates simulated X-ray images from DRR templates."""
    
    def __init__(self, drr_template_dir):
        self.drr_templates = [
            cv2.imread(os.path.join(drr_template_dir, img), cv2.IMREAD_GRAYSCALE)
            for img in sorted(os.listdir(drr_template_dir))
        ]
        self.template_names = sorted(os.listdir(drr_template_dir))

    def calculate_noise_probability(self, image, beta):
        """Calculate per-pixel noise probability based on intensity and distance from center."""
        h, w = image.shape
        normalized_image = image.astype(np.float32) / 255.0

        y, x = np.indices((h, w))
        center = (h // 2, w // 2)
        distance = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
        normalized_distance = distance / distance.max()

        base_prob_center = 0.82 # Lowest possiblity in image center
        base_prob_edge = 0.92
        distance_prob = base_prob_center + (base_prob_edge - base_prob_center) * normalized_distance

        gray_penalty = (1.0 - normalized_image) / 0.02 * 0.01
        gray_penalty = np.clip(gray_penalty, 0, None)

        final_prob = distance_prob - gray_penalty
        return np.clip(final_prob, 0, 1)

    def apply_spatially_variant_noise(self, image, mu, sigma, beta):
        """Apply Gaussian noise with spatially varying probability."""
        h, w = image.shape
        noise_prob_map = self.calculate_noise_probability(image, beta)

        random_map = np.random.random((h, w))
        noise = np.random.normal(mu, sigma, (h, w))
        noise_mask = random_map < noise_prob_map

        noisy_image = image.astype(np.float32) + noise * noise_mask
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def apply_mist_effect(self, image, alpha):
        """Apply mist effect via Gaussian blur and blending."""
        blurred_image = gaussian_filter(image.astype(np.float32), sigma=1 / alpha, mode='constant')
        mist_image = image.astype(np.float32) * 0.5 + blurred_image * 0.5
        return np.clip(mist_image, 0, 255).astype(np.uint8)

    def add_gaussian_noise(self, image, mu, sigma):
        """Add uniform Gaussian noise."""
        noise = np.random.normal(mu, sigma, image.shape).astype(np.float32)
        noisy_image = image.astype(np.float32) + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def generate_images(self, params):
        """Generate simulated X-ray images using the given parameters."""
        alpha, beta, mu, sigma = params
        generated_images = []
        for template in self.drr_templates:
            noised_image = self.apply_spatially_variant_noise(template, mu, sigma, beta)
            realx_image = self.apply_mist_effect(noised_image, alpha)
            realx_image = realx_image * st_mask
            generated_images.append(realx_image)
        return generated_images


if __name__ == "__main__":
    drr_input_dir = 'path/to/drr/templates'
    simulated_output_dir = 'path/to/simulated/xray/output'
    gaussian_output_dir = 'path/to/gaussian/noise/output'

    os.makedirs(simulated_output_dir, exist_ok=True)
    os.makedirs(gaussian_output_dir, exist_ok=True)

    generator = RealXRayGenerator(drr_input_dir)
    best_params = [0.68, 0.1, 8.9, 2.98]
    alpha, beta, mu, sigma = best_params

    simulated_images = generator.generate_images(best_params)
    simulated_txt_path = os.path.join(simulated_output_dir, 'parameters.txt')

    with open(simulated_txt_path, 'w') as file:
        file.write("Simulated X-ray images generated using parameters:\n")
        file.write(f"Alpha (Mist Effect): {alpha}\n")
        file.write(f"Beta (Noise Probability Factor): {beta}\n")
        file.write(f"Gaussian Noise Mean (mu): {mu}\n")
        file.write(f"Gaussian Noise Std (sigma): {sigma}\n")
        file.write("Noise probability rules:\n")
        file.write("- Center probability: 0.62, Edge probability: 0.72\n")
        file.write("- Gray value penalty: probability decreases 0.01 per 0.02 gray value decrease\n")

    for template, template_name in zip(simulated_images, generator.template_names):
        cv2.imwrite(os.path.join(simulated_output_dir, template_name), template)

    gaussian_txt_path = os.path.join(gaussian_output_dir, 'parameters.txt')
    with open(gaussian_txt_path, 'w') as file:
        file.write("Gaussian noise applied to DRR images with parameters:\n")
        file.write(f"Mean (mu): {mu}\n")
        file.write(f"Std (sigma): {sigma}\n")
        file.write(f"Gaussian blur applied with alpha: {alpha}\n")

    for template, template_name in zip(generator.drr_templates, generator.template_names):
        noisy_image = generator.add_gaussian_noise(template, mu, sigma)
        blurred_image = gaussian_filter(noisy_image.astype(np.float32), sigma=1 / alpha, mode='constant')
        final_image = noisy_image.astype(np.float32) * 0.5 + blurred_image * 0.5
        final_image = np.clip(final_image, 0, 255).astype(np.uint8)
        final_image = final_image * st_mask
        cv2.imwrite(os.path.join(gaussian_output_dir, template_name), final_image)

    print("All images and descriptions have been saved successfully!")
