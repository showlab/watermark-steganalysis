import argparse
from dataclasses import dataclass
from typing import List, Callable, Dict, Any
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

@dataclass
class Pattern:
    num_images: int
    pattern: np.ndarray

@dataclass
class RemovalMethod:
    description: str
    diff_list: List[Pattern]
    removal_func: Callable
    kwargs: Dict[str, Any]

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def remove_excess_spaces(s: str):
    return ' '.join(s.split())

def sum_images(image_path, num_images, image_size):
    '''Returns a list of avged images [5 img avg, 10 img avg, 20 img avg, etc.]'''

    image_sum = None
    image_sum_list = []
    count_image = 0

    files = os.listdir(image_path)
    files = sorted(files)[:max(num_images)]

    for file in tqdm(files):
        if file.lower().endswith(('png', 'jpg', 'jpeg')):
            image = np.array(Image.open(os.path.join(image_path, file)).resize(image_size))

            if image_sum is None:
                image_sum = image.astype(float)
            else:
                image_sum += image.astype(float)
            count_image += 1

            if count_image in num_images:
                image_sum_list.append(Pattern(count_image, image_sum / count_image))

    return image_sum_list

def get_difference_list(clean_image_list, watermark_image_list, num_images):
    difference_list = []
    for num_images_index in range(len(num_images)):
        assert clean_image_list[num_images_index].num_images == watermark_image_list[num_images_index].num_images
        difference_list.append(Pattern(
            num_images[num_images_index],
            watermark_image_list[num_images_index].pattern - clean_image_list[num_images_index].pattern
        ))
    return difference_list

def no_removal(image: np.ndarray, *args, **kwargs):
    return image

def scale_removal(image: np.ndarray, watermark: np.ndarray, factor: int, watermark_bound: int, sign: bool, random_flip: bool = False, invert: bool = False):
    image = copy.deepcopy(image)
    watermark = copy.deepcopy(watermark)
    if invert:
        watermark *= -1
    if sign:
        watermark = np.sign(watermark)
    if random_flip:
        watermark *= np.sign(np.random.randn(*watermark.shape))
    if watermark_bound is not None:
        return np.clip(image - np.clip(factor * watermark, -watermark_bound, watermark_bound), 0, 255)
    return np.clip(image - factor * watermark, 0, 255)

def add_noise(image: np.ndarray, watermark: np.ndarray, std: float, **kwargs):
    return np.clip(image + np.random.randn(*image.shape) * std, 0, 255)

def evaluate_watermark_removal(eval_images_dir, num_eval_images, title, watermark_method, removal_methods, num_images, image_size, save_path = None):

    files = os.listdir(eval_images_dir)
    files = sorted([file for file in files if file.lower().endswith(('png', 'jpg', 'jpeg'))])[-num_eval_images:]

    for file in tqdm(files, desc = f'[{watermark_method}] {title}'):
        image = Image.open(os.path.join(eval_images_dir, file)).resize(image_size)
        for removal_method_index in range(len(removal_methods)):
            removal_method = removal_methods[removal_method_index]
            difference_list = removal_method.diff_list
            assert len(difference_list) == len(num_images)
            for num_images_index in range(len(num_images)):
                difference_instance = difference_list[num_images_index]
                image_removed = removal_method.removal_func(image = np.array(image).astype(float), watermark = difference_instance.pattern, **removal_method.kwargs)

                if save_path is not None:
                    this_image_save_path = os.path.join(save_path, remove_excess_spaces(removal_method.description), str(difference_instance.num_images), file.rsplit('.', 1)[0] + '.png')
                    create_path(os.path.dirname(this_image_save_path))
                    Image.fromarray(image_removed.astype(np.uint8)).save(this_image_save_path)

def visualise(diff_list, num_images):
    for num_images_index in range(len(num_images)):
        difference_pattern = diff_list[num_images_index].pattern
        difference_scale = 255 / (np.max(difference_pattern) - np.min(difference_pattern))
        difference_pattern -= np.min(difference_pattern)
        difference_pattern /= np.max(difference_pattern)
        
        plt.figure(figsize = (6, 6))
        plt.title(f'Blackbox $\Delta_{"{"}{num_images[num_images_index]}{"}"}$ ({difference_scale:.1f}x)')
        plt.imshow(difference_pattern)
        plt.axis('off')
        plt.show()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Watermark removal evaluation script')
    
    # Required arguments
    parser.add_argument('--watermark_method', type=str, required=True,
                       help='Name of the watermark method (e.g., Stable_Signature, RoSteALS)')
    
    parser.add_argument('--width', type=int, required=True,
                       help='Width of the images')
    parser.add_argument('--height', type=int, required=True,
                       help='Height of the images')
    
    # Paths
    parser.add_argument('--ood_clean_path', type=str, required=True,
                       help='Path to out-of-distribution non-watermarked images')
    parser.add_argument('--ind_clean_path', type=str, required=True,
                       help='Path to in-distribution non-watermarked images')
    parser.add_argument('--watermarked_path', type=str, required=True,
                       help='Path to watermarked images')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save watermark-removed images')
    
    # Optional arguments with defaults
    parser.add_argument('--num_images', nargs='+', type=int,
                       default=[5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
                       help='List of image counts to average')
    parser.add_argument('--num_eval_images', type=int, default=1000,
                       help='Number of images to use during evaluation')
    
    return parser.parse_args()

def setup_removal_methods(blackbox_watermarks: List, graybox_watermarks: List) -> List[RemovalMethod]:
    """Set up the removal methods with their parameters."""
    methods_config = [
        # Blackbox methods
        ("None                    [Blackbox]", blackbox_watermarks, no_removal, {}),
        ("StatWatermark           [Blackbox]", blackbox_watermarks, scale_removal, 
         {'factor': 1, 'watermark_bound': None, 'sign': False}),
        ("StatForge               [Blackbox]", blackbox_watermarks, scale_removal,
         {'factor': 1, 'watermark_bound': None, 'sign': False, 'invert': True}),
        
        # Graybox methods
        ("None                    [Graybox]", graybox_watermarks, no_removal, {}),
        ("StatWatermark           [Graybox]", graybox_watermarks, scale_removal,
         {'factor': 1, 'watermark_bound': None, 'sign': False}),
        ("StatForge               [Graybox]", graybox_watermarks, scale_removal,
         {'factor': 1, 'watermark_bound': None, 'sign': False, 'invert': True}),
    ]
    
    return [RemovalMethod(desc, diff, func, kwargs) 
            for desc, diff, func, kwargs in methods_config]

def main():
    args = parse_args()
    image_size = (args.width, args.height)
    
    # Process images
    ood_clean_images = sum_images(
        image_path=args.ood_clean_path,
        num_images=args.num_images,
        image_size=image_size
    )
    
    ind_clean_images = sum_images(
        image_path=args.ind_clean_path,
        num_images=args.num_images,
        image_size=image_size
    )
    
    watermarked_images = sum_images(
        image_path=args.watermarked_path,
        num_images=args.num_images,
        image_size=image_size
    )
    
    # Extract watermarks
    blackbox_watermarks = get_difference_list(
        ood_clean_images,
        watermarked_images,
        args.num_images
    )
    
    graybox_watermarks = get_difference_list(
        ind_clean_images,
        watermarked_images,
        args.num_images
    )
    
    # Setup removal methods
    removal_methods = setup_removal_methods(blackbox_watermarks, graybox_watermarks)
    
    # Evaluate watermark removal
    evaluate_watermark_removal(
        eval_images_dir=args.watermarked_path,
        num_eval_images=args.num_eval_images,
        title='Watermark removal',
        watermark_method=args.watermark_method,
        removal_methods=removal_methods,
        num_images=args.num_images,
        image_size=image_size,
        save_path=args.output_path
    )

if __name__ == "__main__":
    main()
