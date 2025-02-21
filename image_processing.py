import numpy as np
from PIL import Image

def resize_image(image, max_size=1024):
    """
    Resize the image to a maximum dimension (width or height) while maintaining the aspect ratio.
    :param image: Input image as a NumPy array.
    :param max_size: Maximum dimension (width or height) for the resized image.
    :return: Resized image as a NumPy array.
    """
    height, width = image.shape[:2]
    
    # Check if resizing is needed
    if max(height, width) <= max_size:
        return image
    
    # Compute new dimensions while maintaining aspect ratio
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    # Convert NumPy array to PIL image, resize it, and convert back to NumPy array
    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return np.asarray(pil_image, dtype=np.uint8)
