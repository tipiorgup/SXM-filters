import pandas as pd
import numpy as np
import cv2

def extract_features_to_dataframe(data_dict, target_size=(128, 128), include_image_pixels=False):
    """
    Extract comprehensive image features and return as pandas DataFrame
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing image data and metadata
    target_size : tuple
        Target size for image resizing (height, width)
    include_image_pixels : bool
        Whether to include flattened image pixels as features (can be very large)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with samples as rows and features as columns
    """
    
    names = list(data_dict.keys())
    all_data = []
    
    for name in names:
        # Process image
        img = data_dict[name]['img']
        if len(img.shape) == 2:  # Grayscale
            resized = cv2.resize(img, target_size)
            gray = img.copy()
        else:  # Color
            resized = cv2.resize(img, target_size)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Start building feature dictionary for this sample
        feature_dict = {'sample_name': name}
        
        # Original metadata features
        feature_dict['bias'] = data_dict[name]['bias']
        feature_dict['entropy'] = data_dict[name]['entropy']
        feature_dict['acq_time'] = data_dict[name]['acq_time']
        
        # Intensity features
        feature_dict['max_intensity'] = np.max(img)
        feature_dict['min_intensity'] = np.min(img)
        feature_dict['intensity_range'] = feature_dict['max_intensity'] - feature_dict['min_intensity']
        feature_dict['mean_intensity'] = np.mean(img)
        
        # Edge features
        feature_dict['edge_density'] = calculate_edge_density_from_array(img)
        
        # Object counting features
        feature_dict['num_contours'] = count_objects_contours(gray)
        
        # Texture features
        feature_dict['contrast'] = calculate_contrast(gray)
        feature_dict['homogeneity'] = calculate_homogeneity(gray)
        
        # Shape/structure features
        feature_dict['laplacian_variance'] = calculate_sharpness(gray)
        feature_dict['gradient_magnitude'] = calculate_gradient_strength(gray)
        
        # Frequency domain features
        freq_high, freq_low = calculate_frequency_features(gray)
        feature_dict['freq_energy_high'] = freq_high
        feature_dict['freq_energy_low'] = freq_low
        feature_dict['freq_ratio_high_low'] = freq_high / (freq_low + 1e-8)  # Avoid division by zero

        # Entropy
        feature_dict['entropy']=calculate_entropy_from_array(img)

        
        all_data.append(feature_dict)

    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Set sample_name as index
    df.set_index('sample_name', inplace=True)
    
    print(f"Features: {list(df.columns)}")
    
    return df

def calculate_entropy_from_array(image_array, bins=256):
    """Calculate entropy directly from numpy array"""
    # Calculate histogram
    hist, _ = np.histogram(image_array.ravel(), bins=bins)
    
    # Calculate probability distribution (remove zeros to avoid log(0))
    prob_dist = hist[hist > 0] / hist.sum()
    
    # Calculate entropy (base 2 for bits)
    image_entropy = entropy(prob_dist, base=2)
    
    return image_entropy

def calculate_entropy_from_array(image_array, bins=256):
    """Calculate entropy directly from numpy array"""
    # Calculate histogram
    hist, _ = np.histogram(image_array.ravel(), bins=bins)
    
    # Calculate probability distribution (remove zeros to avoid log(0))
    prob_dist = hist[hist > 0] / hist.sum()
    
    # Calculate entropy (base 2 for bits)
    image_entropy = entropy(prob_dist, base=2)
    
    return image_entropy

def count_objects_contours(gray_img, min_area=50):
    """Count objects using contour detection"""
    # Convert to uint8 if needed
    if gray_img.dtype != np.uint8:
        # Normalize to 0-255 range and convert to uint8
        gray_img = ((gray_img - gray_img.min()) / (gray_img.max() - gray_img.min()) * 255).astype(np.uint8)
    
    # Threshold the image
    _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    return len(valid_contours)

def calculate_edge_density_from_array(img_array, threshold1=50, threshold2=150):
    """Calculate edge density from numpy array instead of file path"""
    try:
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_array.copy()
        
        # Convert to uint8 if needed
        if gray.dtype != np.uint8:
            gray = ((gray - gray.min()) / (gray.max() - gray.min()) * 255).astype(np.uint8)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, threshold1, threshold2)
        
        # Calculate edge density (percentage of edge pixels)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        return edge_density
    except:
        return 0

def calculate_sharpness(gray_img):
    """Calculate sharpness using Laplacian variance"""
    # Convert to appropriate type for Laplacian
    if gray_img.dtype != np.uint8:
        gray_img = ((gray_img - gray_img.min()) / (gray_img.max() - gray_img.min()) * 255).astype(np.uint8)
    
    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
    return laplacian.var()

def calculate_gradient_strength(gray_img):
    """Calculate average gradient magnitude"""
    # Convert to appropriate type for Sobel
    if gray_img.dtype != np.uint8:
        gray_img = ((gray_img - gray_img.min()) / (gray_img.max() - gray_img.min()) * 255).astype(np.uint8)
    
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return np.mean(gradient_magnitude)

def calculate_contrast(gray_img):
    """Calculate local contrast using standard deviation of pixel intensities"""
    return np.std(gray_img) / np.mean(gray_img) if np.mean(gray_img) > 0 else 0

def calculate_homogeneity(gray_img):
    """Calculate homogeneity (inverse of contrast)"""
    # Using local binary patterns or simple variance measure
    kernel = np.ones((5,5), np.float32) / 25
    smooth = cv2.filter2D(gray_img.astype(np.float32), -1, kernel)
    variance = np.var(gray_img - smooth)
    return 1 / (1 + variance)

def calculate_sharpness(gray_img):
    """Calculate sharpness using Laplacian variance"""
    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
    return laplacian.var()

def calculate_gradient_strength(gray_img):
    """Calculate average gradient magnitude"""
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return np.mean(gradient_magnitude)

def calculate_frequency_features(gray_img):
    """Calculate frequency domain features using FFT"""
    # Apply FFT
    fft = np.fft.fft2(gray_img)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    
    # Get center coordinates
    h, w = gray_img.shape
    center_y, center_x = h // 2, w // 2
    
    # Create frequency masks
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # High frequency (edges, details)
    high_freq_mask = distance > min(h, w) * 0.1
    high_freq_energy = np.sum(magnitude[high_freq_mask])
    
    # Low frequency (general structure)
    low_freq_mask = distance <= min(h, w) * 0.1
    low_freq_energy = np.sum(magnitude[low_freq_mask])
    
    return high_freq_energy, low_freq_energy
