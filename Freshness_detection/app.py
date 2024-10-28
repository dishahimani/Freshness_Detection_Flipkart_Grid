import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image



def preprocess_image_pudina(image):
    """Preprocess the image to remove shadows and background."""
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Normalize the V channel to reduce the effect of shadows
    v_channel = hsv[:, :, 2]
    v_channel = cv2.normalize(v_channel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Replace the V channel with the normalized version
    hsv[:, :, 2] = v_channel

    # Convert back to BGR
    image_normalized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(image_normalized, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image to create a binary mask
    _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours to isolate the object
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the largest contour
    object_mask = np.zeros_like(gray)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(object_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Apply the mask to the original image
    preprocessed_image = cv2.bitwise_and(image, image, mask=object_mask)

    return preprocessed_image, object_mask

def detect_freshness_pudina(image, preprocessed_image, object_mask):

    # Convert to RGB and HSV color spaces
    image_rgb = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2HSV)

    # Define HSV range for green areas (fresh leaves)
    lower_green = np.array([35, 40, 40])   # Adjusted for green
    upper_green = np.array([85, 255, 255])  # Adjusted for green

    # Define HSV range for yellow areas (starting to spoil)
    lower_yellow = np.array([20, 40, 40])  # Yellow/brown color ranges
    upper_yellow = np.array([35, 255, 255])

    # Define HSV range for dark spots (spoiled regions)
    lower_dark = np.array([0, 0, 0])       # Dark regions (blackish spots)
    upper_dark = np.array([180, 255, 50])

    # Create masks for each color region
    mask_green = cv2.inRange(image_hsv, lower_green, upper_green)
    mask_yellow = cv2.inRange(image_hsv, lower_yellow, upper_yellow)
    mask_dark = cv2.inRange(image_hsv, lower_dark, upper_dark)

    # Count pixels for each mask
    green_pixels = np.sum(mask_green > 0)
    yellow_pixels = np.sum(mask_yellow > 0)
    dark_pixels = np.sum(mask_dark > 0)
    total_pixels = cv2.countNonZero(object_mask)
    
    # Calculate percentages of each region
    green_percentage = (green_pixels / total_pixels) * 100
    yellow_percentage = (yellow_pixels / total_pixels) * 100
    dark_percentage = (dark_pixels / total_pixels) * 100

    print(f"Green coverage: {green_percentage:.2f}%")
    print(f"Yellow coverage: {yellow_percentage:.2f}%")
    print(f"Dark spots coverage: {dark_percentage:.2f}%")

    # Freshness classification based on detected features
    if green_percentage > 90 and yellow_percentage < 5 and dark_percentage < 2:
        freshness = "100% Fresh Pudina: Fully green, no yellowing or dark spots."
    elif green_percentage > 70 and yellow_percentage < 10 and dark_percentage < 5:
        freshness = "75% Fresh Pudina: Mostly green with minor yellowing or dark spots."
    elif green_percentage > 50 and (yellow_percentage > 10 or dark_percentage > 10):
        freshness = "50% Fresh Pudina: Noticeable yellowing or dark spots, but still mostly green."
    elif yellow_percentage > 20 or dark_percentage > 15:
        freshness = "25% Fresh Pudina: Significant yellowing or dark spots, spoiling."
    else:
        freshness = "Rotten Pudina: Large areas with yellowing or dark spots, spoiled."

    print(f"Pudina Freshness: {freshness}")
    discolored_areas = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_dark)
    return(freshness, discolored_areas)

def preprocess_image_cap(image):
    """Preprocess the image to isolate the capsicum."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Normalize the V channel to reduce shadows
    v_channel = hsv[:, :, 2]
    v_channel = cv2.normalize(v_channel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    hsv[:, :, 2] = v_channel

    image_normalized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(image_normalized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold to create binary mask
    _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # Find largest contour (capsicum region)
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    object_mask = np.zeros_like(gray)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(object_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    preprocessed_image = cv2.bitwise_and(image, image, mask=object_mask)
    return preprocessed_image, object_mask

def detect_freshness_capsicum(image, preprocessed_image, object_mask):

    # Convert to RGB and HSV color spaces
    image_rgb = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2HSV)

    # Define HSV range for green (fresh capsicum)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(image_hsv, lower_green, upper_green)

    # Define HSV range for yellow and brownish regions (indicating spoilage)
    lower_yellow_brown = np.array([15, 40, 40])
    upper_yellow_brown = np.array([35, 255, 255])
    mask_yellow_brown = cv2.inRange(image_hsv, lower_yellow_brown, upper_yellow_brown)

    # Exclude green areas from the mask of yellow/brown regions
    mask_discoloration = cv2.bitwise_and(mask_yellow_brown, mask_yellow_brown, mask=cv2.bitwise_not(mask_green))

    # Wrinkle detection using Laplacian for edge detection
    gray = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    wrinkles = np.var(laplacian)

    # Count the number of discolored pixels
    discolored_pixels = np.sum(mask_discoloration > 0)
    total_pixels = cv2.countNonZero(object_mask)
    discoloration_percentage = (discolored_pixels / total_pixels) * 100

    print(f"Discoloration coverage: {discoloration_percentage:.2f}%")
    print(f"Wrinkle level (variance in texture): {wrinkles:.2f}")

    # Classification logic for capsicum freshness
    if discoloration_percentage <= 1 and wrinkles < 300:
        freshness = "100% Fresh Capsicum: Entirely green, no wrinkles or discoloration."
    elif discoloration_percentage <= 10 and wrinkles < 500:
        freshness = "75% Fresh Capsicum: Mostly green with few wrinkles or small discolored spots."
    elif discoloration_percentage <= 25 or wrinkles >= 500:
        freshness = "50% Fresh Capsicum: Noticeable wrinkles or discoloration, but still some green areas."
    elif discoloration_percentage <= 40 or wrinkles >= 1000:
        freshness = "25% Fresh Capsicum: Significant wrinkles and discoloration."
    else:
        freshness = "Rotten Capsicum: Large discolored areas and wrinkled surface."

    print(f"Capsicum Freshness: {freshness}")

    # Display discolored areas
    discolored_areas = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_discoloration)
    return(freshness, discolored_areas)




def preprocess_image_banana(image):
    """Preprocess the image without removing the background."""
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Normalize the V channel to reduce the effect of shadows
    v_channel = hsv[:, :, 2]
    v_channel = cv2.normalize(v_channel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Replace the V channel with the normalized version
    hsv[:, :, 2] = v_channel

    # Convert back to BGR
    image_normalized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return image_normalized

def detect_freshness_banana(image, preprocessed_image):

    # Convert to RGB and HSV color spaces
    image_rgb = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2HSV)
    
    # Define HSV ranges for yellow (fresh banana), green (slightly unripe), and dark spots (rotten)
    lower_yellow = np.array([20, 100, 100])  # Lower bound for yellow
    upper_yellow = np.array([30, 255, 255])  # Upper bound for yellow
    
    lower_green = np.array([35, 50, 50])  # Lower bound for green
    upper_green = np.array([85, 255, 255])  # Upper bound for green
    
    lower_brown = np.array([10, 100, 20])    # Lower bound for brown/dark spots
    upper_brown = np.array([30, 255, 100])   # Upper bound for brown/dark spots

    # Create masks for yellow, green, and brown regions
    mask_yellow = cv2.inRange(image_hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(image_hsv, lower_green, upper_green)
    mask_brown = cv2.inRange(image_hsv, lower_brown, upper_brown)

    # Calculate the percentage of yellow, green, and brown areas
    yellow_pixels = np.sum(mask_yellow > 0)
    green_pixels = np.sum(mask_green > 0)
    brown_pixels = np.sum(mask_brown > 0)
    total_pixels = image.shape[0] * image.shape[1]
    
    yellow_percentage = (yellow_pixels / total_pixels) * 100
    green_percentage = (green_pixels / total_pixels) * 100
    brown_percentage = (brown_pixels / total_pixels) * 100

    # Print percentages for debugging
    print(f"Yellow coverage: {yellow_percentage:.2f}%")
    print(f"Green coverage: {green_percentage:.2f}%")
    print(f"Brown spots coverage: {brown_percentage:.2f}%")
    
    # Classify freshness based on the percentage of yellow, green, and brown spots
    if yellow_percentage >= 11 and brown_percentage <= 1:
        freshness = "100% Fresh Banana: Entirely yellow or mostly yellow with no significant spots."
    elif yellow_percentage >= 10 and brown_percentage <= 5:
        freshness = "75% Fresh Banana: Predominantly yellow with very minimal dark spots."
    elif yellow_percentage >= 40 and brown_percentage <= 10:
        freshness = "50% Fresh Banana: Noticeable spots, but still mostly yellow."
    elif yellow_percentage >= 20 or brown_percentage <= 2:
        freshness = "25% Fresh Banana: Significant spots, more green or brown areas visible."
    else:
        freshness = "Rotten Banana: More than 15% dark spots, spoiled."

    print(f"Banana Freshness: {freshness}")
    
    # Apply the brown spots mask to the original image
    spoiled_spots = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_brown)

    return(freshness, spoiled_spots)



def preprocess_image_broccoli(image):
    """Preprocess the image to remove shadows and background."""
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Normalize the V channel to reduce the effect of shadows
    v_channel = hsv[:, :, 2]
    v_channel = cv2.normalize(v_channel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Replace the V channel with the normalized version
    hsv[:, :, 2] = v_channel

    # Convert back to BGR
    image_normalized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(image_normalized, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image to create a binary mask
    _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours to isolate the object
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the largest contour
    object_mask = np.zeros_like(gray)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(object_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Apply the mask to the original image
    preprocessed_image = cv2.bitwise_and(image, image, mask=object_mask)

    return preprocessed_image, object_mask

def detect_dark_spots_broccoli(image, preprocessed_image, object_mask):

    # Convert to RGB and HSV color spaces
    image_rgb = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2HSV)

    
    # Refined HSV range for green (fresh broccoli)
    lower_green = np.array([40, 40, 40])  # Lower bound for green
    upper_green = np.array([90, 255, 255])  # Upper bound for green
    
    # Refined HSV range for brown spots (indicating spoilage)
    lower_brown = np.array([10, 100, 20])    # Narrow the lower bound for brown
    upper_brown = np.array([30, 255, 150])   # Upper bound for brown

    # Refined HSV range for yellowish spots (indicating spoilage)
    lower_yellow = np.array([20, 100, 100])  # Lower bound for yellow
    upper_yellow = np.array([30, 255, 255])   # Upper bound for yellow

    # Create masks for green, brown, and yellow regions
    mask_green = cv2.inRange(image_hsv, lower_green, upper_green)
    mask_brown = cv2.inRange(image_hsv, lower_brown, upper_brown)
    mask_yellow = cv2.inRange(image_hsv, lower_yellow, upper_yellow)
    
    # Combine both brown and yellow masks for spoilage detection
    mask_spots = cv2.bitwise_or(mask_brown, mask_yellow)

    # Calculate the percentage of green and spots
    green_pixels = np.sum(mask_green > 0)
    dark_pixels = np.sum(mask_spots > 0)
    total_pixels = image.shape[0] * image.shape[1]
    
    green_percentage = (green_pixels / total_pixels) * 100
    dark_percentage = (dark_pixels / total_pixels) * 100

    # Print percentages for debugging
    print(f"Green coverage: {green_percentage:.2f}%")
    print(f"Dark spots coverage (brown/yellow): {dark_percentage:.2f}%")
    
    # Adjust the classification based on the green and dark spot percentages
    if green_percentage >= 80 and dark_percentage <= 2:
        freshness = "100% Fresh Broccoli: Entirely green or almost entirely green with no significant spots."
    elif dark_percentage <= 2:
        freshness = "100% Fresh Broccoli: Predominantly green with very minimal dark spots."
    elif dark_percentage <= 10:
        freshness = "75% Fresh Broccoli: Mostly green with a few small dark spots."
    elif dark_percentage <= 15:
        freshness = "50% Fresh Broccoli: Noticeable spots, but still mostly green."
    elif dark_percentage <= 40:
        freshness = "25% Fresh Broccoli: Significant spots found"
    else:
        freshness = "Rotten Broccoli: More than 40% spots, spoiled."

    print(f"Broccoli Freshness: {freshness}")
    
    # Apply the spots mask to the original image
    spoiled_spots = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_spots)

    return(freshness, spoiled_spots)


def detect_dark_spots_cauli(preprocessed_image, object_mask):
    """Detect dark spots on cauliflower and classify its freshness"""

    # Convert to RGB and HSV color spaces
    image_rgb = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2HSV)


    # Define refined HSV range for dark spots (brown/black regions)
    lower_dark = np.array([0, 0, 20])     # Lower bound for dark spots
    upper_dark = np.array([180, 255, 80])  # Upper bound for dark spots

    # Threshold the HSV image to get only dark spots
    mask_dark = cv2.inRange(image_hsv, lower_dark, upper_dark)

    # Define HSV range for green color to exclude it from the dark spots
    lower_green = np.array([40, 40, 40])   # Adjusted lower bound for green
    upper_green = np.array([90, 255, 255])  # Adjusted upper bound for green

    # Create a mask to exclude green areas
    mask_green = cv2.inRange(image_hsv, lower_green, upper_green)

    # Combine masks to exclude green areas from the dark spots
    mask_dark_excluded_green = cv2.bitwise_and(mask_dark, mask_dark, mask=cv2.bitwise_not(mask_green))

    # Further refine dark spots by excluding any light green or yellow areas
    lower_yellowish_green = np.array([20, 40, 40])   # Lower bound for yellow-greenish areas
    upper_yellowish_green = np.array([40, 255, 255])  # Upper bound for yellow-greenish areas
    mask_yellowish_green = cv2.inRange(image_hsv, lower_yellowish_green, upper_yellowish_green)

    # Exclude yellowish green areas from the dark spots mask
    mask_final = cv2.bitwise_and(mask_dark_excluded_green, mask_dark_excluded_green, mask=cv2.bitwise_not(mask_yellowish_green))

    dark_spots = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_final)

    # Count the number of dark pixels
    dark_pixels = np.sum(mask_final > 0)
    total_pixels = cv2.countNonZero(object_mask)
    dark_percentage = (dark_pixels / total_pixels) * 100

    # Count the number of white pixels
    white_pixels = np.sum(cv2.bitwise_and(cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY), object_mask) > 200)
    white_percentage = (white_pixels / total_pixels) * 100

    print(f"Dark spots coverage: {dark_percentage:.2f}%")
    print(f"White coverage: {white_percentage:.2f}%")

    # Adjust the classification thresholds here
    if dark_percentage <= 1:
        freshness = "100% Fresh Cauliflower: Entirely white, or almost entirely white with no visible dark spots."
    elif dark_percentage <= 10:
        freshness = "75% Fresh Cauliflower: Mostly white with a few small dark spots."
    elif dark_percentage <= 25:
        freshness = "50% Fresh Cauliflower: Noticeable dark spots, but still mostly white."
    elif dark_percentage <= 40:
        freshness = "25% Fresh Cauliflower: Significant dark spots, equal white and dark areas."
    else:
        freshness = "Rotten Cauliflower: More than 40% dark spots, spoiled."

    print(f"Cauliflower Freshness: {freshness}")

    return(freshness, dark_spots)

def preprocess_image(image_path):
    """Preprocess the image to remove shadows and background."""
    # Convert to HSV color space
    hsv = cv2.cvtColor(image_path, cv2.COLOR_BGR2HSV)

    # Normalize the V channel to reduce the effect of shadows
    v_channel = hsv[:, :, 2]
    v_channel = cv2.normalize(v_channel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Replace the V channel with the normalized version
    hsv[:, :, 2] = v_channel

    # Convert back to BGR
    image_normalized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(image_normalized, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image to create a binary mask
    _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours to isolate the object
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the largest contour
    object_mask = np.zeros_like(gray)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(object_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Apply the mask to the original image
    preprocessed_image = cv2.bitwise_and(image_path, image_path, mask=object_mask)
    return preprocessed_image, object_mask



def main():
    st.title("Freshness Detection App")

    # Upload an image
    uploaded_image = st.file_uploader("Choose an image of a vegetable")

    if uploaded_image is not None:
        # Convert the uploaded image to a NumPy array
        img1 = Image.open(uploaded_image)
        img = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)

        # Create a dropdown select box
        option = st.selectbox("Choose the vegetable:", ["Cauliflower", "Broccoli", "Banana", "Capsicum","Pudina"])

        # Apply the selected operation
        if option == "Cauliflower":
            preprocessed_image, object_mask = preprocess_image(img)
            text_display, processed_img = detect_dark_spots_cauli(preprocessed_image, object_mask)
        elif option == "Broccoli":
            preprocessed_image, object_mask = preprocess_image_broccoli(img)
            text_display, processed_img = detect_dark_spots_broccoli(img, preprocessed_image, object_mask)
        elif option == "Banana":
            preprocessed_image = preprocess_image_banana(img)
            text_display, processed_img = detect_freshness_banana(img, preprocessed_image)
        elif option == "Capsicum":
            preprocessed_image, object_mask = preprocess_image_cap(img)
            text_display, processed_img = detect_freshness_capsicum(img, preprocessed_image, object_mask)
        elif option == "Pudina":
            preprocessed_image, object_mask = preprocess_image_pudina(img)
            text_display, processed_img = detect_freshness_pudina(img, preprocessed_image, object_mask)

        # Display the processed image
        st.write(text_display)
        st.write("Original Image:")
        st.image(img1, caption=f"This is original image")
        st.write("Masked Image:")
        st.image(processed_img, caption=f"Processed image using {option}")
        

if __name__ == '__main__':
    main()