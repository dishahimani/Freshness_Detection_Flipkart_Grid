
import torch
import streamlit as st
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision.ops import nms

# Load the Faster R-CNN model pre-trained on COCO dataset
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define transformation to convert image to tensor
transform = T.Compose([T.ToTensor()])

def func1(image, image_tensor):

    # Run the image through the model
    with torch.no_grad():
        # Send image tensor as a batch (even if it's a single image)
        predictions = model([image_tensor])

    # Extract predictions (bounding boxes, labels, and scores)
    predicted_boxes = predictions[0]['boxes'].cpu().numpy()
    predicted_labels = predictions[0]['labels'].cpu().numpy()
    predicted_scores = predictions[0]['scores'].cpu().numpy()

    # Filter predictions by a higher confidence threshold
    confidence_threshold = 0.7  # Adjust threshold based on your dataset
    high_confidence_boxes = predicted_boxes[predicted_scores >= confidence_threshold]
    high_confidence_labels = predicted_labels[predicted_scores >= confidence_threshold]
    high_confidence_scores = predicted_scores[predicted_scores >= confidence_threshold]

    # Apply Non-Maximum Suppression (NMS)
    keep = nms(torch.tensor(high_confidence_boxes), torch.tensor(high_confidence_scores), iou_threshold=0.5)

    # Only keep the boxes, labels, and scores after NMS
    nms_boxes = high_confidence_boxes[keep]
    nms_labels = high_confidence_labels[keep]
    nms_scores = high_confidence_scores[keep]

    # Count the number of objects detected after NMS
    num_objects_detected = len(nms_boxes)
    print(f"Number of objects detected: {num_objects_detected}")

    # Define a function to draw bounding boxes on the image
    def draw_boxes(image, boxes):
        image_with_boxes = np.array(image.copy())
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Red box
        return Image.fromarray(image_with_boxes)

    # Draw the bounding boxes on the original image
    image_with_boxes = draw_boxes(image, nms_boxes)
    return(num_objects_detected, image_with_boxes)




def main():
    st.title("Object Detection & Counting App")

    # Upload an image
    uploaded_image = st.file_uploader("Choose an image")

    if uploaded_image is not None:
        # Convert the uploaded image to a NumPy array
        img1 = Image.open(uploaded_image)
        img = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
        image_tensor = transform(img1)

        # Create a dropdown select box
        option = st.selectbox("Choose the otpion:", ["Object Counting", "Brand Detection"])

        # Apply the selected operation
        if option == "Object Counting":
            text_display, processed_img = func1(img1, image_tensor)
        elif option == "Brand Detection":
            preprocessed_image, object_mask = preprocess_image_broccoli(img)
            text_display, processed_img = detect_dark_spots_broccoli(img, preprocessed_image, object_mask)

        # Display the processed image
        st.write(text_display)
        st.image(processed_img, caption=f"Processed image using {option}")
        

if __name__ == '__main__':
    main()