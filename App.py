import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.models import load_model
import cv2
import os

plt.style.use("ggplot")

# Custom functions (loss and metrics) for model deserialization
from keras.saving import register_keras_serializable

@register_keras_serializable()
def dice_coefficients(y_true, y_pred, smooth=100):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
    return (2 * intersection + smooth) / (union + smooth)

@register_keras_serializable()
def dice_coefficients_loss(y_true, y_pred, smooth=100):
    return -dice_coefficients(y_true, y_pred, smooth)

@register_keras_serializable()
def iou(y_true, y_pred, smooth=100):
    intersection = K.sum(y_true * y_pred)
    sum = K.sum(y_true + y_pred)
    return (intersection + smooth) / (sum - intersection + smooth)

@register_keras_serializable()
def jaccard_distance(y_true, y_pred):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    return -iou(y_true_flatten, y_pred_flatten)

@register_keras_serializable()
def pixel_accuracy(y_true, y_pred):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    correct_predictions = K.sum(K.cast(K.equal(y_true_flatten, K.round(y_pred_flatten)), K.floatx()))
    total_pixels = K.cast(K.shape(y_true_flatten)[0], K.floatx())
    return correct_predictions / total_pixels

# Streamlit app
st.title("PLant Segmentation")

# Load model
model_path = r"D:\CSE428_project\plant_segment2.keras"
try:
    model = load_model(
        model_path,
        custom_objects={
            'dice_loss': dice_coefficients_loss,
            'mean_iou': iou,
            'dice_coef': dice_coefficients,
            'pixel_accuracy': pixel_accuracy
        }
    )
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Image dimensions
im_height = 256
im_width = 256

# File uploader
files = st.file_uploader(
    "Upload image files", type=["png", "jpg"], accept_multiple_files=True
)

if files:
    for file in files:
        # Display original image
        st.header("Original Image:")
        st.image(file)

        # Convert file to OpenCV format
        content = file.getvalue()
        image = np.asarray(bytearray(content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        img_resized = cv2.resize(image, (im_height, im_width))
        img_normalized = img_resized / 255.0
        img_input = img_normalized[np.newaxis, :, :, :]

        # Predict and display result
        if st.button(f"Predict Output for {file.name}"):
            try:
                pred_img = model.predict(img_input)
                
                # Rescale the predicted mask for visualization
                pred_mask = pred_img[0] * 255.0  # Rescale for visualization
                pred_mask = cv2.resize(pred_mask, (image.shape[1], image.shape[0]))  # Resize to original image size
                pred_mask = pred_mask.astype(np.uint8)  # Ensure correct format
                
                # Create overlay (add transparency to the mask)
                overlay = cv2.applyColorMap(pred_mask, cv2.COLORMAP_JET)  # Apply a colormap
                overlay = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)  # Blend original image with mask
        
                # Display results
                st.header("Predicted Mask:")
                st.image(pred_mask, caption="Predicted Mask", use_column_width=True)
        
                st.header("Overlayed Image:")
                st.image(overlay, caption="Original Image with Mask Overlay", use_column_width=True)
        
            except Exception as e:
                st.error(f"Error during prediction: {e}")
