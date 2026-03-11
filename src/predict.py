import tensorflow as tf
import numpy as np
import os
import sys

# Standard size for MobileNetV2
img_height = 224
img_width = 224

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "real_estate_room_classifier.keras")

if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}. Please run train.py first.")
    sys.exit(1)

model = tf.keras.models.load_model(model_path)

# In real scenario, test image path would be provided as argument
if len(sys.argv) > 1:
    test_image_path = sys.argv[1]
else:
    # Look for a sample image if not provided
    sample_img = os.path.join(current_dir, "..", "dataset", "bedroom", "bedroom_1.jpg")
    if not os.path.exists(sample_img):
        sample_img = os.path.join(current_dir, "..", "dataset_dummy", "bedroom", "bedroom_1.jpg")
        
    print(f"No image provided. Trying sample image: {sample_img}")
    if os.path.exists(sample_img):
        test_image_path = sample_img
    else:
        print("Error: No test image provided and no sample found.")
        sys.exit(1)

# Ensure class names match alphabetically exactly as they did in training
# (Because image_dataset_from_directory sorts folders alphabetically by default)
class_names = ['bathroom', 'bedroom', 'kitchen', 'living_room']

print(f"Loading image from: {test_image_path}")
img = tf.keras.utils.load_img(
    test_image_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch of 1

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to '{}' with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
