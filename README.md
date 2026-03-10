# Real Estate Room Classification Proof of Concept

This is a proof of concept implementation of an image classification model to classify real estate room types (bedroom, bathroom, kitchen, living_room).

It uses **Transfer Learning** with a pre-trained **MobileNetV2** model fine-tuned on the real estate dataset using TensorFlow and Keras.

## Project Structure

*   `dataset/` - Directory where real-world images are placed, organized by class into subdirectories.
*   `src/create_dummy_dataset.py` - Script to generate synthetic sample images to test the model pipeline out of the box.
*   `src/train.py` - Reads the dataset, loads MobileNetV2, builds the classification head, compiles, and trains the newly structured model.
*   `src/predict.py` - Sample script that loads the saved `.keras` model and infers the category of a provided test image.
*   `requirements.txt` - Python dependencies needed for the application.

## Quick Start

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Generate dummy dataset** (so you can test without having real data yet):
    ```bash
    python src/create_dummy_dataset.py
    ```

3.  **Train the model**:
    ```bash
    python src/train.py
    ```

4.  **Test the model**:
    ```bash
    python src/predict.py
    ```
    OR to test a specific image:
    ```bash
    python src/predict.py /path/to/some/image.jpg
    ```
