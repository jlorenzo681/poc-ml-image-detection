import os
from PIL import Image, ImageDraw
import random

# Define parameters
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "..", "dataset_dummy")
classes = ['bedroom', 'bathroom', 'kitchen', 'living_room']
images_per_class = 20
img_size = (224, 224)

# Create dataset structure and dummy images
os.makedirs(data_dir, exist_ok=True)

for room_class in classes:
    class_dir = os.path.join(data_dir, room_class)
    os.makedirs(class_dir, exist_ok=True)
    for i in range(images_per_class):
        img_path = os.path.join(class_dir, f"{room_class}_{i+1}.jpg")
        
        # Create a simple image with a background color and some random shapes
        bg_color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
        img = Image.new('RGB', img_size, color=bg_color)
        d = ImageDraw.Draw(img)
        
        num_shapes = random.randint(1, 5)
        for _ in range(num_shapes):
            shape_type = random.choice(['rectangle', 'ellipse'])
            x1, y1 = random.randint(0, img_size[0]-50), random.randint(0, img_size[1]-50)
            x2, y2 = x1 + random.randint(20, 100), y1 + random.randint(20, 100)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            if shape_type == 'rectangle':
                d.rectangle([x1, y1, x2, y2], fill=color)
            else:
                d.ellipse([x1, y1, x2, y2], fill=color)
        
        # Draw class name roughly in middle
        d.text((img_size[0]//2 - 20, img_size[1]//2), room_class, fill=(255, 255, 255))
        img.save(img_path)

print(f"Dummy dataset generated successfully at {data_dir} with {images_per_class} images per class.")
