import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np
from PIL import Image
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Any

# Define the model architecture
class ModifiedYOLOv1(nn.Module):
    grid_size: int = 13  # 13x13 grid
    num_boxes: int = 2  # 2 boxes per cell
    num_classes: int = 1  # "car"

    @nn.compact
    def __call__(self, x, train: bool = False):
        x = nn.Conv(64, kernel_size=(7, 7), strides=(2, 2), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = jax.nn.leaky_relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))  # 416/4 = 104

        x = nn.Conv(192, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = jax.nn.leaky_relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))  # 104/2 = 52

        x = nn.Conv(128, kernel_size=(1, 1), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = jax.nn.leaky_relu(x)

        x = nn.Conv(256, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = jax.nn.leaky_relu(x)

        x = nn.Conv(256, kernel_size=(1, 1), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = jax.nn.leaky_relu(x)

        x = nn.Conv(512, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = jax.nn.leaky_relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))  # 52/2 = 26

        x = nn.Conv(256, kernel_size=(1, 1), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = jax.nn.leaky_relu(x)

        x = nn.Conv(512, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = jax.nn.leaky_relu(x)

        x = nn.Conv(512, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = jax.nn.leaky_relu(x)

        x = nn.Conv(1024, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)  # 26/2 = 13
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = jax.nn.leaky_relu(x)

        out_channels = self.num_boxes * (4 + 1) + self.num_classes  # 11
        x = nn.Conv(out_channels, kernel_size=(1, 1), padding='SAME')(x)
        return x

# TrainState for managing params and batch_stats
class TrainState(train_state.TrainState):
    batch_stats: Any

# Load saved model parameters
dataset_dir = os.environ.get("COMVIS_DATASET_DIR", "./comvis")
save_path = os.path.join(dataset_dir, "model_params.pkl")
with open(save_path, 'rb') as f:
    model_state = pickle.load(f)

# Initialize model and state
model = ModifiedYOLOv1()
rng = jax.random.PRNGKey(0)
variables = model.init(rng, jnp.ones((1, 416, 416, 3)), train=False)
state = TrainState.create(
    apply_fn=model.apply,
    params=model_state['params'],
    batch_stats=model_state['batch_stats'],
    tx=optax.adam(1e-3)  # Optimizer not needed for inference, but required by TrainState
)

# Preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path).resize((416, 416))
    img_array = np.array(img) / 255.0  # Normalize to [0,1]
    return jnp.expand_dims(img_array, 0)

# Decode predictions
def decode_predictions(pred_grid, grid_size=13, conf_threshold=0.5):
    batch_size = pred_grid.shape[0]
    boxes = []
    scores = []
    for b in range(batch_size):
        pred_grid_b = pred_grid[b]  # Shape: (13, 13, 11)
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(2):  # 2 boxes per cell
                    offset = k * 5
                    conf = jax.nn.sigmoid(pred_grid_b[i, j, offset + 4])
                    if conf >= conf_threshold:
                        x = (j + jax.nn.sigmoid(pred_grid_b[i, j, offset])) * (1.0 / grid_size)
                        y = (i + jax.nn.sigmoid(pred_grid_b[i, j, offset + 1])) * (1.0 / grid_size)
                        w = pred_grid_b[i, j, offset + 2] ** 2  # Convert back from sqrt
                        h = pred_grid_b[i, j, offset + 3] ** 2  # Convert back from sqrt
                        x_min = x - w / 2
                        y_min = y - h / 2
                        x_max = x + w / 2
                        y_max = y + h / 2
                        boxes.append([x_min, y_min, x_max, y_max])
                        scores.append(float(conf))
    return np.array(boxes), np.array(scores)

# Visualize predictions
def visualize_predictions(image_path, boxes, scores, output_path):
    img = Image.open(image_path).resize((416, 416))
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for box, score in zip(boxes, scores):
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle((x_min * 416, y_min * 416), (x_max - x_min) * 416, (y_max - y_min) * 416,
                                linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min * 416, y_min * 416, f'{score:.2f}', color='r', fontsize=8)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Test the model
test_dir = os.path.join(dataset_dir, "test_images")
os.makedirs(test_dir, exist_ok=True)

# Example: Test on all images in test_images folder
test_images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]
if not test_images:
    print("No test images found in test_images folder. Please add images and rerun.")
else:
    for img_file in test_images:
        image_path = os.path.join(test_dir, img_file)
        processed_image = preprocess_image(image_path)
        pred_grid = state.apply_fn({'params': state.params, 'batch_stats': state.batch_stats}, processed_image, train=False)
        boxes, scores = decode_predictions(pred_grid)
        print(f"Image: {img_file}, Detected boxes: {len(boxes)}, Scores: {scores}")
        output_path = os.path.join(test_dir, f"predicted_{img_file}")
        visualize_predictions(image_path, boxes, scores, output_path)
        print(f"Prediction saved to {output_path}")

# Example: Test on a single image (uncomment and adjust path as needed)
single_image_path = "comvis/test1.jpg"
processed_image = preprocess_image(single_image_path)
pred_grid = state.apply_fn({'params': state.params, 'batch_stats': state.batch_stats}, processed_image, train=False)
boxes, scores = decode_predictions(pred_grid)
print(f"Detected boxes: {len(boxes)}, Scores: {scores}")
visualize_predictions(single_image_path, boxes, scores, "predicted_single_image.png")