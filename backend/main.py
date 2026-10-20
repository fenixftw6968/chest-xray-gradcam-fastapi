import os
import json
import io
import base64
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image

# Suppress TensorFlow logging verbosity
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from contextlib import asynccontextmanager
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_saved_model()
    yield

# Initialize FastAPI App
app = FastAPI(
    title="AI-Powered Medical Image Analyzer API",
    description="Educational and research API for chest X-ray screening prototype using MobileNetV2.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration for React + Vite development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local dev and cloud testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "medical_model.keras")
CONFIG_PATH = os.path.join(BASE_DIR, "models", "model_config.json")

# Global variables for loaded model and metadata
model: Optional[tf.keras.Model] = None
model_config: dict = {
    "target_size": [224, 224],
    "class_names": ["NORMAL", "PNEUMONIA"],
    "last_conv_layer_name": "Conv_1"
}


def load_saved_model():
    """Load the trained Keras model and configuration once at startup."""
    global model, model_config
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                model_config = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load config from {CONFIG_PATH}: {e}")

    if os.path.exists(MODEL_PATH):
        try:
            print(f"Loading model from {MODEL_PATH}...")
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Model file not found at {MODEL_PATH}.")
        model = None

# Load model on module import for immediate availability
load_saved_model()


def make_gradcam_heatmap(img_array: np.ndarray, keras_model: tf.keras.Model, last_conv_layer_name: str = "Conv_1") -> Optional[np.ndarray]:
    """
    Computes Grad-CAM heatmap for the binary classification model,
    supporting both direct layers and nested MobileNetV2 base models.
    """
    try:
        # Check if the model has a nested base model (e.g. mobilenetv2_1.00_224)
        base_model = None
        for layer in keras_model.layers:
            if "mobilenetv2" in layer.name.lower() or hasattr(layer, "layers"):
                base_model = layer
                break

        if base_model is not None:
            # Build feature extractor for the base model
            target_conv = base_model.get_layer(last_conv_layer_name)
            sub_model = tf.keras.models.Model(
                inputs=base_model.inputs,
                outputs=[target_conv.output, base_model.output]
            )

            # Build head model for subsequent layers
            head_inputs = tf.keras.Input(shape=base_model.output.shape[1:])
            x = head_inputs
            head_layer_found = False
            for layer in keras_model.layers:
                if layer == base_model:
                    head_layer_found = True
                    continue
                if head_layer_found:
                    x = layer(x)
            head_model = tf.keras.models.Model(inputs=head_inputs, outputs=x)

            with tf.GradientTape() as tape:
                conv_outputs, base_features = sub_model(img_array)
                tape.watch(conv_outputs)
                preds = head_model(base_features)
                loss = preds[:, 0]

            grads = tape.gradient(loss, conv_outputs)
        else:
            grad_model = tf.keras.models.Model(
                inputs=[keras_model.inputs],
                outputs=[keras_model.get_layer(last_conv_layer_name).output, keras_model.output]
            )
            with tf.GradientTape() as tape:
                conv_outputs, preds = grad_model(img_array)
                loss = preds[:, 0]
            grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            return None

        # Mean gradient intensity per feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU to retain only positive contributions
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()
    except Exception as e:
        print(f"Grad-CAM generation error: {e}")
        return None


def overlay_gradcam(original_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> str:
    """
    Superimposes heatmap onto the original image and returns a base64 encoded PNG string.
    """
    import matplotlib as mpl

    # Resize heatmap to match image dimensions
    heatmap_uint8 = np.uint8(255 * heatmap)
    jet = mpl.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]

    jet_heatmap = Image.fromarray((jet_heatmap * 255).astype(np.uint8))
    jet_heatmap = jet_heatmap.resize(original_image.size, resample=Image.BICUBIC)

    # Convert original to RGB if not already
    orig_rgb = original_image.convert("RGB")
    overlay = Image.blend(orig_rgb, jet_heatmap, alpha=alpha)

    buffered = io.BytesIO()
    overlay.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@app.get("/")
def root():
    return {
        "message": "AI-Powered Medical Image Analyzer API is active",
        "documentation": "/docs",
        "disclaimer": "For educational and research purposes only. This tool is not intended for medical diagnosis."
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "instructions": (
            "Model is ready for inference."
            if model is not None
            else "Model weights not found. Please train the model using notebook/train_model.ipynb or place medical_model.keras into backend/models/"
        )
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Verify model is available
    if model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "ModelNotLoaded",
                "message": (
                    "The model file 'backend/models/medical_model.keras' has not been generated yet. "
                    "Please run the Jupyter training notebook 'notebook/train_model.ipynb' to train and save the model, "
                    "or verify that the model file exists."
                )
            }
        )

    # Validate file extension and MIME type
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    filename = file.filename or ""
    if not filename.lower().endswith(valid_extensions):
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Supported formats are: PNG, JPG, JPEG, BMP, TIFF."
        )

    # Read and parse image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image.verify()  # Validate image integrity
        image = Image.open(io.BytesIO(contents))  # Reopen after verify
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Corrupted or unreadable image file. Please upload a valid image."
        )

    try:
        # Preprocessing image for MobileNetV2
        rgb_image = image.convert("RGB")
        target_size = tuple(model_config.get("target_size", [224, 224]))
        resized_img = rgb_image.resize(target_size)
        
        img_array = np.array(resized_img, dtype=np.float32)
        # Expand dimensions to create batch: (1, 224, 224, 3)
        img_batch = np.expand_dims(img_array, axis=0)
        # Apply MobileNetV2 standard normalization (maps pixels to [-1, 1])
        preprocessed_batch = preprocess_input(img_batch.copy())

        # Model inference
        preds = model.predict(preprocessed_batch)
        prob_pneumonia = float(preds[0][0])
        prob_normal = 1.0 - prob_pneumonia

        # Determine predicted class and confidence
        class_names = model_config.get("class_names", ["NORMAL", "PNEUMONIA"])
        if prob_pneumonia >= 0.5:
            predicted_class = class_names[1]
            confidence = prob_pneumonia
        else:
            predicted_class = class_names[0]
            confidence = prob_normal

        # Attempt Grad-CAM visualization
        heatmap_base64: Optional[str] = None
        last_conv_name = model_config.get("last_conv_layer_name", "Conv_1")
        try:
            heatmap = make_gradcam_heatmap(preprocessed_batch, model, last_conv_name)
            if heatmap is not None:
                heatmap_base64 = overlay_gradcam(rgb_image, heatmap)
        except Exception as e:
            print(f"Skipping Grad-CAM: {e}")
            heatmap_base64 = None

        return JSONResponse(
            content={
                "prediction": predicted_class,
                "confidence": round(float(confidence), 4),
                "probabilities": {
                    "NORMAL": round(float(prob_normal), 4),
                    "PNEUMONIA": round(float(prob_pneumonia), 4)
                },
                "gradcam_base64": heatmap_base64,
                "disclaimer": "For educational and research purposes only. This tool is not intended for medical diagnosis."
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
