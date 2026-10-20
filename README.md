# AI-Powered Medical Image Analyzer

> **Educational & Research Prototype**: This system is designed for computer vision study and educational demonstration. It is **not** a certified medical diagnostic device and should **never** be used for clinical diagnoses or patient management.

An end-to-end Deep Learning and Cloud-ready medical image analysis application featuring **MobileNetV2 transfer learning**, a **FastAPI backend with Grad-CAM explainability**, and a modern **React + Vite dashboard**.

---

## 1. Architecture Overview

```text
medical-image-analyzer/
├── notebook/
│   └── train_model.ipynb         # End-to-end MobileNetV2 training, evaluation & model export
├── backend/
│   ├── main.py                   # FastAPI server with health check, prediction & Grad-CAM
│   ├── requirements.txt          # Python dependencies
│   └── models/
│       ├── medical_model.keras   # Saved Keras model (MobileNetV2 transfer learning)
│       ├── model_config.json     # Inference parameters & class mapping
│       └── .gitkeep
├── frontend/
│   ├── src/
│   │   ├── App.jsx               # React dashboard with X-ray dropzone & results
│   │   ├── index.css             # Responsive theme
│   │   └── main.jsx
│   ├── .env                      # Local environment: VITE_API_URL=http://localhost:8000
│   ├── .env.example              # Template for local & production configurations
│   ├── package.json              # Frontend dependencies and build scripts
│   └── vite.config.js
├── .gitignore
└── README.md
```

---

## 2. Dataset Setup

The dataset follows the standard Kaggle Chest X-Ray format:
```text
dataset/chest_xray/
├── train/
│   ├── NORMAL/      (1,341 images)
│   └── PNEUMONIA/   (3,875 images)
├── val/
│   ├── NORMAL/      (8 images)
│   └── PNEUMONIA/   (8 images)
└── test/
    ├── NORMAL/      (234 images)
    └── PNEUMONIA/   (390 images)
```

The notebook dynamically detects this path (`../dataset/chest_xray`), resizes images to `224 × 224`, normalizes pixel values to `[-1, 1]` via `preprocess_input`, and handles class imbalance using calculated class weights.

---

## 3. Step-by-Step Local Setup & Execution (Windows)

Open **PowerShell** or Command Prompt and follow these steps:

### Step A: Setup Python Virtual Environment & Backend
```powershell
# 1. Navigate to the backend directory
cd medical-image-analyzer/backend

# 2. Create a virtual environment (if not already created)
python -m venv venv

# 3. Activate the virtual environment
.\venv\Scripts\Activate.ps1

# 4. Install backend dependencies
pip install -r requirements.txt
```

### Step B: Run the Jupyter Training Notebook (Optional if model exists)
The repository includes a ready-to-run notebook and pre-configured model weights. To retrain or inspect the workflow:
```powershell
# 1. Install Jupyter in the active environment (if not present)
pip install jupyter notebook

# 2. Launch Jupyter Notebook
cd ../notebook
jupyter notebook train_model.ipynb
```
Run all cells from top to bottom. It will:
1. Load dataset with batch size `32` and apply data augmentation.
2. Train the binary classification head for 5 epochs with base MobileNetV2 frozen.
3. Unfreeze layers from layer 120 onward and fine-tune for 5 epochs with learning rate `1e-5`.
4. Plot accuracy/loss and generate a confusion matrix & classification report on the `test` set.
5. Save the trained weights to `backend/models/medical_model.keras` and configuration to `backend/models/model_config.json`.

### Step C: Start the FastAPI Backend
```powershell
cd medical-image-analyzer/backend
# Ensure your virtual environment is active
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
- Interactive Swagger API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health check endpoint: [http://localhost:8000/health](http://localhost:8000/health)

### Step D: Start the React + Vite Frontend
Open a **second terminal window**:
```powershell
cd medical-image-analyzer/frontend

# Install node dependencies
npm install

# Start Vite dev server
npm run dev
```
Open your browser at: **`http://localhost:5173`** (or the URL printed by Vite).

---

## 4. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Root endpoint displaying API metadata and legal disclaimer |
| `GET` | `/health` | Health status and model readiness check |
| `POST` | `/predict` | Multipart form upload accepting an image (`file`), returning class predictions, confidence, probability breakdown, and base64 Grad-CAM activation map |

### Sample Response (`POST /predict`):
```json
{
  "prediction": "PNEUMONIA",
  "confidence": 0.8842,
  "probabilities": {
    "NORMAL": 0.1158,
    "PNEUMONIA": 0.8842
  },
  "gradcam_base64": "<base64_encoded_png_overlay>",
  "disclaimer": "For educational and research purposes only. This tool is not intended for medical diagnosis."
}
```

---

## 5. Amazon AWS Deployment Guide

### Option 1: Backend Deployment on Amazon EC2 (Ubuntu 22.04 LTS)
1. **Launch EC2 Instance**:
   - Instance type: `t3.medium` or `t3.large` (minimum 4 GB RAM recommended for TensorFlow CPU inference).
   - In Security Group, open inbound ports:
     - `SSH (22)`: Your IP
     - `Custom TCP (8000)` or `HTTP (80)`: `0.0.0.0/0`
2. **Connect & Setup Environment**:
   ```bash
   sudo apt update && sudo apt install -y python3-pip python3-venv git
   git clone <YOUR_REPOSITORY_URL>
   cd medical-image-analyzer/backend
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Run Production Server with Gunicorn/Uvicorn or systemd**:
   ```bash
   pip install gunicorn
   gunicorn -w 2 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000 --daemon
   ```

### Option 2: Frontend Deployment on AWS S3 + CloudFront or AWS Amplify
1. **Configure Production API URL**:
   In `frontend/.env`:
   ```bash
   VITE_API_URL=http://<YOUR_EC2_PUBLIC_IP_OR_DOMAIN>:8000
   ```
2. **Build Production Bundle**:
   ```bash
   npm run build
   # Outputs optimized static files to frontend/dist/
   ```
3. **Deploy via AWS S3 / CloudFront**:
   - Create an S3 bucket with **Static Website Hosting** enabled.
   - Upload the contents of `frontend/dist/` to the root of the bucket.
   - Point an **AWS CloudFront Distribution** to the S3 bucket origin for free SSL (HTTPS) and CDN caching.
   - Alternatively, connect your GitHub repository directly to **AWS Amplify Console** for continuous deployments with zero server management.

### Security & Cost Optimization Tips
- **AWS Budget Alert**: Set a $5-$10 AWS billing budget alert in AWS Cost Management.
- **Stop EC2 Instances**: Stop the EC2 instance when not in use to avoid hourly charges.
- **Never Commit Secrets**: Do not place AWS access keys, secret keys, or `.env` files with credentials in Git. Use IAM Roles for EC2 rather than hardcoded credentials.

---

## 6. Educational Disclaimer
This project is developed solely for educational and portfolio demonstration. Artificial intelligence models in medicine can produce false positives and false negatives. This software must not be used as a substitute for professional medical advice, diagnosis, or treatment.
