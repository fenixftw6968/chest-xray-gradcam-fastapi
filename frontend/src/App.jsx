import React, { useState, useEffect, useRef } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [serverHealth, setServerHealth] = useState(null);
  const [isDragActive, setIsDragActive] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/health`);
      if (res.ok) {
        const data = await res.json();
        setServerHealth(data);
      } else {
        setServerHealth({ status: 'unreachable' });
      }
    } catch {
      setServerHealth({ status: 'unreachable' });
    }
  };

  const handleFileChange = (file) => {
    if (!file) return;

    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/tiff'];
    if (!validTypes.includes(file.type)) {
      setError('Please upload a valid image (PNG or JPG).');
      return;
    }

    setSelectedFile(file);
    setError(null);
    setResult(null);

    const reader = new FileReader();
    reader.onload = () => {
      setPreviewUrl(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const onDragOver = (e) => {
    e.preventDefault();
    setIsDragActive(true);
  };

  const onDragLeave = () => {
    setIsDragActive(false);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setIsDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        const msg = data?.detail?.message || data?.detail || 'Analysis request failed.';
        throw new Error(msg);
      }

      setResult(data);
    } catch (err) {
      setError(err.message || 'Unable to connect to backend.');
    } finally {
      setLoading(false);
      checkHealth();
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const statusClass =
    serverHealth?.status === 'healthy'
      ? 'healthy'
      : serverHealth?.status === 'degraded'
      ? 'degraded'
      : 'unreachable';

  const statusLabel =
    serverHealth?.status === 'healthy'
      ? 'Model Ready'
      : serverHealth?.status === 'degraded'
      ? 'Weights Missing'
      : 'API Offline';

  return (
    <div className="container">
      {/* Header */}
      <header className="header">
        <div className="brand-row">
          <h1 className="brand-title">AI Medical Image Analyzer</h1>
          <button
            className="status-badge"
            onClick={checkHealth}
            title="Click to recheck API connection"
            type="button"
          >
            <span className={`status-dot ${statusClass}`}></span>
            {statusLabel}
          </button>
        </div>
        <p className="subtitle">
          Screening prototype using MobileNetV2 transfer learning.
        </p>
      </header>

      {/* Main Analysis Card */}
      <main className="card">
        {!previewUrl ? (
          <div
            className={`dropzone ${isDragActive ? 'drag-active' : ''}`}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
            onClick={() => fileInputRef.current && fileInputRef.current.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/png, image/jpeg, image/jpg, image/tiff"
              style={{ display: 'none' }}
              onChange={(e) => handleFileChange(e.target.files[0])}
            />
            <svg
              className="icon"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
            >
              <rect width="18" height="18" x="3" y="3" rx="2" ry="2" />
              <circle cx="9" cy="9" r="2" />
              <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
            </svg>
            <span className="drop-prompt">Choose an X-ray or drag & drop here</span>
            <span className="drop-hint">PNG, JPEG up to 10MB</span>
          </div>
        ) : (
          <div className="preview-layout">
            <div className={`images-row ${result?.gradcam_base64 ? 'has-gradcam' : ''}`}>
              <div className="image-panel">
                <span className="image-tag">Original X-Ray</span>
                <img src={previewUrl} alt="Chest X-Ray" />
              </div>
              {result?.gradcam_base64 && (
                <div className="image-panel">
                  <span className="image-tag">Grad-CAM Activation</span>
                  <img
                    src={`data:image/png;base64,${result.gradcam_base64}`}
                    alt="Grad-CAM heatmap"
                  />
                </div>
              )}
            </div>

            <div className="actions-bar">
              <span className="file-meta">{selectedFile?.name}</span>
              <div className="btn-row">
                <button
                  className="btn btn-ghost"
                  onClick={handleReset}
                  disabled={loading}
                  type="button"
                >
                  Change Image
                </button>
                <button
                  className="btn btn-primary"
                  onClick={handleAnalyze}
                  disabled={loading || !selectedFile}
                  type="button"
                >
                  {loading ? (
                    <>
                      <span className="spinner"></span> Analyzing...
                    </>
                  ) : (
                    'Analyze'
                  )}
                </button>
              </div>
            </div>
          </div>
        )}

        {error && <div className="error-msg">{error}</div>}

        {result && !loading && (
          <div className="result-box">
            <div className="result-main">
              <div className="result-title-group">
                <span className="result-kicker">Prediction</span>
                <span className={`result-finding ${result.prediction}`}>
                  {result.prediction}
                </span>
              </div>
              <div className="result-score">
                <div className="score-num">{(result.confidence * 100).toFixed(1)}%</div>
                <div className="score-label">Confidence</div>
              </div>
            </div>

            {result.probabilities && (
              <div className="bars-container">
                <div className="bar-row">
                  <span className="bar-label">NORMAL</span>
                  <div className="bar-track">
                    <div
                      className="bar-fill NORMAL"
                      style={{ width: `${result.probabilities.NORMAL * 100}%` }}
                    ></div>
                  </div>
                  <span className="bar-val">
                    {(result.probabilities.NORMAL * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="bar-row">
                  <span className="bar-label">PNEUMONIA</span>
                  <div className="bar-track">
                    <div
                      className="bar-fill PNEUMONIA"
                      style={{ width: `${result.probabilities.PNEUMONIA * 100}%` }}
                    ></div>
                  </div>
                  <span className="bar-val">
                    {(result.probabilities.PNEUMONIA * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="disclaimer">
        Educational prototype only. Not intended for clinical diagnosis.
      </footer>
    </div>
  );
}
