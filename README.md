# Glove Detection using YOLOv8

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A complete object detection pipeline to identify and differentiate between **gloved hands** and **bare hands** for workplace safety compliance, built with YOLOv8. This solution provides robust detection capabilities for industrial safety monitoring and PPE (Personal Protective Equipment) compliance checking.

## 🎯 Project Overview

This project addresses the critical need for automated safety compliance monitoring in industrial environments. By accurately detecting whether workers are wearing protective gloves, it helps ensure workplace safety standards and reduces the risk of hand injuries.

### Key Features

- **Dual-class Detection**: Distinguishes between `gloved_hand` and `bare_hand`
- **Real-world Performance**: Optimized for factory and industrial environments
- **Complete CLI Pipeline**: Easy-to-use command-line interface
- **Batch Processing**: Process multiple images simultaneously
- **Detailed Logging**: Per-image JSON logs with detection metadata
- **Annotated Outputs**: Visual results with bounding boxes and confidence scores

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- GPU support recommended (CUDA for NVIDIA, MPS for Apple Silicon)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sauravghoshal26/Glove-Detection-using-YOLOv8.git
   cd Glove-Detection-using-YOLOv8
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Roboflow API** (for dataset download)
   ```bash
   export ROBOFLOW_API_KEY=your_api_key_here
   ```

## 📊 Dataset

The project uses high-quality glove detection datasets from Roboflow Universe, specifically curated for industrial safety applications.

- **Classes**: `gloved_hand`, `bare_hand`
- **Format**: YOLO format with bounding box annotations
- **Splits**: Training, validation, and test sets
- **Quality**: Real-world industrial images with various lighting conditions

## 🔧 Usage

### 1. Download Dataset

```bash
python download_dataset.py --api-key YOUR_ROBOFLOW_API_KEY --output-dir dataset --version 1
```

Or using environment variable:
```bash
python download_dataset.py --output-dir dataset --version 1
```

### 2. Train the Model

**For Apple Silicon (MPS):**
```bash
python train_model.py \
  --data dataset/data.yaml \
  --epochs 30 \
  --batch 8 \
  --imgsz 640 \
  --device mps \
  --name glove_detection_mps
```

**For NVIDIA GPU:**
```bash
python train_model.py \
  --data dataset/data.yaml \
  --epochs 30 \
  --batch 16 \
  --imgsz 640 \
  --device 0 \
  --name glove_detection_gpu
```

**For CPU (development/testing):**
```bash
python train_model.py \
  --data dataset/data.yaml \
  --epochs 10 \
  --batch 4 \
  --imgsz 416 \
  --device cpu \
  --name glove_detection_cpu
```

### 3. Run Inference

```bash
python detection_script.py \
  --input dataset/test/images \
  --output output \
  --logs logs \
  --weights runs/detect/glove_detection_mps/weights/best.pt \
  --confidence 0.3 \
  --device mps \
  --normalize-classes
```

## 📁 Project Structure

```
Glove-Detection-using-YOLOv8/
├── detection_script.py      # Main inference script
├── train_model.py          # Training script
├── download_dataset.py     # Dataset download utility
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── dataset/               # Dataset directory
│   ├── train/
│   ├── val/
│   ├── test/
│   └── data.yaml
├── runs/                  # Training outputs
│   └── detect/
├── output/                # Annotated images
└── logs/                  # JSON detection logs
```

## 📈 Output Format

### Annotated Images
Visual outputs saved to the `output/` directory with:
- Bounding boxes around detected hands
- Class labels (`gloved_hand` or `bare_hand`)
- Confidence scores

### JSON Logs
Structured logs saved to `logs/` directory with the following schema:

```json
{
  "filename": "image1.jpg",
  "detections": [
    {
      "label": "gloved_hand",
      "confidence": 0.92,
      "bbox": [x1, y1, x2, y2]
    },
    {
      "label": "bare_hand",
      "confidence": 0.85,
      "bbox": [x1, y1, x2, y2]
    }
  ]
}
```

## ⚙️ Command Line Options

### Training Options
- `--data`: Path to dataset YAML file
- `--epochs`: Number of training epochs
- `--batch`: Batch size
- `--imgsz`: Input image size
- `--device`: Device to use (cpu, mps, cuda)
- `--name`: Experiment name

### Inference Options
- `--input/-i`: Input image folder
- `--output/-o`: Output folder for annotated images
- `--logs/-l`: Logs folder for JSON files
- `--weights/-w`: Path to trained model weights
- `--confidence/-c`: Confidence threshold (default: 0.25)
- `--imgsz`: Inference image size (default: 640)
- `--device`: Device to use
- `--normalize-classes`: Normalize class names to standard format

## 🔍 Model Performance

The model achieves robust performance on real-world industrial images:

- **Training Time**: 30-50 epochs recommended for optimal performance
- **Quick Validation**: 5-10 epochs for pipeline testing
- **mAP**: Varies based on dataset quality and training duration
- **Inference Speed**: Real-time capable on modern hardware

## 🛠️ Technical Challenges & Solutions

### Dataset Availability
- **Challenge**: Roboflow dataset versioning issues
- **Solution**: Fallback to alternative datasets and explicit version handling

### Performance Optimization
- **Challenge**: Limited GPU resources on various platforms
- **Solution**: Multi-platform support (CUDA, MPS, CPU) with optimized parameters

### Model Accuracy
- **Challenge**: Distinguishing between gloved and bare hands in various conditions
- **Solution**: Transfer learning from YOLOv8 pretrained weights with custom fine-tuning

## 🚧 Future Enhancements

- [ ] Real-time video stream processing
- [ ] Integration with safety alert systems
- [ ] Additional PPE detection (helmets, safety vests)
- [ ] Advanced data augmentation techniques


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

⭐ **Star this repository if you found it helpful!**
