#!/usr/bin/env python3
"""
Train YOLOv8 model for gloved vs bare hand detection
"""

import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO

def update_data_yaml(data_path: str, output_path: str = None):
    """Update data.yaml to use standard class names"""
    if output_path is None:
        output_path = data_path
    
    with open(data_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Map original class names to our standard names
    class_mapping = {
        'gloved': 'gloved_hand',
        'not-gloved': 'bare_hand',
        'glove': 'gloved_hand',
        'no_glove': 'bare_hand',
        'bare': 'bare_hand'
    }
    
    # Update class names if they exist
    if 'names' in data:
        if isinstance(data['names'], list):
            data['names'] = [class_mapping.get(name.lower(), name) for name in data['names']]
        elif isinstance(data['names'], dict):
            new_names = {}
            for key, value in data['names'].items():
                new_names[key] = class_mapping.get(value.lower(), value)
            data['names'] = new_names
    
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for gloved vs bare hand detection")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8s.pt", help="Base model (yolov8n.pt, yolov8s.pt, yolov8m.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--device", default="0", help="Device (cpu, 0, 1, etc.)")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--save-period", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--project", default="runs/detect", help="Project directory")
    parser.add_argument("--name", default="glove_detection", help="Experiment name")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use pretrained weights")
    
    # Data augmentation parameters
    parser.add_argument("--hsv-h", type=float, default=0.015, help="HSV hue augmentation")
    parser.add_argument("--hsv-s", type=float, default=0.7, help="HSV saturation augmentation")
    parser.add_argument("--hsv-v", type=float, default=0.4, help="HSV value augmentation")
    parser.add_argument("--degrees", type=float, default=0.0, help="Rotation degrees")
    parser.add_argument("--translate", type=float, default=0.1, help="Translation fraction")
    parser.add_argument("--scale", type=float, default=0.5, help="Scaling factor")
    parser.add_argument("--shear", type=float, default=0.0, help="Shear degrees")
    parser.add_argument("--perspective", type=float, default=0.0, help="Perspective transformation")
    parser.add_argument("--flipud", type=float, default=0.0, help="Vertical flip probability")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Horizontal flip probability")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation probability")
    parser.add_argument("--mixup", type=float, default=0.0, help="MixUp augmentation probability")
    
    args = parser.parse_args()
    
    # Update data.yaml with standard class names
    data_path = update_data_yaml(args.data)
    
    # Load model
    model = YOLO(args.model)
    
    # Training parameters
    train_args = {
        'data': data_path,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': args.device,
        'workers': args.workers,
        'patience': args.patience,
        'save_period': args.save_period,
        'project': args.project,
        'name': args.name,
        'resume': args.resume,
        'optimizer': 'AdamW',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'nbs': 64,
        # Augmentations
        'hsv_h': args.hsv_h,
        'hsv_s': args.hsv_s,
        'hsv_v': args.hsv_v,
        'degrees': args.degrees,
        'translate': args.translate,
        'scale': args.scale,
        'shear': args.shear,
        'perspective': args.perspective,
        'flipud': args.flipud,
        'fliplr': args.fliplr,
        'mosaic': args.mosaic,
        'mixup': args.mixup,
        'copy_paste': 0.0,
    }
    
    print("Starting training with parameters:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    
    # Start training
    results = model.train(**train_args)
    
    print("Training completed!")
    print(f"Best weights saved at: {model.trainer.best}")
    print(f"Last weights saved at: {model.trainer.last}")
    
    # Validate the model
    print("Running validation...")
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP75: {metrics.box.map75}")

if __name__ == "__main__":
    main()
