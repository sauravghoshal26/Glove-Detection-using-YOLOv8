#!/usr/bin/env python3
"""
Gloved vs Bare Hand Detection - Production Ready
Processes folder of images, saves annotated outputs, and logs per-image JSON detections
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import cv2
import numpy as np
from ultralytics import YOLO

# Supported image extensions
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

def get_image_files(input_dir: Path) -> List[Path]:
    """Get all valid image files from input directory"""
    image_files = []
    for ext in VALID_EXTENSIONS:
        image_files.extend(input_dir.glob(f'*{ext}'))
        image_files.extend(input_dir.glob(f'*{ext.upper()}'))
    return sorted(image_files)

def ensure_directory(path: Path) -> None:
    """Create directory if it doesn't exist"""
    path.mkdir(parents=True, exist_ok=True)

def normalize_class_name(class_name: str, use_mapping: bool = True) -> str:
    """Normalize class names to standard format"""
    if not use_mapping:
        return class_name
    
    # Mapping from various possible class names to standard names
    class_mapping = {
        'gloved': 'gloved_hand',
        'glove': 'gloved_hand', 
        'surgical-gloves': 'gloved_hand',
        'gloves': 'gloved_hand',
        'with_glove': 'gloved_hand',
        'not-gloved': 'bare_hand',
        'no_glove': 'bare_hand',
        'bare': 'bare_hand',
        'hand': 'bare_hand',
        'ungloved': 'bare_hand',
        'without_glove': 'bare_hand',
        'bare_hand': 'bare_hand',
        'gloved_hand': 'gloved_hand'
    }
    
    normalized = class_name.lower().strip()
    return class_mapping.get(normalized, class_name)

def process_detections(results, confidence_threshold: float, normalize_classes: bool) -> List[Dict[str, Any]]:
    """Process YOLO detection results into required format"""
    detections = []
    
    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        class_names = results.names
        
        for i in range(len(boxes)):
            if confidences[i] >= confidence_threshold:
                x1, y1, x2, y2 = boxes[i]
                class_name = class_names[class_ids[i]]
                
                if normalize_classes:
                    class_name = normalize_class_name(class_name)
                
                detection = {
                    "label": class_name,
                    "confidence": round(float(confidences[i]), 4),
                    "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
                }
                detections.append(detection)
    
    return detections

def save_annotated_image(results, output_path: Path, quality: int = 95) -> None:
    """Save annotated image with detections"""
    annotated_img = results.plot(
        conf=True,
        labels=True,
        boxes=True,
        line_width=2
    )
    
    # Ensure high quality output
    if output_path.suffix.lower() in ['.jpg', '.jpeg']:
        cv2.imwrite(str(output_path), annotated_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif output_path.suffix.lower() == '.png':
        cv2.imwrite(str(output_path), annotated_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    else:
        cv2.imwrite(str(output_path), annotated_img)

def save_detection_log(filename: str, detections: List[Dict[str, Any]], log_path: Path) -> None:
    """Save detection results to JSON file"""
    log_data = {
        "filename": filename,
        "detections": detections
    }
    
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(
        description="Gloved vs Bare Hand Detection using YOLOv8",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--input", "-i", required=True, type=str,
                        help="Input directory containing images")
    parser.add_argument("--output", "-o", required=True, type=str,
                        help="Output directory for annotated images")
    parser.add_argument("--logs", "-l", required=True, type=str,
                        help="Directory for JSON detection logs")
    parser.add_argument("--weights", "-w", required=True, type=str,
                        help="Path to trained YOLOv8 model weights")
    
    # Optional arguments
    parser.add_argument("--confidence", "-c", type=float, default=0.25,
                        help="Confidence threshold for detections")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size for inference")
    parser.add_argument("--device", type=str, default="0",
                        help="Device to run inference on (cpu, 0, 1, etc.)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for inference")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--half", action="store_true",
                        help="Use FP16 half-precision inference")
    parser.add_argument("--normalize-classes", action="store_true", default=True,
                        help="Normalize class names to gloved_hand/bare_hand")
    parser.add_argument("--save-txt", action="store_true",
                        help="Save results to *.txt files")
    parser.add_argument("--save-conf", action="store_true", default=True,
                        help="Save confidences in labels")
    parser.add_argument("--max-det", type=int, default=1000,
                        help="Maximum detections per image")
    parser.add_argument("--agnostic-nms", action="store_true",
                        help="Class-agnostic NMS")
    parser.add_argument("--augment", action="store_true",
                        help="Augmented inference")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize features")
    parser.add_argument("--update", action="store_true",
                        help="Update all models")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    
    args = parser.parse_args()
    
    # Validate paths
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    logs_dir = Path(args.logs)
    weights_path = Path(args.weights)
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    if not weights_path.exists():
        print(f"Error: Model weights file does not exist: {weights_path}")
        sys.exit(1)
    
    # Create output directories
    ensure_directory(output_dir)
    ensure_directory(logs_dir)
    
    # Get image files
    image_files = get_image_files(input_dir)
    if not image_files:
        print(f"Warning: No valid images found in {input_dir}")
        print(f"Supported extensions: {sorted(VALID_EXTENSIONS)}")
        sys.exit(0)
    
    print(f"Found {len(image_files)} images to process")
    
    # Load model
    try:
        model = YOLO(str(weights_path))
        if args.verbose:
            print(f"Loaded model: {weights_path}")
            print(f"Model classes: {model.names}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Run inference
    start_time = time.time()
    
    try:
        results = model.predict(
            source=[str(img) for img in image_files],
            conf=args.confidence,
            imgsz=args.imgsz,
            device=args.device if args.device != 'cpu' else 'cpu',
            half=args.half,
            max_det=args.max_det,
            agnostic_nms=args.agnostic_nms,
            augment=args.augment,
            visualize=args.visualize,
            save=False,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            stream=False,
            verbose=args.verbose
        )
        
        # Process results
        processed_count = 0
        total_detections = 0
        
        for i, (result, image_file) in enumerate(zip(results, image_files)):
            filename = image_file.name
            
            # Process detections
            detections = process_detections(result, args.confidence, args.normalize_classes)
            total_detections += len(detections)
            
            # Save annotated image
            output_image_path = output_dir / filename
            save_annotated_image(result, output_image_path)
            
            # Save detection log
            log_file_path = logs_dir / f"{image_file.stem}.json"
            save_detection_log(filename, detections, log_file_path)
            
            processed_count += 1
            
            if args.verbose and (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Summary
        print(f"\n{'='*50}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*50}")
        print(f"Images processed: {processed_count}")
        print(f"Total detections: {total_detections}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Average time per image: {processing_time/processed_count:.3f} seconds")
        print(f"Annotated images saved to: {output_dir}")
        print(f"Detection logs saved to: {logs_dir}")
        print(f"Confidence threshold: {args.confidence}")
        print(f"Image size: {args.imgsz}")
        print(f"Device: {args.device}")
        
        if args.normalize_classes:
            print("Class names normalized to: gloved_hand, bare_hand")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
