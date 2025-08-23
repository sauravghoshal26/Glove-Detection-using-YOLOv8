#!/usr/bin/env python3
"""
Download and prepare the Gloves Annotated Dataset from Roboflow
"""

import argparse
import os
from pathlib import Path
from roboflow import Roboflow

def main():
    parser = argparse.ArgumentParser(description="Download gloves dataset from Roboflow")
    parser.add_argument("--api-key", 
                        default=os.getenv('ROBOFLOW_API_KEY'),
                        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)")
    parser.add_argument("--output-dir", default="dataset", help="Output directory for dataset")
    parser.add_argument("--format", default="yolov8", help="Dataset format (yolov8, coco, etc.)")
    parser.add_argument("--version", type=int, default=1, help="Dataset version number")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: No API key provided. Use --api-key or set ROBOFLOW_API_KEY environment variable")
        return 1

    try:
        # Initialize Roboflow
        rf = Roboflow(api_key=args.api_key)
        
        # Try the original dataset first
        print("Attempting to download Gloves Annotated Dataset...")
        try:
            project = rf.workspace("sana-ali").project("gloves-annotated-dataset")
            dataset = project.version(args.version).download(args.format, location=args.output_dir)
            print(f"✅ Successfully downloaded Gloves Annotated Dataset")
        except Exception as e:
            print(f"❌ Failed to download original dataset: {e}")
            print("🔄 Trying alternative dataset: Gloves and bare hands detection...")
            
            # Try alternative dataset
            project = rf.workspace("moksha-me3nv").project("gloves-and-bare-hands-detection-pxk9g")
            dataset = project.version(1).download(args.format, location=args.output_dir)
            print(f"✅ Successfully downloaded alternative dataset")
        
        print(f"Dataset downloaded to: {args.output_dir}")
        print(f"Data.yaml location: {dataset.location}/data.yaml")
        
        # Print dataset info
        data_yaml_path = Path(dataset.location) / "data.yaml"
        if data_yaml_path.exists():
            print(f"\n📊 Dataset Information:")
            with open(data_yaml_path, 'r') as f:
                print(f.read())
        
        return 0
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print(f"\n💡 Alternative datasets you can try:")
        print(f"1. Safety Gloves dataset: roboflow-universe-projects/safety-gloves-xbnf8")
        print(f"2. Hand in glove detection: detr-cjz4w/hand-in-glove-detection") 
        print(f"3. Glove dataset: glove-uylxg/glove-q7czq")
        return 1

if __name__ == "__main__":
    exit(main())
