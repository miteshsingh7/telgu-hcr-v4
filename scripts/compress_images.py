#!/usr/bin/env python3
"""Utility script to compress and downscale JPEG/PNG images.

Reduces large images to lightweight, API- and model-friendly sizes (< 1-2 MB, max ~1200px)
to prevent network timeout errors and memory issues when uploading images to models or apps.

Usage:
    python scripts/compress_images.py <input_path_or_dir> [--output_dir <path>] [--max_size 1200] [--quality 85]
"""

import sys
import argparse
from pathlib import Path
from PIL import Image, ImageOps


def compress_image(image_path: Path, output_path: Path, max_dim: int = 1200, quality: int = 85) -> bool:
    """Resizes and compresses a single image file."""
    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            
            if img.mode in ("RGBA", "P"):
                bg = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "RGBA":
                    bg.paste(img, mask=img.split()[3])
                else:
                    bg.paste(img)
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")
            
            w, h = img.size
            if max(w, h) > max_dim:
                if w > h:
                    new_w = max_dim
                    new_h = int(h * (max_dim / w))
                else:
                    new_h = max_dim
                    new_w = int(w * (max_dim / h))
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path, format="JPEG", quality=quality, optimize=True)
            
            orig_size = image_path.stat().st_size / 1024
            new_size = output_path.stat().st_size / 1024
            print(f"✓ Optimized: {image_path.name} ({orig_size:.1f} KB -> {new_size:.1f} KB, {img.size[0]}x{img.size[1]})")
            return True
            
    except Exception as e:
        print(f"✗ Failed to process {image_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Compress and resize images to prevent timeout errors.")
    parser.add_argument("input", type=str, help="Path to an image file or directory containing images.")
    parser.add_argument("--output_dir", "-o", type=str, default=None, help="Directory to save compressed images (default: appends '_optimized' or writes in-place).")
    parser.add_argument("--max_size", "-m", type=int, default=1200, help="Max width/height in pixels (default: 1200).")
    parser.add_argument("--quality", "-q", type=int, default=85, help="JPEG quality 1-100 (default: 85).")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Path '{input_path}' does not exist.")
        sys.exit(1)
        
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    
    if input_path.is_file():
        if input_path.suffix.lower() not in valid_exts:
            print(f"Error: '{input_path.name}' is not a supported image format.")
            sys.exit(1)
            
        if args.output_dir:
            out_dir = Path(args.output_dir)
            out_file = out_dir / f"{input_path.stem}.jpg"
        else:
            out_file = input_path.parent / f"{input_path.stem}_optimized.jpg"
            
        compress_image(input_path, out_file, max_dim=args.max_size, quality=args.quality)
        print(f"\nSaved compressed file to: {out_file}")
        
    elif input_path.is_dir():
        image_files = [p for p in input_path.rglob("*") if p.suffix.lower() in valid_exts]
        if not image_files:
            print(f"No image files found in '{input_path}'.")
            sys.exit(0)
            
        out_dir = Path(args.output_dir) if args.output_dir else (input_path.parent / f"{input_path.name}_optimized")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing {len(image_files)} image(s) from '{input_path}' -> '{out_dir}'...\n")
        success_count = 0
        for img_p in image_files:
            rel_p = img_p.relative_to(input_path)
            out_file = out_dir / rel_p.with_suffix(".jpg")
            if compress_image(img_p, out_file, max_dim=args.max_size, quality=args.quality):
                success_count += 1
                
        print(f"\nFinished! Successfully compressed {success_count}/{len(image_files)} images.")
        print(f"Output directory: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
