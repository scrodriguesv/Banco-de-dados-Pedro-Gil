"""
bitmapGenerator.py
------------------
Utility for converting COCO-format annotation files into semantic segmentation
bitmaps, where each pixel value corresponds to a category ID.
"""

import argparse
import json
import os
import sys
import numpy as np
from PIL import Image, ImageDraw
from pycocotools.coco import COCO


class bitmapGenerator:
    """Generates semantic segmentation bitmaps from COCO annotation files.

    Each output bitmap is a grayscale PNG where every pixel encodes the
    category ID of the annotated region (0 = background).
    """

    def __init__(self, annotation_file: os.PathLike, bitmap_folder: os.PathLike):
        """Initialise the generator.

        Args:
            annotation_file: Path to the COCO-format JSON annotation file.
            bitmap_folder:   Directory where the output bitmap PNGs will be saved.
        """
        self.annotation_file = annotation_file
        self.bitmap_folder = bitmap_folder

    def coco_to_semantic_bitmaps(self):
        """Convert all annotated images in the COCO file to semantic bitmaps.

        For every image listed in the annotation file a grayscale PNG is
        created alongside each annotation polygon drawn with the corresponding
        category ID as the fill value.  Overlapping annotations are resolved
        with ``np.maximum``, so higher category IDs take precedence.

        Output files are written to ``self.bitmap_folder`` and follow the
        naming convention ``<original_stem>_semantic.png``.
        """
        # Load COCO annotations
        with open(self.annotation_file, 'r') as f:
            coco_data = json.load(f)

        coco = COCO(self.annotation_file)

        # Create a dictionary to map image IDs to file names
        image_id_to_file = {img["id"]: img["file_name"] for img in coco_data["images"]}

        # Create bitmaps for each image
        for image_id, file_name in image_id_to_file.items():
            # Load the image dimensions
            image_info = coco.loadImgs(image_id)[0]
            width, height = image_info["width"], image_info["height"]

            # Initialize a blank bitmap with zeros (background class)
            semantic_bitmap = np.zeros((height, width), dtype=np.uint8)

            # Get all annotations for this image
            annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id))

            # Process each annotation
            for ann in annotations:
                category_id = ann["category_id"]
                segmentation = ann["segmentation"]

                # Create a blank mask for the current annotation
                mask = Image.new("L", (width, height), 0)  # "L" mode for grayscale

                # Draw the polygon(s) onto the mask
                draw = ImageDraw.Draw(mask)
                for poly in segmentation:
                    draw.polygon(poly, fill=category_id)

                # Combine the annotation mask into the semantic bitmap
                semantic_bitmap = np.maximum(semantic_bitmap, np.array(mask, dtype=np.uint8))

            # Save the semantic bitmap as an image
            output_path = f"{self.bitmap_folder}/{file_name.replace('.jpg', '_semantic.png')}"
            semantic_image = Image.fromarray(semantic_bitmap)
            semantic_image.save(output_path)
            print(f"Saved semantic bitmap: {output_path}")


def main(argv=None):
    """CLI entry-point for the bitmap generator.

    Examples
    --------
    # Run as a module
    python -m source.bitmapGenerator -a annotations.json -o bitmaps/

    # Run as an installed console script
    generate-bitmaps -a annotations.json -o bitmaps/
    """
    parser = argparse.ArgumentParser(
        prog="generate-bitmaps",
        description=(
            "Convert a COCO-format annotation file into semantic segmentation "
            "bitmaps (grayscale PNGs where each pixel value equals the category ID)."
        ),
    )
    parser.add_argument(
        "-a", "--annotation-file",
        required=True,
        metavar="FILE",
        help="Path to the COCO-format JSON annotation file.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        required=True,
        metavar="DIR",
        help="Directory where the output bitmap PNGs will be saved. Created if it does not exist.",
    )

    args = parser.parse_args(argv)

    annotation_file = args.annotation_file
    output_dir = args.output_dir

    if not os.path.isfile(annotation_file):
        parser.error(f"Annotation file not found: {annotation_file}")

    os.makedirs(output_dir, exist_ok=True)

    generator = bitmapGenerator(
        annotation_file=annotation_file,
        bitmap_folder=output_dir,
    )
    generator.coco_to_semantic_bitmaps()


if __name__ == "__main__":
    main()
