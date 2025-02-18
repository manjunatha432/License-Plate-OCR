# """
# The OCR model (PaddleOCR).
# Preprocessing functions for the images before they are passed to the OCR model.
# """
# import warnings
# warnings.filterwarnings("ignore", message="No ccache found.*")  # Ignore ccache warning

# from paddleocr import PaddleOCR

# def ocr_model(image):
#     ocr_model = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

#     results = ocr_model.ocr(image)

#     extracted_values = [item[1][0] for item in results[0]]
#     combined_string = " ".join(extracted_values)
#     return combined_string

"""
License Plate OCR Module

This module provides OCR (Optical Character Recognition) functionality using PaddleOCR,
specifically optimized for license plate text extraction.
"""

import warnings
from typing import List, Optional
from paddleocr import PaddleOCR

# Suppress specific PaddleOCR warnings
warnings.filterwarnings("ignore", message="No ccache found.*")

class PlateOCR:
    def __init__(self, language: str = 'en', use_angle_cls: bool = True) -> None:
        """
        Initialize the OCR model with specified parameters.

        Args:
            language (str): Language model to use for OCR. Defaults to 'en'.
            use_angle_cls (bool): Whether to use angle classification. Defaults to True.
        """
        self.model = PaddleOCR(use_angle_cls=use_angle_cls,lang=language,show_log=False)

    def extract_text(self, image) -> Optional[str]:
        """
        Extract text from the provided image.

        Args:
            image: Input image (numpy array or path to image file)

        Returns:
            str: Extracted text as a single string, or None if no text was found
        """
        try:
            results = self.model.ocr(image)
            
            if not results or not results[0]:
                return None
            
            # Extract text values and combine them
            extracted_values = [item[1][0] for item in results[0]]
            return " ".join(extracted_values)
            
        except Exception as e:
            print(f"Error during OCR processing: {str(e)}")
            return None