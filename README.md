# PaddleOCR-Wrapper

## Description
Wrapper for PaddleOCR

## Input
The wrapper takes either an ImageDocument or a VideoDocument with SWT TimeFrame annotations. The app specifically uses the representative TimePoint annotations from SWT v4 TimeFrame annotations to extract specific frames for OCR

## Output
The output will contain three objects: **Uri.SENTENCE** contains the smallest text unit recognized by the PaddleOCR; **AnnotationTypes.BoundingBox** contains the rectangular object in an image or video that captures the target text; **DocumentTypes.TextDocument** contains all the text recognized by PaddleOCR in an image with "\\" as the separation.

## User instruction
General user instructions for CLAMS apps are available at CLAMS Apps documentation: https://apps.clams.ai/clamsapp/

### System requirements
- Requires **mmif-python[cv]** for the VideoDocument helper functions
- Requires python package **paddlepaddle** and **paddleocr>=2.0.1**
- This version of the app is not available for GPU
