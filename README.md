# Image Text Extractor using Donut Library

This repository contains a scratch implementation for extracting text from images using the Donut library. The provided code allows you to convert an input image into readable text by leveraging the power of transformer-based deep learning models.

## Overview

The code in this repository uses the `DonutProcessor` and `VisionEncoderDecoderModel` from the transformers library, which is fine-tuned on CORD-V2 dataset. It also utilizes PIL (Python Imaging Library) to load and manipulate images, processing them as input for our deep learning model.

### Required Libraries

Make sure to install these libraries before running the script:
```
pip install torch
pip install transformers
pip install Pillow
```
