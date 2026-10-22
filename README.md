# RF-DETR Nano: GPU-Optimized Object Detection

High-performance transformer-based object detection on NVIDIA Jetson Orin Nano Super, accelerated with TensorRT.

**19.74 FPS end-to-end at 1920 x 1080, with detailed GPU inference and pipeline profiling.**

This implementation measures the full inference pipeline, from video decoding and CUDA memory transfers to TensorRT execution, postprocessing and video encoding.

The benchmark identifies processing bottlenecks and shows how GPU inference latency differs from application throughput.

## Overview

This project evaluates transformer-based object detection using RF-DETR Nano and NVIDIA TensorRT.

It measures the complete video processing pipeline, including preprocessing, GPU transfers, inference, postprocessing, visualization and encoding.

## Performance

Measurements obtained on NVIDIA Jetson Orin Nano Super.

| Metric | Result |
|---|---:|
| Input resolution | 1920 x 1080 |
| Source frame rate | 25 FPS |
| Processed frames | 341 |
| Total processing time | 17.27 s |
| End-to-end throughput | 19.74 FPS |
| Average frame time | 50.65 ms |
| Preprocessing | 6.86 ms |
| GPU transfers and TensorRT | 7.85 ms |
| Postprocessing | 1.20 ms |
| Drawing and encoding | 28.42 ms |

These are measurements from a completed run, not universal performance guarantees.

## Architecture

1. Decode video frames.
2. Preprocess and normalize input.
3. Transfer tensors to GPU memory.
4. Execute RF-DETR Nano using TensorRT.
5. Decode predictions.
6. Render bounding boxes.
7. Encode the annotated output video.
8. Collect timing statistics.

## Engineering Highlights

- Transformer-based object detection with RF-DETR Nano.
- NVIDIA TensorRT execution and CUDA memory management.
- End-to-end throughput and per-stage latency measurements.
- Performance analysis at Full HD resolution.
- Identification of visualization and encoding overhead.

## Implementation

- `trt_frame_test.py`: single-frame TensorRT execution.
- `trt_video.py`: complete video inference and benchmarking.
- `Dockerfile`: original RF-DETR dependency environment.

The Dockerfile uses an NVIDIA PyTorch container and installs RF-DETR 1.11.0. Container compatibility with the target Jetson must be checked before use.

## Runtime Requirements

- NVIDIA Jetson Orin Nano Super 8 GB
- NVIDIA TensorRT
- CUDA Python bindings
- OpenCV
- NumPy
- Compatible RF-DETR Nano TensorRT engine

The benchmark scripts expect a locally generated engine at:

    output/trt/rfdetr-nano-fp16.engine

The model engine and test media are excluded from Git.

## Execution

Place the compatible TensorRT engine and input video in the locations configured by the Python scripts.

Run a single-frame test:

    python3 trt_frame_test.py

Run the video benchmark:

    python3 trt_video.py

The TensorRT engine must be generated using a compatible platform, TensorRT version and model configuration.

## Engineering Findings

TensorRT execution and memory transfers accounted for 7.85 ms per frame, while drawing and encoding accounted for 28.42 ms.

The main observed end-to-end bottleneck was outside neural network inference.

This illustrates why edge inference optimization must consider the complete application pipeline.

## Limitations

Measurements represent one video and one hardware configuration.

Performance depends on power mode, input resolution, engine precision, model configuration, preprocessing and encoding.

The repository does not include a complete model conversion pipeline.
