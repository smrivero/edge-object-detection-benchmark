FROM nvcr.io/nvidia/pytorch:25.08-py3

WORKDIR /workspace

RUN python3 -m pip install --no-cache-dir \
    "numpy==1.26.4" \
    "opencv-python-headless==4.10.0.84" \
    "rfdetr==1.11.0"

RUN python3 -m pip install --no-cache-dir --no-deps \
    "onnxconverter-common==1.16.0"

CMD ["/bin/bash"]
