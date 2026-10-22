import cv2
import numpy as np
import tensorrt as trt

from cuda.bindings import runtime as cudart


ENGINE = "output/trt/rfdetr-nano-fp16.engine"
VIDEO = "data/people-walking.mp4"
OUTPUT = "output/rfdetr-nano-trt-frame.jpg"

THRESHOLD = 0.30
INPUT_H = 384
INPUT_W = 384
NUM_SELECT = 300


COCO_CLASSES = {
    1:"person", 2:"bicycle", 3:"car", 4:"motorcycle", 5:"airplane",
    6:"bus", 7:"train", 8:"truck", 9:"boat", 10:"traffic light",
    11:"fire hydrant", 13:"stop sign", 14:"parking meter", 15:"bench",
    16:"bird", 17:"cat", 18:"dog", 19:"horse", 20:"sheep", 21:"cow",
    22:"elephant", 23:"bear", 24:"zebra", 25:"giraffe", 27:"backpack",
    28:"umbrella", 31:"handbag", 32:"tie", 33:"suitcase", 34:"frisbee",
    35:"skis", 36:"snowboard", 37:"sports ball", 38:"kite",
    39:"baseball bat", 40:"baseball glove", 41:"skateboard",
    42:"surfboard", 43:"tennis racket", 44:"bottle", 46:"wine glass",
    47:"cup", 48:"fork", 49:"knife", 50:"spoon", 51:"bowl",
    52:"banana", 53:"apple", 54:"sandwich", 55:"orange", 56:"broccoli",
    57:"carrot", 58:"hot dog", 59:"pizza", 60:"donut", 61:"cake",
    62:"chair", 63:"couch", 64:"potted plant", 65:"bed",
    67:"dining table", 70:"toilet", 72:"tv", 73:"laptop", 74:"mouse",
    75:"remote", 76:"keyboard", 77:"cell phone", 78:"microwave",
    79:"oven", 80:"toaster", 81:"sink", 82:"refrigerator", 84:"book",
    85:"clock", 86:"vase", 87:"scissors", 88:"teddy bear",
    89:"hair drier", 90:"toothbrush"
}


def cuda_check(err):
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def preprocess(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(
        rgb,
        (INPUT_W, INPUT_H),
        interpolation=cv2.INTER_LINEAR
    )

    x = resized.astype(np.float32) / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    ).reshape(1, 1, 3)

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    ).reshape(1, 1, 3)

    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, 0)

    return np.ascontiguousarray(x, dtype=np.float32)


def decode(boxes, logits, image_w, image_h):
    boxes = boxes[0]
    logits = logits[0]

    scores = sigmoid(logits)

    flat = scores.reshape(-1)

    k = min(NUM_SELECT, flat.size)

    idx = np.argpartition(flat, -k)[-k:]
    idx = idx[np.argsort(flat[idx])[::-1]]

    query_ids = idx // scores.shape[1]
    class_ids = idx % scores.shape[1]
    confidences = flat[idx]

    keep = (
        (confidences >= THRESHOLD) &
        np.array([int(c) in COCO_CLASSES for c in class_ids])
    )

    query_ids = query_ids[keep]
    class_ids = class_ids[keep]
    confidences = confidences[keep]

    selected = boxes[query_ids]

    cx = selected[:, 0]
    cy = selected[:, 1]
    bw = np.maximum(selected[:, 2], 0)
    bh = np.maximum(selected[:, 3], 0)

    x1 = (cx - bw / 2.0) * image_w
    y1 = (cy - bh / 2.0) * image_h
    x2 = (cx + bw / 2.0) * image_w
    y2 = (cy + bh / 2.0) * image_h

    xyxy = np.stack([x1, y1, x2, y2], axis=1)

    return xyxy, class_ids, confidences


print("===== RF-DETR NANO RAW TENSORRT =====")

logger = trt.Logger(trt.Logger.WARNING)

with open(ENGINE, "rb") as f:
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(f.read())

if engine is None:
    raise RuntimeError("No se pudo cargar el engine")

context = engine.create_execution_context()

print("TensorRT:", trt.__version__)

for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)

    print(
        name,
        "shape=", tuple(engine.get_tensor_shape(name)),
        "dtype=", engine.get_tensor_dtype(name),
        "mode=", engine.get_tensor_mode(name)
    )

input_name = "input"
boxes_name = "dets"
logits_name = "labels"

input_array = np.empty(
    (1, 3, INPUT_H, INPUT_W),
    dtype=np.float32
)

boxes = np.empty(
    (1, 300, 4),
    dtype=np.float32
)

logits = np.empty(
    (1, 300, 91),
    dtype=np.float32
)

err, d_input = cudart.cudaMalloc(input_array.nbytes)
cuda_check(err)

err, d_boxes = cudart.cudaMalloc(boxes.nbytes)
cuda_check(err)

err, d_logits = cudart.cudaMalloc(logits.nbytes)
cuda_check(err)

err, stream = cudart.cudaStreamCreate()
cuda_check(err)

context.set_tensor_address(input_name, int(d_input))
context.set_tensor_address(boxes_name, int(d_boxes))
context.set_tensor_address(logits_name, int(d_logits))

cap = cv2.VideoCapture(VIDEO)

ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError("No se pudo leer el frame")

h, w = frame.shape[:2]

input_array[:] = preprocess(frame)

cuda_check(
    cudart.cudaMemcpyAsync(
        d_input,
        input_array.ctypes.data,
        input_array.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        stream
    )[0]
)

if not context.execute_async_v3(stream_handle=int(stream)):
    raise RuntimeError("TensorRT execute_async_v3 falló")

cuda_check(
    cudart.cudaMemcpyAsync(
        boxes.ctypes.data,
        d_boxes,
        boxes.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        stream
    )[0]
)

cuda_check(
    cudart.cudaMemcpyAsync(
        logits.ctypes.data,
        d_logits,
        logits.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        stream
    )[0]
)

cuda_check(cudart.cudaStreamSynchronize(stream)[0])

xyxy, class_ids, confidences = decode(
    boxes,
    logits,
    w,
    h
)

print()
print("Detecciones:", len(xyxy))

annotated = frame.copy()

for i, (box, class_id, conf) in enumerate(
    zip(xyxy, class_ids, confidences)
):
    x1, y1, x2, y2 = box.astype(int)

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))

    name = COCO_CLASSES[int(class_id)]

    print(
        f"{i:02d}: "
        f"{name:15s} "
        f"{conf:.3f} "
        f"[{x1}, {y1}, {x2}, {y2}]"
    )

    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated,
        f"{name} {conf:.2f}",
        (x1, max(20, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )

cv2.imwrite(OUTPUT, annotated)

print()
print("Guardado:", OUTPUT)

cuda_check(cudart.cudaFree(d_input)[0])
cuda_check(cudart.cudaFree(d_boxes)[0])
cuda_check(cudart.cudaFree(d_logits)[0])
cuda_check(cudart.cudaStreamDestroy(stream)[0])
