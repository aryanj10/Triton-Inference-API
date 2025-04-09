from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List
import numpy as np
from PIL import Image
import io
import time
import tritonclient.http as httpclient
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="Cluster Classifier Inference API", description="Batch inference via Triton")

client = httpclient.InferenceServerClient(url="localhost:8000")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess(file: UploadFile):
    try:
        content = file.file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        tensor = transform(img).numpy()
        return (file.filename, tensor)
    except Exception as e:
        print(f"⚠️ Error processing {file.filename}: {e}")
        return None

@app.post("/predict_batch/")
async def predict_batch(files: List[UploadFile] = File(...)):
    if len(files) > 100000:
        return JSONResponse(status_code=400, content={"error": "Upload limit is 100000 images."})

    try:
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(preprocess, files))

        t1 = time.time()  # End of preprocessing
        results = [r for r in results if r is not None]

        if not results:
            return JSONResponse(status_code=400, content={"error": "No valid images found."})

        filenames, images = zip(*results)
        input_batch = np.stack(images).astype(np.float32)

        input0 = httpclient.InferInput("input__0", input_batch.shape, "FP32")
        input0.set_data_from_numpy(input_batch)
        output0 = httpclient.InferRequestedOutput("output__0")

        t2 = time.time()

        response = client.infer("classifier", inputs=[input0], outputs=[output0])

        t3 = time.time()  # End of inference

        output_data = response.as_numpy("output__0")
        predicted_indices = np.argmax(output_data, axis=1)

        results = [
            {"filename": name, "predicted_cluster": int(pred)}
            for name, pred in zip(filenames, predicted_indices)
        ]

        return JSONResponse(content={
            "results": results,
            "timing": {
                "total_images": len(results),
                "preprocessing_sec": round(t1 - t0, 4),
                "inference_sec": round(t3 - t2, 4),
                "total_sec": round(t3 - t0, 4),
                "average_latency_per_image_ms": round((t3 - t0) / len(results) * 1000, 2)
            }
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
