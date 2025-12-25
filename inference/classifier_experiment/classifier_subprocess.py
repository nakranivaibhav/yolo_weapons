import sys
import json
import base64
import numpy as np
from pathlib import Path
import torch
from transformers import ConvNextV2ForImageClassification, AutoImageProcessor
from PIL import Image

model_path = sys.argv[1]
conf_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConvNextV2ForImageClassification.from_pretrained(model_path)
model = model.to(device)
model.eval()

processor = AutoImageProcessor.from_pretrained(model_path)

id2label = {int(k): v for k, v in model.config.id2label.items()}

sys.stderr.write("CLASSIFIER_MODEL_READY\n")
sys.stderr.flush()

while True:
    line = sys.stdin.readline()
    if not line:
        break
    
    data = json.loads(line.strip())
    
    crop_bytes = base64.b64decode(data['crop'])
    crop_arr = np.frombuffer(crop_bytes, dtype=np.uint8).reshape(data['shape'])
    
    pil_image = Image.fromarray(crop_arr[:, :, ::-1])
    
    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
    
    predicted_class = int(probs.argmax())
    confidence = float(probs[predicted_class])
    class_name = id2label[predicted_class]
    
    all_probs = {id2label[i]: float(probs[i]) for i in range(len(probs))}
    
    result = {
        'class': predicted_class,
        'class_name': class_name,
        'confidence': confidence,
        'all_probs': all_probs
    }
    
    print(json.dumps(result), flush=True)

