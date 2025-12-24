import os
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import time
from dotenv import load_dotenv
import json
from datetime import datetime

load_dotenv()

import google.generativeai as genai

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

INPUT_DIR = "/workspace/yolo_dangerous_weapons/classification/crops/outside_left"
OUTPUT_DIR = "classified/outside_left"
HOLDING_SOMETHING_DIR = Path(OUTPUT_DIR) / "holding_something"
HOLDING_NOTHING_DIR = Path(OUTPUT_DIR) / "holding_nothing"
UNCERTAIN_DIR = Path(OUTPUT_DIR) / "uncertain"  # For ambiguous cases

SUPPORTED_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

MODEL_NAME = "gemini-2.0-flash-exp"  # Latest model with better vision
SLEEP_BETWEEN_CALLS = 0.2  # Avoid rate limiting
MAX_RETRIES = 3  # Retry failed classifications

# Save classification log
LOG_FILE = Path(OUTPUT_DIR) / "classification_log.json"

# -------------------------------------------------
# PROMPT - HAND STATE CLASSIFICATION
# -------------------------------------------------

PROMPT = """
You are a precise visual classifier that determines if a person is holding something in their hands.

Analyze the image and classify into ONE category:

HOLDING_SOMETHING:
- Person is actively holding/grasping any object
- Objects include: weapons, tools, phones, bags, bottles, papers, ANY item
- Both hands or one hand holding something
- Object is clearly visible in hand(s)

HOLDING_NOTHING:
- Hands are completely empty
- Hands are open, relaxed, or in pockets
- Person's hands are visible but not holding anything
- Arms are crossed or hands are by their sides

UNCERTAIN:
- Hands are not visible (behind back, out of frame, obscured)
- Image quality too poor to determine
- Hands are partially visible but unclear if holding something
- Cannot make confident determination

Critical rules:
- Focus ONLY on what is in the hands
- Ignore background objects not being held
- If you see an object in hand = HOLDING_SOMETHING
- If hands are clearly empty = HOLDING_NOTHING
- If you cannot tell = UNCERTAIN

Respond with EXACTLY ONE WORD:
HOLDING_SOMETHING
or
HOLDING_NOTHING
or
UNCERTAIN

Do not include any explanation, only the classification.
"""

# -------------------------------------------------
# SETUP GEMINI
# -------------------------------------------------

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not set. Please add it to your .env file")

genai.configure(api_key=api_key)
model = genai.GenerativeModel(MODEL_NAME)

# -------------------------------------------------
# HELPERS
# -------------------------------------------------

def classify_image(image_path: Path, retry_count: int = 0) -> tuple[str, float]:
    """
    Returns (classification, confidence_score)
    classification: 'HOLDING_SOMETHING', 'HOLDING_NOTHING', or 'UNCERTAIN'
    confidence_score: estimated confidence (0-1)
    """
    try:
        img = Image.open(image_path).convert("RGB")

        response = model.generate_content(
            [PROMPT, img],
            generation_config={
                "temperature": 0.0,  # Deterministic
                "max_output_tokens": 10,
                "top_p": 1.0,
                "top_k": 1
            }
        )

        text = response.text.strip().upper()
        
        # Parse response
        if "HOLDING_SOMETHING" in text and "HOLDING_NOTHING" not in text:
            return "HOLDING_SOMETHING", 0.9
        elif "HOLDING_NOTHING" in text:
            return "HOLDING_NOTHING", 0.9
        elif "UNCERTAIN" in text:
            return "UNCERTAIN", 0.5
        else:
            # Ambiguous response - treat as uncertain
            print(f"  ⚠️  Ambiguous response: '{text}' - marking as UNCERTAIN")
            return "UNCERTAIN", 0.3

    except Exception as e:
        if retry_count < MAX_RETRIES:
            print(f"  🔄 Retry {retry_count + 1}/{MAX_RETRIES} for {image_path.name}")
            time.sleep(1)  # Wait before retry
            return classify_image(image_path, retry_count + 1)
        else:
            print(f"  ❌ Failed after {MAX_RETRIES} retries: {e}")
            return "UNCERTAIN", 0.0


def save_classification_log(log_data: list, log_path: Path):
    """Save classification results to JSON log"""
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)


def print_statistics(log_data: list):
    """Print detailed statistics"""
    total = len(log_data)
    holding_something = sum(1 for x in log_data if x['classification'] == 'HOLDING_SOMETHING')
    holding_nothing = sum(1 for x in log_data if x['classification'] == 'HOLDING_NOTHING')
    uncertain = sum(1 for x in log_data if x['classification'] == 'UNCERTAIN')
    
    avg_confidence = sum(x['confidence'] for x in log_data) / total if total > 0 else 0
    
    print("\n" + "="*60)
    print("📊 CLASSIFICATION STATISTICS")
    print("="*60)
    print(f"Total images processed: {total}")
    print(f"\n📦 Holding Something: {holding_something} ({holding_something/total*100:.1f}%)")
    print(f"✋ Holding Nothing:   {holding_nothing} ({holding_nothing/total*100:.1f}%)")
    print(f"❓ Uncertain:         {uncertain} ({uncertain/total*100:.1f}%)")
    print(f"\n🎯 Average Confidence: {avg_confidence:.2f}")
    print("="*60)


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():
    input_dir = Path(INPUT_DIR)
    if not input_dir.exists():
        raise RuntimeError(f"Input dir not found: {INPUT_DIR}")

    # Create output directories
    HOLDING_SOMETHING_DIR.mkdir(parents=True, exist_ok=True)
    HOLDING_NOTHING_DIR.mkdir(parents=True, exist_ok=True)
    UNCERTAIN_DIR.mkdir(parents=True, exist_ok=True)

    # Find all images
    images = [
        p for p in input_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS
    ]

    if len(images) == 0:
        print(f"❌ No images found in {INPUT_DIR}")
        print(f"   Supported formats: {', '.join(SUPPORTED_EXTS)}")
        return

    print(f"🔍 Found {len(images)} images to classify")
    print(f"🤖 Using model: {MODEL_NAME}")
    print(f"📁 Output directory: {OUTPUT_DIR}\n")

    classification_log = []
    start_time = time.time()

    for img_path in tqdm(images, desc="Classifying images"):
        try:
            # Classify
            label, confidence = classify_image(img_path)

            # Determine destination directory
            if label == "HOLDING_SOMETHING":
                dst_dir = HOLDING_SOMETHING_DIR
            elif label == "HOLDING_NOTHING":
                dst_dir = HOLDING_NOTHING_DIR
            else:
                dst_dir = UNCERTAIN_DIR

            # Copy to destination
            dst = dst_dir / img_path.name
            shutil.copy(img_path, dst)

            # Log result
            classification_log.append({
                "filename": img_path.name,
                "classification": label,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            })

            # Rate limiting
            time.sleep(SLEEP_BETWEEN_CALLS)

        except KeyboardInterrupt:
            print("\n\n⚠️  Classification interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error processing {img_path.name}: {e}")
            classification_log.append({
                "filename": img_path.name,
                "classification": "ERROR",
                "confidence": 0.0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

    # Save log
    save_classification_log(classification_log, LOG_FILE)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Print results
    print("\n✅ Classification complete!")
    print(f"⏱️  Time elapsed: {elapsed_time:.1f} seconds")
    print(f"📄 Log saved to: {LOG_FILE}")
    
    # Print statistics
    print_statistics(classification_log)
    
    # Print directory info
    print(f"\n📂 Output directories:")
    print(f"   Holding Something: {HOLDING_SOMETHING_DIR}")
    print(f"   Holding Nothing:   {HOLDING_NOTHING_DIR}")
    print(f"   Uncertain:         {UNCERTAIN_DIR}")


if __name__ == "__main__":
    main()