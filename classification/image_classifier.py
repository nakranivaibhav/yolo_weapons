import os
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import time
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

INPUT_DIR = "/workspace/yolo_dangerous_weapons/captum/output/parking_lot_front/crops2"                  # folder with input images
OUTPUT_DIR = "classified"
WEAPON_DIR = Path(OUTPUT_DIR) / "weapons"
NON_WEAPON_DIR = Path(OUTPUT_DIR) / "non_weapons"

SUPPORTED_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

MODEL_NAME = "gemini-2.5-flash"  # fast + cheap + vision capable
SLEEP_BETWEEN_CALLS = 0.2         # avoid rate limiting

# -------------------------------------------------
# PROMPT (THIS IS CRITICAL)
# -------------------------------------------------

PROMPT = """
You are a strict visual weapon classifier.

Look at the image and decide if it contains a REAL WEAPON.

Weapons include:
- Guns (handgun, pistol, revolver)
- Rifles
- Shotguns
- Knives used as weapons
- Batons, machetes, crowbars used threateningly
- Baseball bats ONLY if clearly used as a weapon

NON-WEAPON includes:
- Empty hands
- Tools (hammer, wrench, screwdriver)
- Phones, wallets, bags
- umbrella
- People with no visible weapon

Respond with EXACTLY ONE WORD:
WEAPON
or
NON_WEAPON
"""

# -------------------------------------------------
# SETUP GEMINI
# -------------------------------------------------

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not set")

genai.configure(api_key=api_key)
model = genai.GenerativeModel(MODEL_NAME)

# -------------------------------------------------
# HELPERS
# -------------------------------------------------

def classify_image(image_path: Path) -> str:
    """Returns 'WEAPON' or 'NON_WEAPON'"""
    img = Image.open(image_path).convert("RGB")

    response = model.generate_content(
        [PROMPT, img],
        generation_config={
            "temperature": 0.0,
            "max_output_tokens": 5
        }
    )

    text = response.text.strip().upper()
    if "WEAPON" in text and "NON" not in text:
        return "WEAPON"
    return "NON_WEAPON"

# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():
    input_dir = Path(INPUT_DIR)
    if not input_dir.exists():
        raise RuntimeError(f"Input dir not found: {INPUT_DIR}")

    WEAPON_DIR.mkdir(parents=True, exist_ok=True)
    NON_WEAPON_DIR.mkdir(parents=True, exist_ok=True)

    images = [
        p for p in input_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS
    ]

    print(f"Found {len(images)} images")

    for img_path in tqdm(images, desc="Classifying"):
        try:
            label = classify_image(img_path)

            if label == "WEAPON":
                dst = WEAPON_DIR / img_path.name
            else:
                dst = NON_WEAPON_DIR / img_path.name

            shutil.copy(img_path, dst)

            time.sleep(SLEEP_BETWEEN_CALLS)

        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")

    print("\n✅ Classification complete")
    print(f"Weapons: {len(list(WEAPON_DIR.iterdir()))}")
    print(f"Non-weapons: {len(list(NON_WEAPON_DIR.iterdir()))}")

if __name__ == "__main__":
    main()
