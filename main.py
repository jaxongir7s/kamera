"""Vision Assistant: show an object to the webcam and OpenAI's vision model
identifies it and shares a short fact.

Keys: SPACE = identify | A = live (automatic) mode | C = clear | Q = quit
"""

import base64
import os
import sys
import textwrap
import threading
import time

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

load_dotenv()

API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
MODEL = os.getenv("MODEL", "gpt-4o")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
LANGUAGE = os.getenv("LANGUAGE", "uz").lower()
AUTO_INTERVAL = float(os.getenv("AUTO_INTERVAL", "4"))
FONT_PATH = os.getenv("FONT_PATH", "")

WINDOW_TITLE = "Vision Assistant (OpenAI)"
MAX_IMAGE_SIDE = 800

PROMPTS = {
    "uz": (
        "Ushbu rasmdagi asosiy ob'yekt nima? "
        "U haqida qisqacha va qiziqarli ma'lumot ber."
    ),
    "en": (
        "What is the main object in this image? "
        "Provide a brief and interesting fact about it."
    ),
}

UI = {
    "uz": {
        "busy": "Tahlil qilinmoqda, kuting...",
        "done": "Tayyor!  [C] - tozalash",
        "live": "JONLI REJIM",
        "wait": "Biroz kuting, oldingi so'rov ustida ishlanmoqda...",
        "searching": "[AI] Ma'lumot qidirilmoqda, kuting...",
        "result": "NATIJA",
        "error": "Xatolik",
        "hint": "SPACE - aniqlash | A - jonli | C - tozalash | Q - chiqish",
    },
    "en": {
        "busy": "Analyzing, please wait...",
        "done": "Done!  [C] - clear",
        "live": "LIVE MODE",
        "wait": "Please wait, the previous request is still running...",
        "searching": "[AI] Looking up information, please wait...",
        "result": "RESULT",
        "error": "Error",
        "hint": "SPACE - identify | A - live | C - clear | Q - quit",
    },
}
T = UI.get(LANGUAGE, UI["uz"])
PROMPT = PROMPTS.get(LANGUAGE, PROMPTS["uz"])

FONT_CANDIDATES = [
    FONT_PATH,
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]


def load_font(size):
    """Return a TrueType font that supports Uzbek letters (o', g', etc.)."""
    for path in FONT_CANDIDATES:
        if path and os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


FONT = load_font(20)
FONT_SMALL = load_font(16)

client = None  # created in main() after the API key is validated


class State:
    """Shared state between the UI loop and the worker thread."""

    def __init__(self):
        self.lock = threading.Lock()
        self.busy = False
        self.text = ""
        self.is_error = False

    def start(self):
        """Mark a request as started. Returns False if one is already running."""
        with self.lock:
            if self.busy:
                return False
            self.busy = True
            self.text = ""
            self.is_error = False
            return True

    def finish(self, text, is_error=False):
        with self.lock:
            self.text = text
            self.is_error = is_error
            self.busy = False

    def clear(self):
        with self.lock:
            if not self.busy:
                self.text = ""
                self.is_error = False

    def snapshot(self):
        with self.lock:
            return self.busy, self.text, self.is_error


state = State()


def encode_image(frame):
    """Downscale the frame and return it as a base64 JPEG string."""
    height, width = frame.shape[:2]
    if max(height, width) > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / max(height, width)
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Could not encode the camera frame as JPEG.")
    return base64.b64encode(buffer).decode("utf-8")


def analyze_frame(frame):
    """Worker thread: send one frame to the vision model."""
    try:
        base64_image = encode_image(frame)
        print("\n" + T["searching"])

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        answer = (response.choices[0].message.content or "").strip()
        print("\n" + "=" * 30)
        print(f"{T['result']}:")
        print(answer)
        print("=" * 30 + "\n")
        state.finish(answer)
    except Exception as exc:  # network, auth, quota, etc.
        message = f"{T['error']}: {exc}"
        print(f"\n[{T['error'].upper()}] {exc}")
        state.finish(message, is_error=True)


def request_analysis(frame):
    if not state.start():
        return False
    threading.Thread(target=analyze_frame, args=(frame,), daemon=True).start()
    return True


def draw_overlay(frame, busy, text, is_error, live_mode):
    """Draw status text with Pillow so non-ASCII letters render correctly."""
    height, width = frame.shape[:2]
    out = frame.copy()

    lines = []  # (text, rgb color, font)
    if live_mode:
        lines.append((T["live"], (255, 80, 80), FONT))
    if busy:
        lines.append((T["busy"], (255, 255, 0), FONT))
    elif text:
        lines.append((T["done"], (255, 80, 80) if is_error else (0, 255, 0), FONT))

    # Result panel at the bottom of the frame.
    wrapped = []
    if text and not busy:
        chars_per_line = max(20, width // 10)
        wrapped = textwrap.wrap(text, width=chars_per_line)
        max_lines = max(1, (height // 2) // 22)
        if len(wrapped) > max_lines:
            wrapped = wrapped[:max_lines]
            wrapped[-1] = wrapped[-1].rstrip(" .,") + "..."

    panel_height = 22 * len(wrapped) + 16 if wrapped else 0
    if panel_height:
        overlay = out.copy()
        cv2.rectangle(overlay, (0, height - panel_height), (width, height), (0, 0, 0), -1)
        out = cv2.addWeighted(overlay, 0.6, out, 0.4, 0)

    image = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)

    y = 10
    for content, color, font in lines:
        draw.text((12, y), content, font=font, fill=color)
        y += 28

    for i, line in enumerate(wrapped):
        draw.text((12, height - panel_height + 8 + i * 22), line, font=FONT_SMALL, fill=(255, 255, 255))

    if not wrapped:
        draw.text((12, height - 26), T["hint"], font=FONT_SMALL, fill=(200, 200, 200))

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def main():
    global client

    if not API_KEY:
        sys.exit(
            "API key not found. Copy .env.example to .env and set API_KEY "
            "(or OPENAI_API_KEY) to your OpenAI key."
        )
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=30.0, max_retries=1)

    print("Starting camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        sys.exit("Camera not found or cannot be opened. Try CAMERA_INDEX=1 in .env.")

    print(f"\n=== READY === (model: {MODEL})")
    print(T["hint"] + "\n")

    live_mode = False
    last_auto = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Could not read a frame from the camera.")
                break

            busy, text, is_error = state.snapshot()

            if live_mode and not busy and time.time() - last_auto >= AUTO_INTERVAL:
                last_auto = time.time()
                request_analysis(frame.copy())

            cv2.imshow(WINDOW_TITLE, draw_overlay(frame, busy, text, is_error, live_mode))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                state.clear()
            elif key == ord("a"):
                live_mode = not live_mode
                last_auto = 0.0
                print("Live mode:", "ON" if live_mode else "OFF")
            elif key == 32:  # SPACE
                if not request_analysis(frame.copy()):
                    print(T["wait"])
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Program closed.")


if __name__ == "__main__":
    main()
