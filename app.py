"""
YouTube Content Reaction Study - Streamlit app.
Captures webcam images while watching a video and evaluates reactions via OpenAI.
"""

import base64
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------------------------------------------------------
# Logging: all major events to server terminal with timestamps
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Paths and config
# -----------------------------------------------------------------------------
load_dotenv()
IMAGES_DIR = Path("images")
DATA_JSON = Path("youtube_data.json")
PROMPT_TEMPLATE_FILE = Path("prompt_reaction.txt")
OPENAI_MODEL = "gpt-5-nano"
CAPTURE_INTERVAL_SEC = 10
MAX_IMAGES = 20


def ensure_images_dir():
    """Create images directory if it does not exist."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def clear_images_folder():
    """Remove all files in the images folder."""
    if not IMAGES_DIR.exists():
        return
    for f in IMAGES_DIR.iterdir():
        if f.is_file():
            f.unlink()
    logger.info("Images folder cleared.")


def load_video_data():
    """Load video list from youtube_data.json."""
    if not DATA_JSON.exists():
        logger.warning("youtube_data.json not found.")
        return []
    with open(DATA_JSON, encoding="utf-8") as f:
        return json.load(f)


def load_prompt_template():
    """Load prompt template from prompt_reaction.txt."""
    if not PROMPT_TEMPLATE_FILE.exists():
        return "Analyze the viewer's reaction based on these images. Video: {video_title} (ID: {video_id})"
    return PROMPT_TEMPLATE_FILE.read_text(encoding="utf-8").strip()


def get_video_capture():
    """Cross-platform webcam capture. Windows uses CAP_DSHOW."""
    if sys.platform == "win32":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)
    return cap


def capture_frame(cap):
    """Read one frame from the camera. Returns (success, BGR image or None)."""
    if cap is None or not cap.isOpened():
        return False, None
    ret, frame = cap.read()
    return ret, frame


def count_images_in_folder():
    """Count image files in the images folder."""
    if not IMAGES_DIR.exists():
        return 0
    return sum(1 for f in IMAGES_DIR.iterdir() if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png"))


def get_base64_images():
    """Read all images from the images folder and return list of base64 strings (JPEG)."""
    if not IMAGES_DIR.exists():
        return []
    out = []
    for f in sorted(IMAGES_DIR.iterdir()):
        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            data = f.read_bytes()
            b64 = base64.standard_b64encode(data).decode("utf-8")
            out.append(b64)
    return out


def evaluate_response_with_openai(video_title: str, video_id: str) -> str:
    """Format prompt, encode images, call OpenAI (gpt-5-nano), return response text."""
    prompt_template = load_prompt_template()
    prompt_text = prompt_template.format(video_title=video_title, video_id=video_id)

    b64_images = get_base64_images()
    if not b64_images:
        return "No images captured. Start recording and capture some images first."

    content = [{"type": "text", "text": prompt_text}]
    for b64 in b64_images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set in .env")
        return "Error: OPENAI_API_KEY not found in .env."

    logger.info("Calling OpenAI API (model=%s) with %d image(s).", OPENAI_MODEL, len(b64_images))
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=1024,
        )
        result = response.choices[0].message.content or ""
        logger.info("OpenAI API call completed successfully.")
        return result
    except Exception as e:
        logger.exception("OpenAI API call failed: %s", e)
        return f"Error calling OpenAI: {e}"


# -----------------------------------------------------------------------------
# Streamlit app
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="YouTube Content Reaction Study", layout="wide")
    st.title("YouTube Content Reaction Study")

    ensure_images_dir()
    videos = load_video_data()
    if not videos:
        st.error("No video data found. Add entries to youtube_data.json.")
        return

    # Session state
    if "last_video_id" not in st.session_state:
        st.session_state.last_video_id = None
    if "recording" not in st.session_state:
        st.session_state.recording = False
    if "last_capture_time" not in st.session_state:
        st.session_state.last_capture_time = 0
    if "cap" not in st.session_state:
        st.session_state.cap = None

    # Video dropdown
    options = [v["title"] for v in videos]
    selected_title = st.selectbox("Select video", options, key="video_select")
    selected_video = next((v for v in videos if v["title"] == selected_title), None)
    if not selected_video:
        return
    video_id = selected_video["id"]
    video_title = selected_video["title"]
    iframe_html = selected_video.get("iframe", "")

    # Clear images when video selection changes
    if st.session_state.last_video_id is not None and st.session_state.last_video_id != video_id:
        clear_images_folder()
        logger.info("Video changed to '%s', images folder cleared.", video_title)
    st.session_state.last_video_id = video_id

    # Show video iframe
    if iframe_html:
        st.markdown(iframe_html, unsafe_allow_html=True)
    st.divider()

    # Two-column layout
    col1, col2 = st.columns(2)
    with col1:
        recording = st.toggle("Start Recording", key="recording_toggle")
        st.session_state.recording = recording
        evaluate_clicked = st.button("Evaluate Response", type="primary")
    with col2:
        image_count = count_images_in_folder()
        st.metric("Images captured", f"{image_count} / {MAX_IMAGES}")

    # Recording logic: capture every 10 seconds, max 20 images
    if st.session_state.recording:
        if st.session_state.cap is None or not st.session_state.cap.isOpened():
            st.session_state.cap = get_video_capture()
            if st.session_state.cap.isOpened():
                st.session_state.last_capture_time = time.time()  # first capture in 10 sec
                logger.info("Camera started.")
            else:
                st.session_state.cap = None
                st.warning("Could not open camera.")
        cap = st.session_state.cap
        if cap is not None and cap.isOpened() and image_count < MAX_IMAGES:
            now = time.time()
            if now - st.session_state.last_capture_time >= CAPTURE_INTERVAL_SEC:
                ret, frame = capture_frame(cap)
                if ret and frame is not None:
                    ensure_images_dir()
                    filename = IMAGES_DIR / f"capture_{int(now * 1000)}.jpg"
                    cv2.imwrite(str(filename), frame)
                    st.session_state.last_capture_time = now
                    logger.info("Image saved: %s (total: %d)", filename.name, count_images_in_folder())
    else:
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
            logger.info("Camera stopped.")

    # Evaluate button
    if evaluate_clicked:
        with st.spinner("Evaluating response with OpenAI..."):
            response_text = evaluate_response_with_openai(video_title, video_id)
        st.subheader("Reaction analysis")
        st.write(response_text)

    # Rerun to update capture count and periodic capture
    if st.session_state.recording and image_count < MAX_IMAGES:
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
