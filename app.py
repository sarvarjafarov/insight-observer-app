"""
YouTube Content Reaction Study - Streamlit app.
Uses WebRTC (browser camera) to capture images while watching a video; evaluates reactions via OpenAI.
"""

import base64
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from streamlit_webrtc import webrtc_streamer

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
        return "No images captured. Start the stream and capture some snapshots first."

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


def save_frame_from_webrtc(frame) -> Optional[str]:
    """Save a WebRTC frame to images folder as JPEG. Returns filename or None."""
    try:
        img_bgr = frame.to_ndarray(format="bgr24")
        img_rgb = img_bgr[:, :, ::-1]
    except Exception as e:
        logger.warning("Could not get ndarray from frame: %s", e)
        return None
    ensure_images_dir()
    if count_images_in_folder() >= MAX_IMAGES:
        return None
    filename = IMAGES_DIR / f"capture_{int(time.time() * 1000)}.jpg"
    Image.fromarray(img_rgb).save(filename, "JPEG", quality=85)
    logger.info("Image saved: %s (total: %d)", filename.name, count_images_in_folder())
    return filename.name


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

    if "last_video_id" not in st.session_state:
        st.session_state.last_video_id = None
    if "_stream_connected" not in st.session_state:
        st.session_state._stream_connected = False

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

    # ---- Webcam first (so Start button is visible without scrolling) ----
    st.subheader("📷 Webcam capture")
    image_count = count_images_in_folder()
    col1, col2 = st.columns(2)
    with col1:
        evaluate_clicked = st.button("Evaluate Response", type="primary", key="eval_btn")
    with col2:
        st.metric("Images captured", f"{image_count} / {MAX_IMAGES}")

    st.markdown(
    "**Step 1:** Click the green **Start** button below → **Step 2:** In the picker, select your camera (e.g. MacBook Pro Camera) and click **DONE** → "
    "**Step 3:** When the live feed appears here, use **Capture snapshot** to save frames."
)

    # Upload fallback when WebRTC fails (e.g. "Connection taking longer" on prod)
    st.markdown("---")
    st.caption("**Camera not connecting?** (e.g. \"Connection taking longer\" on Streamlit Cloud)")
    uploaded = st.file_uploader(
        "Upload images instead (no camera needed)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="img_upload",
    )
    if uploaded:
        ensure_images_dir()
        current = count_images_in_folder()
        added = 0
        for i, f in enumerate(uploaded):
            if current >= MAX_IMAGES:
                st.warning(f"Only the first {MAX_IMAGES} images are used. Max limit reached.")
                break
            path = IMAGES_DIR / f"upload_{int(time.time() * 1000)}_{i}_{f.name}"
            path.write_bytes(f.getvalue())
            current += 1
            added += 1
            logger.info("Image uploaded: %s (total: %d)", path.name, current)
        if added:
            st.success(f"Added {added} image(s). Use **Evaluate Response** above.")
            st.rerun()

    with st.expander("Camera not showing? Troubleshooting"):
        st.markdown("""
        - **Allow camera** when the browser prompts (Chrome: click Allow in the address bar).
        - Use **Chrome** or **Edge**; Safari can be finicky with WebRTC.
        - **Reload the page** and click Start again.
        - If you're on **Streamlit Cloud**, the app must run over **HTTPS** (it does by default).
        - Make sure no other app (Zoom, FaceTime) is using the camera.
        - **"Connection taking longer / STUN/TURN"** (common on Streamlit Cloud): We use STUN + free TURN relays. If it still fails:
          - **Run locally** for reliable camera: `git clone ... && streamlit run app.py` (no NAT/firewall issues).
          - Try a different network (e.g. mobile hotspot) or disable VPN.
          - Reload and click Start again; wait 10–15 seconds for TURN fallback.
        """)
    ctx = webrtc_streamer(
        key="reaction_capture",
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 30},
            },
            "audio": False,
        },
        rtc_configuration={
            "iceServers": [
                {"urls": "stun:stun.l.google.com:19302"},
                {"urls": "stun:stun1.l.google.com:19302"},
                {"urls": "stun:stun2.l.google.com:19302"},
                {"urls": "stun:stun.stunprotocol.org:3478"},
                # TURN relay for prod when STUN fails (e.g. strict firewalls)
                {"urls": "turn:freeturn.net:3478", "username": "free", "credential": "free"},
                {"urls": "turn:freestun.net:3478", "username": "free", "credential": "free"},
            ]
        },
    )

    if ctx.video_receiver:
        device_label = None
        try:
            track = ctx.video_receiver.get_track()
            if track and getattr(track, "label", None):
                device_label = track.label
        except Exception:
            pass

        if not st.session_state._stream_connected:
            st.session_state._stream_connected = True
            name = device_label or "(device name pending)"
            logger.info("Camera started (WebRTC stream connected — using device: %s)", name)

        if device_label:
            st.info(f"**Using device:** {device_label}")
        else:
            st.info("**Using device:** (starting stream…)")

        if image_count >= MAX_IMAGES:
            st.warning(f"Maximum of {MAX_IMAGES} images reached. Use **Evaluate Response** or select another video to clear.")
        elif st.button("📸 Capture snapshot"):
            try:
                frame = ctx.video_receiver.get_frame()
                saved_name = save_frame_from_webrtc(frame)
                if saved_name:
                    st.success(f"Saved {saved_name}")
                    st.rerun()
                else:
                    st.error("Could not save image or max images reached.")
            except Exception as e:
                st.error(f"Waiting for video to start… ({e})")
    else:
        if st.session_state._stream_connected:
            st.session_state._stream_connected = False
            logger.info("Camera stopped (WebRTC stream stopped).")
        st.info(
            "Click the green **Start** button above → choose your camera (e.g. MacBook Pro Camera) → click **DONE**. "
            "If the feed still doesn’t appear (or you see \"Connection taking longer\"), wait 10–15 seconds or use **Upload images instead** below."
        )

    # YouTube video (below webcam so Start is visible first)
    st.divider()
    st.subheader("📺 Watch the video")
    if iframe_html:
        st.markdown(iframe_html, unsafe_allow_html=True)
    else:
        st.caption("No iframe for this video.")

    # Evaluate Response
    if evaluate_clicked:
        with st.spinner("Evaluating response with OpenAI..."):
            response_text = evaluate_response_with_openai(video_title, video_id)
        st.subheader("Reaction analysis")
        st.write(response_text)


if __name__ == "__main__":
    main()
