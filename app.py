import streamlit as st 
import os
import cv2
import requests
import textwrap
import numpy as np
from together import Together
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

# ✅ Load environment variables
load_dotenv()

# ✅ Fetch API keys from .env
TOGETHER_AI_API_KEY = os.getenv("TOGETHER_AI_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

# ✅ Check if API keys are loaded properly
if not TOGETHER_AI_API_KEY or not PEXELS_API_KEY:
    st.error("❌ API keys are missing. Make sure you have a valid `.env` file.")
    st.stop()

# ✅ Initialize Together.AI Client
client = Together(api_key=TOGETHER_AI_API_KEY)

# ============================
# 📌 Function: Generate Script via Together.AI
# ============================
def generate_script(topic, style):
    """Generates a YouTube script using Together.AI's LLaMA 3.3 70B."""
    
    prompt = (
        f"Write a detailed YouTube script about '{topic}' in {style} style. The script should include:\n"
        "- Introduction (hook, background)\n"
        "- Main Content (key sections, storytelling, examples)\n"
        "- Conclusion (summary, call to action)."
    )

    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>", "<|eom_id|>"],
            stream=False
        )

        if hasattr(response, "choices"):
            script = response.choices[0].message.content
        else:
            script = "Error: Could not generate script."

        # ✅ Save script to file
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/script.txt", "w", encoding="utf-8") as file:
            file.write(script)

        return script
    except Exception as e:
        st.error(f"❌ Error generating script: {e}")
        return None

# ============================
# 📌 Function: Fetch Enough Images from Pexels
# ============================
def fetch_images(script, num_images):
    """Fetch enough images from Pexels API (at least one per sentence)."""
    if not script:
        return []

    query = " ".join(script.split()[:5])  
    headers = {"Authorization": PEXELS_API_KEY}
    url = f"https://api.pexels.com/v1/search?query={query}&per_page={num_images}"

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        return [photo["src"]["large"] for photo in data["photos"]]
    else:
        return []

# ============================
# 📌 Function: Generate Video from Script + Pexels Images
# ============================
def create_video(script, output_path="outputs/final_video.mp4"):
    """Creates a video by overlaying script text onto Pexels images."""
    
    if not script:
        return None

    sentences = script.split(". ")  # Split script into sentences
    num_sentences = len(sentences)
    
    # ✅ Fetch at least as many images as sentences
    images = fetch_images(script, num_sentences)
    if not images:
        return None

    width, height = 1280, 720
    fps = 1
    temp_video_path = output_path

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    image_paths = []

    for i, sentence in enumerate(sentences):
        img_url = images[i % len(images)]
        img_path = f"outputs/bg_{i}.jpg"
        image_paths.append(img_path)

        # ✅ Download image
        with open(img_path, "wb") as img_file:
            img_file.write(requests.get(img_url).content)

        # ✅ Load image & resize
        frame = cv2.imread(img_path)
        frame = cv2.resize(frame, (width, height))

        # ✅ Add text overlay with word wrapping
        wrapped_text = textwrap.fill(sentence, width=50)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        color = (255, 255, 255)  # White text

        # ✅ Get text size & position
        text_size = cv2.getTextSize(wrapped_text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2

        cv2.putText(frame, wrapped_text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
        video_writer.write(frame)

    video_writer.release()

    # ✅ Delete images after video is generated
    for img_path in image_paths:
        os.remove(img_path)

    return temp_video_path

# ============================
# 📌 Streamlit UI
# ============================
st.set_page_config(page_title="AI YouTube Script to Video", layout="wide")

st.title("🎬 AI YouTube Script & Video Generator")

# ✅ Create two columns
col1, col2 = st.columns(2)

# ============================
# 📌 Column 1: Script Generation & Editing
# ============================
with col1:
    st.subheader("📜 Script Generator")

    # User Input
    topic = st.text_input("📌 Enter a topic:", "History of AI")
    style = st.selectbox("🎭 Choose a style:", ["Informative", "Storytelling", "Documentary"])

    # ✅ Session state for script storage
    if "script_text" not in st.session_state:
        st.session_state.script_text = ""

    # ✅ Generate Script
    if st.button("🚀 Generate Script"):
        st.session_state.script_text = generate_script(topic, style)

    # ✅ Edit the Script
    if st.session_state.script_text:
        updated_script = st.text_area("✏️ Edit the script before making a video:", 
                                      st.session_state.script_text, height=400)

        # ✅ Save script
        with open("outputs/script.txt", "w", encoding="utf-8") as file:
            file.write(updated_script)

        # ✅ Provide script download button
        with open("outputs/script.txt", "r") as file:
            st.download_button("📥 Download Script", file, "script.txt", "text/plain")

# ============================
# 📌 Column 2: Convert Script to Video
# ============================
with col2:
    st.subheader("🎥 Convert Script to Video")

    if st.session_state.script_text:
        if st.button("🎬 Create Video"):
            st.markdown("### 📷 Fetching images & generating video...")

            video_file = create_video(st.session_state.script_text)

            if video_file:
                st.video(video_file)
                with open(video_file, "rb") as file:
                    st.download_button("📥 Download Video", file, "final_video.mp4", "video/mp4")
            else:
                st.error("❌ Failed to generate video. Try again.")
