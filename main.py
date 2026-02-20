import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import base64
import requests
import io

# ============================================================================
# 1. ARSITEKTUR MODEL LOKAL
# ============================================================================
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ============================================================================
# 2. FUNGSI MODEL LOKAL (OFFLINE)
# ============================================================================
def clean_class_name(raw_label):
    parts = raw_label.split('_')
    if parts[0] == 'math':
        clean_char = "_".join(parts[1:-1])
        symbol_map = {
            'equal': '=', 'plus': '+', 'minus': '-', 'div': '÷', 'times': '×',
            'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'sigma': 'Σ', 'theta': 'θ',
            'pi': 'π', 'sqrt': '√', 'left_bracket': '(', 'right_bracket': ')',
            'exclamation': '!', 'question': '?', 'gt': '>', 'lt': '<'
        }
        return symbol_map.get(clean_char, clean_char)
    elif parts[0] == 'emnist':
        return parts[1]
    return raw_label

def local_prediction(image_data, model, class_names):
    gray = cv2.cvtColor(image_data, cv2.COLOR_RGBA2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "Kosong"

    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes), key=lambda b: b[1][0]))

    full_text = ""
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 10 or h < 10:
            continue
        roi = thresh[y:y+h, x:x+w]
        max_side = max(w, h)
        square_img = np.zeros((max_side, max_side), dtype=np.uint8)
        off_x, off_y = (max_side - w) // 2, (max_side - h) // 2
        square_img[off_y:off_y+h, off_x:off_x+w] = roi
        pad = int(max_side * 0.2)
        square_img = cv2.copyMakeBorder(square_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        final = cv2.resize(square_img, (28, 28), interpolation=cv2.INTER_AREA)
        final_input = 255 - final
        img_norm = (final_input.astype('float32') / 255.0 - 0.5) / 0.5
        tensor_input = torch.tensor(img_norm).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            output = model(tensor_input)
            _, predicted = torch.max(output, 1)
            full_text += clean_class_name(class_names[predicted.item()])
    return full_text

# ============================================================================
# 3. FUNGSI OPENROUTER API
# ============================================================================
OPENROUTER_MODELS = {
    "🤖 Auto (OpenRouter pilih terbaik)":       "openrouter/auto",
    "🏆 Qwen 2.5 VL 72B (Terbaik OCR/Math)":   "qwen/qwen2.5-vl-72b-instruct:free",
    "⚡ Qwen 2.5 VL 32B (Lebih cepat)":         "qwen/qwen2.5-vl-32b-instruct:free",
    "🦙 Llama 3.2 Vision 11B":                  "meta-llama/llama-3.2-11b-vision-instruct:free",
    "🌙 Kimi VL A3B Thinking":                  "moonshotai/kimi-vl-a3b-thinking:free",
    "✨ Mistral Small 3.1 24B":                  "mistralai/mistral-small-3.1-24b-instruct:free",
    "💎 Google Gemma 3 27B":                    "google/gemma-3-27b-it:free",
    "🔷 Google Gemma 3 12B":                    "google/gemma-3-12b-it:free",
}

# Prompt khusus agar AI langsung hitung hasilnya, bukan cuma tulis ulang ekspresi
PROMPT_TEXT = """Kamu adalah kalkulator matematika cerdas. Lihat tulisan tangan ini dan:
1. Kenali ekspresi atau soal matematikanya
2. HITUNG dan berikan HASIL AKHIRNYA saja (angka / nilai)
3. Jika hasilnya irasional (seperti akar yang tidak bisa disederhanakan), berikan nilai desimalnya dibulatkan 4 angka
4. Format jawaban: [ekspresi] = [hasil]
Contoh: √16 = 4, 7+7 = 14, √2 = 1.4142
Jawab singkat saja, TANPA penjelasan panjang."""

def ask_openrouter(image_data, api_key, model_id):
    if not api_key:
        return "⚠️ Error: API Key kosong!", None

    try:
        img_pil = Image.fromarray(image_data.astype('uint8'), 'RGBA').convert('RGB')
        buffered = io.BytesIO()
        img_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://hybrid-ai-recognizer.streamlit.app",
            "X-Title": "Hybrid AI Recognizer",
        }
        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT_TEXT},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                    ]
                }
            ],
            "max_tokens": 200,
        }

        response = requests.post(url, json=payload, headers=headers, timeout=30)

        if response.status_code == 200:
            res_json = response.json()
            text = res_json['choices'][0]['message']['content'].strip()
            model_used = res_json.get('model', model_id)
            return text, model_used
        elif response.status_code == 429:
            return "⚠️ Rate limit. Coba ganti model lain di sidebar.", None
        elif response.status_code == 401:
            return "❌ API Key tidak valid. Cek di https://openrouter.ai/settings/keys", None
        elif response.status_code == 402:
            return "❌ Credit habis. Cek di https://openrouter.ai/credits", None
        elif response.status_code == 404:
            err = response.json().get('error', {}).get('message', '')
            return f"❌ Model tidak tersedia: {err}\nGanti model lain di sidebar.", None
        else:
            return f"❌ Error {response.status_code}: {response.text[:200]}", None

    except requests.exceptions.Timeout:
        return "❌ Timeout 30 detik. Coba lagi.", None
    except requests.exceptions.ConnectionError:
        return "❌ Tidak ada koneksi internet.", None
    except Exception as e:
        return f"❌ Kesalahan: {str(e)}", None


# ============================================================================
# 4. LOAD RESOURCES
# ============================================================================
@st.cache_resource
def load_resources():
    try:
        path = 'best_model.pth'
        checkpoint = torch.load(path, map_location='cpu')
        model = Net(len(checkpoint['class_names']))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, checkpoint['class_names']
    except FileNotFoundError:
        return None, "File 'best_model.pth' tidak ditemukan."
    except Exception as e:
        return None, str(e)


# ============================================================================
# 5. AMBIL API KEY: dari st.secrets (deploy) atau input manual (lokal)
# ============================================================================
def get_api_key():
    """
    Urutan prioritas:
    1. st.secrets["OPENROUTER_API_KEY"]  → dipakai saat deploy di Streamlit Cloud
    2. Input manual dari sidebar          → dipakai saat run lokal
    """
    try:
        secret_key = st.secrets["OPENROUTER_API_KEY"]
        if secret_key:
            return secret_key, True  # (key, dari_secrets)
    except Exception:
        pass
    return None, False


# ============================================================================
# 6. UI UTAMA
# ============================================================================
def main():
    st.set_page_config(page_title="Hybrid AI Recognizer", layout="wide", page_icon="✍️")

    st.markdown(
        "<h1 style='text-align: center;'>🧠 Hybrid AI: Local CNN + OpenRouter Cloud</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center;'>Scan Lokal untuk huruf tunggal • OpenRouter untuk kalkulasi rumus</p>",
        unsafe_allow_html=True
    )

    # ---- CEK API KEY ----
    secret_key, from_secrets = get_api_key()

    # ---- SIDEBAR ----
    with st.sidebar:
        st.header("🔑 API Settings")

        if from_secrets:
            # Deploy mode: key dari secrets, tidak ditampilkan ke user
            api_key = secret_key
            st.success("✅ API Key terkonfigurasi (dari server)")
        else:
            # Lokal mode: user input manual
            api_key = st.text_input(
                "OpenRouter API Key",
                value="",
                type="password",
                placeholder="sk-or-v1-...",
                help="Dapatkan API Key gratis di openrouter.ai/settings/keys"
            )
            if api_key:
                st.success("✅ API Key tersimpan")
            else:
                st.warning("⚠️ Masukkan API Key OpenRouter")
                st.markdown("[🔗 Dapatkan API Key gratis](https://openrouter.ai/settings/keys)")

        st.divider()

        st.write("### 🤖 Pilih Model AI")
        selected_model_name = st.selectbox(
            "Model (semua FREE)",
            options=list(OPENROUTER_MODELS.keys()),
            index=0,
        )
        selected_model_id = OPENROUTER_MODELS[selected_model_name]
        st.caption(f"`{selected_model_id}`")

        st.divider()
        st.write("### 📖 Petunjuk:")
        st.write("1. Tulis angka/huruf/rumus di kanvas.")
        st.write("2. Klik **Scan Lokal** (CNN, cepat).")
        st.write("3. Klik **Scan AI Cloud** (hitung hasil).")
        st.divider()
        st.info("💡 Jika satu model error/rate limit, ganti model lain atau pilih **Auto**.")

    # ---- LOAD MODEL LOKAL ----
    local_model, class_names_or_error = load_resources()

    # ---- LAYOUT UTAMA ----
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.subheader("📝 Papan Tulis")
        canvas = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=12,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=300,
            width=600,
            drawing_mode="freedraw",
            key="canvas"
        )
        if st.button("🗑️ Hapus Papan"):
            st.rerun()

    with col2:
        st.subheader("🔍 Hasil Analisis")

        has_drawing = (
            canvas.image_data is not None and
            np.mean(canvas.image_data) < 254
        )

        if has_drawing:

            # ---- TOMBOL SCAN LOKAL ----
            if st.button("⚡ Scan Lokal (CNN)", type="primary", use_container_width=True):
                if local_model is not None:
                    with st.spinner("Model Lokal sedang bekerja..."):
                        result = local_prediction(canvas.image_data, local_model, class_names_or_error)
                    st.markdown(f"**Hasil Lokal:** `{result}`")
                else:
                    st.error(f"❌ {class_names_or_error}")

            # ---- TOMBOL SCAN OPENROUTER ----
            if st.button("✨ Scan AI Cloud (OpenRouter)", type="secondary", use_container_width=True):
                if not api_key:
                    st.error("❌ Masukkan API Key di sidebar!")
                else:
                    with st.spinner("AI sedang menghitung..."):
                        result, model_used = ask_openrouter(canvas.image_data, api_key, selected_model_id)

                    if result.startswith("❌") or result.startswith("⚠️"):
                        st.error(result)
                    else:
                        if model_used:
                            st.caption(f"Model: `{model_used}`")

                        # Tampilkan hasil dengan styling besar & jelas
                        st.markdown(f"""
                        <div style='background-color: #E8F5E9; padding: 20px; border-radius: 12px;
                                    border-left: 6px solid #4CAF50; margin-top: 10px;'>
                            <p style='margin:0; color:#1B5E20; font-size:14px;'>Hasil Kalkulasi:</p>
                            <h2 style='margin: 5px 0 0 0; color:#2E7D32; font-size:32px;'>{result}</h2>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("✏️ Silakan tulis sesuatu di papan tulis.")


if __name__ == "__main__":
    main()