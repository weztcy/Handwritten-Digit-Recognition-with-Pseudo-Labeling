"""
app.py
Interactive Multi-Page Streamlit application for Handwritten Digit Recognition
demonstrating the Semi-Supervised Model trained with Confidence-Based Pseudo-Labeling.
"""

import os
import io
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import DigitCNN

# Page configuration
st.set_page_config(
    page_title="Digit Recognition - Pseudo-Labeling",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #2563EB;
        margin-bottom: 1rem;
    }
    .pred-highlight {
        font-size: 3.5rem;
        font-weight: 800;
        color: #1D4ED8;
        text-align: center;
        line-height: 1;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    device = torch.device("cpu")
    models = {}
    
    ssl_path = "outputs/models/semisupervised_model.pt"
    if os.path.exists(ssl_path):
        m_ssl = DigitCNN().to(device)
        m_ssl.load_state_dict(torch.load(ssl_path, map_location=device, weights_only=True))
        m_ssl.eval()
        models["Semi-Supervised (Self-Training)"] = m_ssl

    base_path = "outputs/models/baseline_model.pt"
    if os.path.exists(base_path):
        m_base = DigitCNN().to(device)
        m_base.load_state_dict(torch.load(base_path, map_location=device, weights_only=True))
        m_base.eval()
        models["Supervised Baseline (10% Labeled)"] = m_base

    upper_path = "outputs/models/upper_bound_model.pt"
    if os.path.exists(upper_path):
        m_upper = DigitCNN().to(device)
        m_upper.load_state_dict(torch.load(upper_path, map_location=device, weights_only=True))
        m_upper.eval()
        models["Upper-Bound Benchmark (100% Labeled)"] = m_upper

    return models


@st.cache_data
def get_sample_test_images():
    try:
        from src.data_loader import read_idx_images, read_idx_labels
        test_imgs = read_idx_images("dataset/t10k-images-idx3-ubyte.gz")
        test_lbls = read_idx_labels("dataset/t10k-labels-idx1-ubyte.gz")
        samples = {}
        for d in range(10):
            idx = int(np.where(test_lbls == d)[0][0])
            samples[f"Digit {d} (Index {idx})"] = test_imgs[idx]
        return samples
    except Exception as e:
        return {}


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """Preprocesses input image to (28, 28) uint8 array matching MNIST."""
    img = pil_img.convert("L")
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.uint8)
    if np.mean(arr) > 127:
        arr = 255 - arr
    return arr


def predict_digit(model: DigitCNN, img_28x28: np.ndarray):
    tensor = torch.tensor(img_28x28, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).numpy()
    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])
    return pred_class, confidence, probs


# --- Sidebar Navigation ---
st.sidebar.markdown("### 📌 Navigasi Halaman")
selected_page = st.sidebar.radio(
    "Pilih Halaman / Modul:",
    [
        "🎯 Live Digit Inference",
        "📊 Model Comparison & Benchmark",
        "🔄 Pseudo-Labeling Diagnostics",
        "📁 Dataset & Architecture Overview"
    ]
)

models = load_models()
sample_images = get_sample_test_images()

# ==========================================
# PAGE 1: LIVE DIGIT INFERENCE
# ==========================================
if selected_page == "🎯 Live Digit Inference":
    st.markdown('<div class="main-header">🎯 Live Handwritten Digit Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Uji coba inferensi langsung menggunakan model Semi-Supervised Learning (Self-Training) vs Supervised Baseline.</div>', unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Konfigurasi Model & Masukan")
    model_choice = st.sidebar.selectbox(
        "Pilih Varian Model:",
        options=list(models.keys()) if models else ["No model found"],
        index=0
    )

    input_source = st.sidebar.radio(
        "Pilih Sumber Masukan:",
        options=["Preset MNIST Samples", "Upload Image", "Ambiguous / Synthetic Test Cases"]
    )

    col1, col2 = st.columns([1, 1.2])

    image_to_process = None
    sample_name = ""

    with col1:
        st.subheader("🖼️ Input Digit Image")

        if input_source == "Preset MNIST Samples":
            if sample_images:
                sample_name = st.selectbox("Pilih sampel digit resmi MNIST:", list(sample_images.keys()), index=3)
                image_to_process = sample_images[sample_name]
            else:
                st.warning("Dataset MNIST tidak dapat dimuat.")

        elif input_source == "Upload Image":
            uploaded_file = st.file_uploader("Unggah citra digit (PNG, JPG):", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                pil_uploaded = Image.open(uploaded_file)
                image_to_process = preprocess_image(pil_uploaded)
                st.image(pil_uploaded, caption="Original Uploaded Image", width=150)
            else:
                st.info("Silakan unggah citra atau gunakan preset digit.")

        elif input_source == "Ambiguous / Synthetic Test Cases":
            case = st.selectbox(
                "Pilih pengujian kasus batas (*edge case*):",
                ["Blank Image (All Zeros)", "Uniform Noise", "Ambiguous '3' or '8' hybrid", "Inverted Ambiguous Stroke"]
            )
            if case == "Blank Image (All Zeros)":
                image_to_process = np.zeros((28, 28), dtype=np.uint8)
            elif case == "Uniform Noise":
                np.random.seed(42)
                image_to_process = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
            elif case == "Ambiguous '3' or '8' hybrid":
                if sample_images and "Digit 3 (Index 18)" in sample_images and "Digit 8 (Index 61)" in sample_images:
                    d3 = sample_images["Digit 3 (Index 18)"]
                    d8 = sample_images["Digit 8 (Index 61)"]
                    image_to_process = np.clip((d3.astype(int) + d8.astype(int)) // 2, 0, 255).astype(np.uint8)
                else:
                    image_to_process = np.full((28, 28), 128, dtype=np.uint8)
            else:
                arr = np.zeros((28, 28), dtype=np.uint8)
                for i in range(28):
                    arr[i, min(i, 27)] = 255
                image_to_process = arr

        if image_to_process is not None:
            fig_img, ax_img = plt.subplots(figsize=(3, 3))
            fig_img.patch.set_facecolor('#FFFFFF')
            ax_img.imshow(image_to_process, cmap="gray_r")
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            ax_img.set_title("Processed Model Input (28x28)", fontsize=10, color='#1E293B', fontweight='bold')
            st.pyplot(fig_img, use_container_width=True)
            plt.close(fig_img)

    with col2:
        st.subheader("🎯 Prediksi & Analisis Probabilitas")

        if image_to_process is not None and model_choice in models:
            selected_model = models[model_choice]
            pred_digit, confidence, probs = predict_digit(selected_model, image_to_process)

            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.95rem; color: #4B5563; font-weight: 600;">PREDICTED DIGIT:</div>
                <div class="pred-highlight">{pred_digit}</div>
                <div style="text-align: center; font-size: 1.1rem; font-weight: 600; color: #1F2937;">
                    Confidence: <span style="color: {'#059669' if confidence >= 0.90 else '#D97706' if confidence >= 0.50 else '#DC2626'};">{confidence * 100:.2f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if confidence < 0.95:
                st.warning("⚠️ Skor keyakinan berada di bawah ambang batas pseudo-labeling (0,95). Dalam self-training, sampel ini ditolak untuk mencegah confirmation bias.")
            else:
                st.success("✅ Prediksi berkeyakinan tinggi (>= 95%). Memenuhi kualifikasi sebagai pseudo-label yang andal.")

            st.markdown("##### 📊 Distribusi Probabilitas Posterior (10 Kelas)")
            fig_bar, ax_bar = plt.subplots(figsize=(6, 3.2))
            fig_bar.patch.set_facecolor('#FFFFFF')
            ax_bar.set_facecolor('#F8FAFC')
            bars = ax_bar.bar(range(10), probs * 100, color="#2563EB", edgecolor="#1E3A8A", linewidth=1.2, alpha=0.85)
            bars[pred_digit].set_color("#059669")
            bars[pred_digit].set_edgecolor("#065F46")
            ax_bar.set_xticks(range(10))
            ax_bar.set_xlabel("Digit Class", fontsize=9, fontweight='bold', color='#1E293B')
            ax_bar.set_ylabel("Probability (%)", fontsize=9, fontweight='bold', color='#1E293B')
            ax_bar.set_ylim(0, 105)
            ax_bar.grid(axis='y', linestyle='--', alpha=0.5, color='#CBD5E1')
            ax_bar.tick_params(colors='#1E293B')

            for bar in bars:
                h = bar.get_height()
                if h >= 1.0:
                    ax_bar.text(bar.get_x() + bar.get_width()/2., h + 2, f"{h:.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1E293B')

            plt.tight_layout()
            st.pyplot(fig_bar, use_container_width=True)
            plt.close(fig_bar)

# ==========================================
# PAGE 2: MODEL COMPARISON & BENCHMARK
# ==========================================
elif selected_page == "📊 Model Comparison & Benchmark":
    st.markdown('<div class="main-header">📊 Model Benchmark & Evaluasi Komparatif</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluasi kuantitatif pada 10.000 citra uji resmi MNIST (Held-out Official Test Set).</div>', unsafe_allow_html=True)

    # Metric KPI Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Supervised Baseline (10%)", "98.46%", "154 eror")
    kpi2.metric("Semi-Supervised (Self-Training)", "99.15%", "+0.69% Akurasi")
    kpi3.metric("Upper-Bound (100% Labeled)", "99.29%", "Batas Teoretis")
    kpi4.metric("Reduksi Kesalahan Uji", "44.81%", "-69 eror")

    st.markdown("---")
    st.subheader("📈 Perbandingan Akurasi & Matriks Konfusi")

    tab_eval1, tab_eval2, tab_eval3 = st.tabs(["📊 Akurasi Uji & F1-Score", "🔍 Matriks Konfusi Komparatif", "⚠️ Analisis Kesalahan Prediksi"])

    with tab_eval1:
        c1, c2 = st.columns(2)
        with c1:
            st.image("outputs/figures/09_test_accuracy_comparison.png", caption="Perbandingan Akurasi Uji Resmi (N=10.000)", use_container_width=True)
        with c2:
            st.image("outputs/figures/11_per_class_f1_comparison.png", caption="Perbandingan F1-Score per Kelas Digit", use_container_width=True)

    with tab_eval2:
        st.image("outputs/figures/10_confusion_matrices.png", caption="Matriks Konfusi: Supervised Baseline vs Semi-Supervised Model", use_container_width=True)

    with tab_eval3:
        st.image("outputs/figures/12_test_error_analysis.png", caption="Sampel Representatif Kesalahan Klasifikasi pada 10.000 Data Uji", use_container_width=True)

# ==========================================
# PAGE 3: PSEUDO-LABELING DIAGNOSTICS
# ==========================================
elif selected_page == "🔄 Pseudo-Labeling Diagnostics":
    st.markdown('<div class="main-header">🔄 Self-Training & Pseudo-Labeling Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Dinamika penyerapan 54.000 data tanpa label melalui 3 putaran seleksi terbobot.</div>', unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3)
    p1.metric("Ambang Batas Keyakinan (τ)", "≥ 0.95", "Threshold Ketat")
    p2.metric("Bobot Loss Pseudo-Label (α)", "0.40", "Regulasi Penalti")
    p3.metric("Total Data Terserap", "52.908 / 54.000", "97.98% Pool Unlabeled")

    st.markdown("---")
    c_diag1, c_diag2 = st.columns(2)
    with c_diag1:
        st.image("outputs/figures/06_pseudolabel_acceptance_by_round.png", caption="Jumlah Sampel Diterima & Akurasi Post-Hoc per Putaran", use_container_width=True)
        st.image("outputs/figures/08_confidence_distribution.png", caption="Distribusi Skor Keyakinan Pseudo-Label", use_container_width=True)
    with c_diag2:
        st.image("outputs/figures/07_pseudolabel_class_distribution.png", caption="Distribusi Kelas Pseudo-Label yang Diterima", use_container_width=True)
        st.image("outputs/figures/05_training_val_accuracy_curves.png", caption="Kurva Konvergensi Akurasi Pembelajaran Iteratif", use_container_width=True)

# ==========================================
# PAGE 4: DATASET & ARCHITECTURE OVERVIEW
# ==========================================
elif selected_page == "📁 Dataset & Architecture Overview":
    st.markdown('<div class="main-header">📁 Dataset MNIST & Arsitektur CNN</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Validasi format biner IDX, pemisahan terstratifikasi, dan spesifikasi arsitektur jaringan.</div>', unsafe_allow_html=True)

    st.markdown("---")
    c_data1, c_data2 = st.columns(2)
    with c_data1:
        st.image("outputs/figures/01_mnist_samples_grid.png", caption="Sampel Representatif Digit Tulisan Tangan MNIST (0-9)", use_container_width=True)
        st.image("outputs/figures/03_data_split_summary.png", caption="Ringkasan Pemisahan Terstratifikasi (10% Labeled vs 90% Unlabeled)", use_container_width=True)
    with c_data2:
        st.image("outputs/figures/02_train_class_distribution.png", caption="Distribusi Kelas Pelatihan 60.000 Citra", use_container_width=True)
        st.image("outputs/figures/04_training_val_loss_curves.png", caption="Kurva Loss Pelatihan dan Validasi", use_container_width=True)

# Footer info
st.divider()
st.caption("🔬 **Project Architecture**: 4-layer Convolutional Neural Network with Batch Normalization & Dropout | **Semi-Supervised Learning**: Confidence-Based Pseudo-Labeling with Weighted Loss Optimization.")
