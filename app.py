import io
import json
import base64
import secrets
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import pywt
from PIL import Image, ImageFilter, ImageOps
import streamlit as st
import pandas as pd
import altair as alt

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import (
    load_pem_private_key, load_pem_public_key,
    Encoding, PrivateFormat, PublicFormat, NoEncryption
)
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# -------------------- Utilities --------------------

def pil_to_array_gray(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L"), dtype=np.float32)


def array_to_pil_gray(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def pil_to_array_rgb(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to RGB numpy array"""
    return np.array(img.convert("RGB"), dtype=np.float32)


def array_to_pil_rgb(arr: np.ndarray) -> Image.Image:
    """Convert RGB numpy array to PIL Image"""
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def bytes_from_image(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def image_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


def image_from_bytes_rgb(b: bytes) -> Image.Image:
    """Load image from bytes and maintain RGB mode"""
    return Image.open(io.BytesIO(b)).convert("RGB")


def b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")


def b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("utf-8"))


def pem_body_base64(pem_bytes: bytes) -> str:
    try:
        lines = pem_bytes.decode("utf-8").splitlines()
    except Exception:
        return ""
    body_lines = [ln.strip() for ln in lines if not ln.startswith("---") and len(ln.strip()) > 0]
    return "".join(body_lines)


# Streamlit image compatibility (new vs old param names)
def st_image_compat(img, caption=None):
    try:
        return st.image(img, caption=caption, use_container_width=True)
    except TypeError:
        return st.image(img, caption=caption, use_column_width=True)



# -------------------- DWT-SVD Watermarking --------------------


@dataclass
class WatermarkSideInfo:
    S_cover: np.ndarray  # Now shape (3, k) for RGB channels
    Uw: np.ndarray      # Now shape (3, m, k) for RGB channels
    Vtw: np.ndarray     # Now shape (3, k, n) for RGB channels
    alpha: float
    cover_shape: Tuple[int, int, int]  # Now (H, W, 3) for RGB
    wm_shape: Tuple[int, int, int]  # Now (H, W, 3) for RGB watermark


def dwt_svd_embed(cover_img: Image.Image, wm_img: Image.Image, alpha: float = 0.05):
    # Convert cover to RGB array
    C = pil_to_array_rgb(cover_img)  # Shape: (H, W, 3)
    
    # Convert watermark to RGB array (preserve colors)
    W_rgb = pil_to_array_rgb(wm_img)  # Shape: (H, W, 3)
    
    # Process each RGB channel separately
    watermarked_channels = []
    side_info_channels = []
    
    for channel in range(3):  # R, G, B channels
        C_channel = C[:, :, channel]
        W_channel = W_rgb[:, :, channel]  # Process watermark channel separately
        
        # 1-level DWT on cover channel
        LL, (LH, HL, HH) = pywt.dwt2(C_channel, 'haar')

        # Resize watermark channel to LL size (preserve color information)
        LL_h, LL_w = LL.shape
        W_channel_resized = Image.fromarray(W_channel.astype(np.uint8), mode='L')
        W_channel_resized = W_channel_resized.resize((LL_w, LL_h), Image.BICUBIC)
        W = np.array(W_channel_resized, dtype=np.float32)

        # SVD on LL and watermark channel
        Uc, Sc, Vct = np.linalg.svd(LL, full_matrices=False)
        Uw, Sw, Vtw = np.linalg.svd(W, full_matrices=False)

        # Embed by modifying singular values
        S_prime = Sc + alpha * Sw
        LL_prime = (Uc @ np.diag(S_prime) @ Vct).astype(np.float32)

        # Inverse DWT to get watermarked cover channel
        Cw = pywt.idwt2((LL_prime, (LH, HL, HH)), 'haar')
        watermarked_channels.append(Cw)
        
        # Store side info for this channel
        side_info_channels.append({
            'S_cover': Sc.astype(np.float32),
            'Uw': Uw.astype(np.float32),
            'Vtw': Vtw.astype(np.float32),
            'alpha': float(alpha),
            'cover_shape': C_channel.shape,
            'wm_shape': W.shape,
        })
    
    # Combine RGB channels
    watermarked_array = np.stack(watermarked_channels, axis=2)
    watermarked = array_to_pil_rgb(watermarked_array)

    # Persist exact shapes used for reconstruction
    side = WatermarkSideInfo(
        S_cover=np.array([ch['S_cover'] for ch in side_info_channels]),
        Uw=np.array([ch['Uw'] for ch in side_info_channels]),
        Vtw=np.array([ch['Vtw'] for ch in side_info_channels]),
        alpha=float(alpha),
        cover_shape=C.shape,   # (H, W, 3) of cover in RGB
        wm_shape=W_rgb.shape,  # (H, W, 3) of original watermark in RGB
    )
    return watermarked, side


def dwt_svd_extract(wm_image: Image.Image, side: WatermarkSideInfo) -> Image.Image:
    # Convert watermarked image to RGB array
    Cw = pil_to_array_rgb(wm_image).astype(np.float32)  # Shape: (H, W, 3)
    
    # Process each RGB channel separately to extract color watermark
    extracted_channels = []
    
    for channel in range(3):  # R, G, B channels
        Cw_channel = Cw[:, :, channel]
        
        # 1-level DWT on watermarked image channel
        LLw, _ = pywt.dwt2(Cw_channel, 'haar')
        
        # SVD on LL of watermarked
        Ucw, Scw, Vtcw = np.linalg.svd(LLw.astype(np.float32), full_matrices=False)

        # Get side info for this channel
        Sc = side.S_cover[channel]
        Uw = side.Uw[channel]
        Vtw = side.Vtw[channel]

        # Align dimensions robustly
        len_scw = Scw.shape[0]
        len_sc = Sc.shape[0]
        uw_cols = Uw.shape[1]
        vtw_rows = Vtw.shape[0]
        k = min(len_scw, len_sc, uw_cols, vtw_rows)
        if k <= 0:
            raise ValueError("Invalid SVD dimensions for extraction")

        Scw_k = Scw[:k]
        Sc_k = Sc[:k]
        Uw_k = Uw[:, :k]
        Vtw_k = Vtw[:k, :]

        # Recover watermark singular values and reconstruct estimated watermark LL
        Sw_est = (Scw_k - Sc_k) / max(side.alpha, 1e-8)
        W_est = (Uw_k @ np.diag(Sw_est) @ Vtw_k)

        # Clip to valid intensity
        W_est = np.clip(W_est, 0, 255).astype(np.uint8)
        extracted_channels.append(W_est)
    
    # Combine RGB channels to preserve color information
    # No averaging - keep each channel separate for RGB reconstruction
    extracted_rgb = np.stack(extracted_channels, axis=2)
    
    # Resize back to the original watermark shape (RGB)
    img = Image.fromarray(extracted_rgb, mode="RGB").resize((side.wm_shape[1], side.wm_shape[0]), Image.NEAREST)
    return img


def sideinfo_to_npz_bytes(side: WatermarkSideInfo) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        S_cover=side.S_cover,
        Uw=side.Uw,
        Vtw=side.Vtw,
        alpha=np.array([side.alpha], dtype=np.float32),
        cover_shape=np.array(side.cover_shape, dtype=np.int32),
        wm_shape=np.array(side.wm_shape, dtype=np.int32),  # Now supports RGB shape (H,W,3)
    )
    return buf.getvalue()


def npz_bytes_to_sideinfo(b: bytes) -> WatermarkSideInfo:
    sio = io.BytesIO(b)
    npz = np.load(sio, allow_pickle=False)
    return WatermarkSideInfo(
        S_cover=npz["S_cover"],
        Uw=npz["Uw"],
        Vtw=npz["Vtw"],
        alpha=float(npz["alpha"][0]),
        cover_shape=tuple(npz["cover_shape"]),
        wm_shape=tuple(npz["wm_shape"]),
    )


# -------------------- Evaluation Helpers --------------------


def compute_mse(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.mean((a - b) ** 2))


def compute_psnr(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    mse = compute_mse(a, b)
    if mse <= 1e-12:
        return float('inf')
    return 20.0 * np.log10(max_val) - 10.0 * np.log10(mse)


def compute_ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def compute_ber(original_bits: np.ndarray, extracted_bits: np.ndarray) -> float:
    """Menghitung Bit Error Rate"""
    if original_bits.size != extracted_bits.size:
        return 1.0
    errors = np.sum(original_bits != extracted_bits)
    return float(errors / original_bits.size)


def compute_uaci(img1: np.ndarray, img2: np.ndarray) -> float:
    """Mengukur perubahan intensitas rata-rata"""
    diff = np.abs(img1.astype(float) - img2.astype(float))
    return float(np.mean(diff) / 255.0 * 100)


def compute_npcr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Mengukur persentase pixel yang berubah"""
    diff = img1 != img2
    return float(np.sum(diff) / img1.size * 100)


def compute_entropy(img: np.ndarray) -> float:
    """Menghitung entropi informasi gambar"""
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
    hist = hist / hist.sum()
    hist = hist[hist > 0]  # Remove zero probabilities
    return float(-np.sum(hist * np.log2(hist)))


def make_histogram_df(arr: np.ndarray, bins: int = 64) -> 'pd.DataFrame':  # Reduced bins for smaller chart
    hist, edges = np.histogram(arr.ravel(), bins=bins, range=(0, 255))
    centers = (edges[:-1] + edges[1:]) / 2.0
    return pd.DataFrame({"intensity": centers.astype(np.float32), "count": hist.astype(np.int64)})


# -------------------- ECIES (ECC + AES-GCM) --------------------


def generate_ecc_keypair() -> Tuple[bytes, bytes]:
    priv = ec.generate_private_key(ec.SECP256R1())
    pub = priv.public_key()
    priv_pem = priv.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    pub_pem = pub.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    return priv_pem, pub_pem


def load_private_key(pem: bytes):
    return load_pem_private_key(pem, password=None)


def load_public_key(pem: bytes):
    return load_pem_public_key(pem)


def ecies_encrypt(plaintext: bytes, recipient_pub_pem: bytes) -> Dict[str, Any]:
    recipient_pub = load_public_key(recipient_pub_pem)
    eph_priv = ec.generate_private_key(ec.SECP256R1())
    eph_pub = eph_priv.public_key()

    shared = eph_priv.exchange(ec.ECDH(), recipient_pub)
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"ECIES-P256-AESGCM").derive(shared)

    aesgcm = AESGCM(key)
    nonce = secrets.token_bytes(12)
    ct = aesgcm.encrypt(nonce, plaintext, associated_data=None)

    eph_pub_bytes = eph_pub.public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)
    return {
        "ephemeral_pub": b64e(eph_pub_bytes),
        "nonce": b64e(nonce),
        "ciphertext": b64e(ct),
    }


def ecies_decrypt(payload: Dict[str, Any], recipient_priv_pem: bytes) -> bytes:
    priv = load_private_key(recipient_priv_pem)
    eph_pub_bytes = b64d(payload["ephemeral_pub"]) 
    eph_pub = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), eph_pub_bytes)
    shared = priv.exchange(ec.ECDH(), eph_pub)
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"ECIES-P256-AESGCM").derive(shared)
    aesgcm = AESGCM(key)
    nonce = b64d(payload["nonce"]) 
    ct = b64d(payload["ciphertext"]) 
    pt = aesgcm.decrypt(nonce, ct, associated_data=None)
    return pt


# -------------------- Streamlit UI --------------------


st.set_page_config(page_title="ECIES + DWT–SVD Watermarking (Single Flow)", layout="wide")

# Custom CSS untuk memperkecil elemen UI dan mengurangi padding
st.markdown("""
<style>
    /* Mengurangi padding atas dan spacing */
    .main .block-container {
        padding-top: 0.1rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }
    
    /* PERBAIKAN: Pastikan konten menempel ke atas */
    .main .block-container > div {
        padding-top: 0 !important;
    }
    
    /* PERBAIKAN: Hapus margin atas dari elemen pertama */
    .main .block-container > div:first-child {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* Mengurangi margin antar elemen */
    .stMarkdown {
        margin-bottom: 0.25rem !important;
    }
    
    /* Mengurangi spacing di sidebar */
    .css-1d391kg {
        padding-top: 0.5rem !important;
    }
    
    /* Memperkecil elemen UI */
    .stFileUploader > div > div > div > div {
        height: 35px !important;
    }
    .stFileUploader > div > div > div > div > div {
        font-size: 11px !important;
    }
    .stButton > button {
        height: 30px !important;
        font-size: 11px !important;
    }
    .stDownloadButton > button {
        height: 25px !important;
        font-size: 10px !important;
    }
    .stImage > img {
        max-width: 80px !important;
        height: auto !important;
    }
    
    /* Memperkecil metrics dan evaluasi */
    .stMetric {
        font-size: 10px !important;
        margin-bottom: 0.25rem !important;
    }
    .stMetric > div > div {
        font-size: 10px !important;
    }
    .stMetric > div > div > div {
        font-size: 9px !important;
    }
    
    .stCaption {
        font-size: 8px !important;
        margin-bottom: 0.1rem !important;
    }
    
    /* Mengurangi spacing antar kolom */
    .stColumn {
        padding-left: 0.25rem !important;
        padding-right: 0.25rem !important;
    }
    
    /* Mengurangi margin header */
    h1, h2, h3, h4, h5, h6 {
        margin-top: 0 !important;
        margin-bottom: 0.25rem !important;
        font-size: 1.1rem !important;
    }
    
    /* PERBAIKAN: Hapus margin atas dari semua elemen */
    .main .block-container > div > div {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* PERBAIKAN KHUSUS UNTUK STREAMLIT CLOUD */
    .main .block-container {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* Hapus spacing dari elemen pertama di main */
    .main > div:first-child {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* Hapus spacing dari stMarkdown pertama */
    .main .stMarkdown:first-child {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* Memperkecil chart dan histogram */
    .vega-embed {
        font-size: 8px !important;
    }
    .vega-embed summary {
        font-size: 8px !important;
    }
    
    /* Mengurangi spacing di evaluasi section */
    .element-container {
        margin-bottom: 0.1rem !important;
    }
    
    /* Memperkecil text di metrics */
    [data-testid="metric-container"] {
        padding: 0.1rem !important;
    }
    [data-testid="metric-container"] > div {
        font-size: 9px !important;
    }
</style>
""", unsafe_allow_html=True)

if "ecc_priv_pem" not in st.session_state:
    priv_pem, pub_pem = generate_ecc_keypair()
    st.session_state.ecc_priv_pem = priv_pem
    st.session_state.ecc_pub_pem = pub_pem

if "sideinfo" not in st.session_state:
    st.session_state.sideinfo = None
if "encrypted_payload" not in st.session_state:
    st.session_state.encrypted_payload = None
if "last_package_bytes" not in st.session_state:
    st.session_state.last_package_bytes = None
if "watermarked_img" not in st.session_state:
    st.session_state.watermarked_img = None
if "cover_rgb" not in st.session_state:
    st.session_state.cover_rgb = None
if "wm_input_rgb" not in st.session_state:
    st.session_state.wm_input_rgb = None

# Tombol Reset Semua
def _reset_all_state():
    # Clear Streamlit caches (if any) and wipe all session_state keys
    try:
        st.cache_data.clear()
    except Exception:
        pass
    try:
        st.cache_resource.clear()
    except Exception:
        pass
    keys = list(st.session_state.keys())
    for k in keys:
        del st.session_state[k]
    # Rerun will re-trigger key generation at startup
    st.rerun()

with st.sidebar:
    alpha = st.slider("Alpha (kekuatan watermark)",  0.01, 0.50, 0.10, 0.01)
    st.divider()
    st.subheader("ECC Keys")
    if st.button("Generate New ECC Keypair"):
        priv_pem, pub_pem = generate_ecc_keypair()
        st.session_state.ecc_priv_pem = priv_pem
        st.session_state.ecc_pub_pem = pub_pem
        st.success("Keypair baru dibuat.")

    st.download_button("Download ECC Public Key (.pem)", data=st.session_state.ecc_pub_pem, file_name="ecc_public.pem", mime="application/x-pem-file")
    st.download_button("Download ECC Private Key (.pem)", data=st.session_state.ecc_priv_pem, file_name="ecc_private.pem", mime="application/x-pem-file")

    st.divider()
    st.subheader("Kunci Pihak Lain")
    up_pub = st.file_uploader("Upload Recipient Public Key (PEM) untuk Enkripsi", type=["pem"], key="upl_pub")
    if up_pub is not None:
        try:
            new_pub = up_pub.read()
            st.session_state.ecc_pub_pem = new_pub
            st.success("Public key diupdate dari upload (menimpa current).")
        except Exception as e:
            st.error(f"Gagal memuat public key: {e}")
    up_priv = st.file_uploader("Upload Recipient Private Key (PEM) untuk Dekripsi", type=["pem"], key="upl_priv")
    if up_priv is not None:
        try:
            new_priv = up_priv.read()
            st.session_state.ecc_priv_pem = new_priv
            st.success("Private key diupdate dari upload (menimpa current).")
        except Exception as e:
            st.error(f"Gagal memuat private key: {e}")

    st.divider()
    if st.button("Reset Semua"):
        _reset_all_state()
    
    # Keys display in sidebar
    st.divider()
    with st.expander("🔑 ECC Keys"):
        st.caption("Public Key (PEM)")
        try:
            st.code(st.session_state.ecc_pub_pem.decode("utf-8")[:200] + "...", language="text")
        except Exception:
            st.code("<invalid public key bytes>")
        st.caption("Private Key (PEM)")
        try:
            st.code(st.session_state.ecc_priv_pem.decode("utf-8")[:200] + "...", language="text")
        except Exception:
            st.code("<invalid private key bytes>")

## -------------------- Main Grid Layout --------------------
# Judul kecil di tengah - HARUS SEBELUM st.columns()
st.markdown("<div style='text-align: center; margin-bottom: 1rem;'><h3 style='color: #666; font-size: 1.2rem; margin: 0;'>🔐 ECIES + DWT-SVD Watermarking System</h3></div>", unsafe_allow_html=True)

# Create 1:3 layout (enkripsi:dekripsi:evaluasi)
col_encrypt, col_decrypt, col_eval = st.columns([1, 1, 1])

with col_encrypt:
    # Enkripsi section
    st.markdown("#### 🔐 ENKRIPSI")
    
    # Input gambar - diperkecil dan sebelahan
    col_upload1, col_upload2 = st.columns(2)
    with col_upload1:
        cover_file = st.file_uploader("📁 Cover", type=["png", "jpg", "jpeg", "bmp"], key="cover_upl")
    with col_upload2:
        wm_file = st.file_uploader("🏷️ WM", type=["png", "jpg", "jpeg", "bmp"], key="wm_upl")

    # Preview gambar kecil dengan wadah diperkecil
    if cover_file is not None or wm_file is not None:
        col_preview1, col_preview2 = st.columns(2)
        with col_preview1:
            if cover_file is not None:
                try:
                    st.image(Image.open(cover_file).convert("RGB"), caption="Cover", width=100)
                except Exception:
                    st.error("Invalid")
        with col_preview2:
            if wm_file is not None:
                try:
                    st.image(Image.open(wm_file).convert("RGB"), caption="WM", width=100)
                except Exception:
                    st.error("Invalid")

    # Embed watermark dengan tampilan hasil di sebelah tombol
    embed_disabled = not (cover_file and wm_file)
    col_button, col_result = st.columns([1, 1])
    
    with col_button:
        if st.button("🔗 Embed", disabled=embed_disabled, type="primary"):
            try:
                cover_img = Image.open(cover_file).convert("RGB")
                wm_img = Image.open(wm_file).convert("RGB")
                wm_out, side = dwt_svd_embed(cover_img, wm_img, alpha=alpha)
                st.session_state.watermarked_img = wm_out
                st.session_state.sideinfo = side
                st.session_state.cover_rgb = pil_to_array_rgb(cover_img)
                st.session_state.wm_input_rgb = pil_to_array_rgb(wm_img)
                
                # Auto encrypt dan download JSON
                try:
                    recipient_pub_pem = st.session_state.ecc_pub_pem
                    watermarked_bytes = bytes_from_image(wm_out, "PNG")
                    side_bytes = sideinfo_to_npz_bytes(side)
                    package = {
                        "image_png_b64": b64e(watermarked_bytes),
                        "sideinfo_npz_b64": b64e(side_bytes),
                    }
                    package_bytes = json.dumps(package).encode("utf-8")
                    enc = ecies_encrypt(package_bytes, recipient_pub_pem)
                    enc_json = json.dumps(enc, indent=2).encode("utf-8")
                    st.session_state.encrypted_payload = enc
                    st.session_state.last_package_bytes = package_bytes
                    st.session_state.enc_json = enc_json
                    
                    # Auto download JSON
                    st.download_button(
                        "📥 Download JSON",
                        data=enc_json,
                        file_name="package.enc.json",
                        mime="application/json",
                        key="auto_download_json"
                    )
                except Exception as e:
                    st.error(f"Encrypt error: {e}")
                    
            except Exception as e:
                st.error(f"Embed error: {e}")
    
    with col_result:
        # Tampilkan hasil watermarked di sebelah tombol
        if st.session_state.watermarked_img is not None:
            st.image(st.session_state.watermarked_img, caption="Result", width=100)

with col_decrypt:
    # Dekripsi section
    st.markdown("#### 🔓 DEKRIPSI")
    
    enc_file = st.file_uploader("📁 JSON", type=["json"], key="enc_pkg_upl_new")
    
    if st.button("🔓 Decrypt", disabled=(enc_file is None), type="secondary"):
        try:
            enc_payload = None
            if enc_file is not None:
                enc_payload = json.loads(enc_file.read().decode("utf-8"))
            if enc_payload is None:
                st.error("Mohon upload file JSON terenkripsi.")
            else:
                recipient_priv_pem = st.session_state.ecc_priv_pem
                pt = ecies_decrypt(enc_payload, recipient_priv_pem)
                package = json.loads(pt.decode("utf-8"))

                img_b = b64d(package["image_png_b64"]) 
                side_b = b64d(package["sideinfo_npz_b64"]) 
                dec_wm_img = image_from_bytes_rgb(img_b)

                side = None
                try:
                    side = npz_bytes_to_sideinfo(side_b)
                    expected_size = (side.cover_shape[1], side.cover_shape[0])  # (W, H) from (H, W, 3)
                    if dec_wm_img.size != expected_size:
                        dec_wm_img = dec_wm_img.resize(expected_size, Image.BICUBIC)
                except Exception as e:
                    st.error(f"Side-info tidak valid: {e}")

                st.session_state.decrypted_img = dec_wm_img
                st.session_state.decrypted_side = side

                if st.session_state.decrypted_side is not None:
                    try:
                        extracted = dwt_svd_extract(dec_wm_img, st.session_state.decrypted_side)
                        st.session_state.extracted_wm = extracted
                    except Exception as e:
                        st.error(f"Extract error: {e}")
        except Exception as e:
            st.error(f"Decrypt error: {e}")

    # Tampilkan gambar hasil dekripsi dengan wadah kecil
    if "decrypted_img" in st.session_state and st.session_state.decrypted_img is not None:
        col_result1, col_result2 = st.columns(2)
        with col_result1:
            st.image(st.session_state.decrypted_img, caption="Cover", width=100)
        with col_result2:
            if "extracted_wm" in st.session_state and st.session_state.extracted_wm is not None:
                st.image(st.session_state.extracted_wm, caption="WM", width=100)
                st.download_button("📥 WM", data=bytes_from_image(st.session_state.extracted_wm, "PNG"), file_name="extracted_watermark.png", mime="image/png")

with col_eval:
    # Evaluasi section - Compact layout
    st.markdown("#### 📊 EVALUASI")
    
    if "decrypted_img" in st.session_state and st.session_state.decrypted_img is not None:
        cover_rgb = st.session_state.cover_rgb
        dec_rgb = pil_to_array_rgb(st.session_state.decrypted_img)
        
        if cover_rgb is None:
            st.caption("Cover tidak tersedia")
        else:
            # Compute all metrics
            psnr_channels = []
            mse_channels = []
            uaci_channels = []
            npcr_channels = []
            entropy_cover_channels = []
            entropy_dec_channels = []
            
            for channel in range(3):
                psnr_val = compute_psnr(cover_rgb[:, :, channel], dec_rgb[:, :, channel])
                mse_val = compute_mse(cover_rgb[:, :, channel], dec_rgb[:, :, channel])
                uaci_val = compute_uaci(cover_rgb[:, :, channel], dec_rgb[:, :, channel])
                npcr_val = compute_npcr(cover_rgb[:, :, channel], dec_rgb[:, :, channel])
                entropy_cov = compute_entropy(cover_rgb[:, :, channel])
                entropy_dec = compute_entropy(dec_rgb[:, :, channel])
                
                psnr_channels.append(psnr_val)
                mse_channels.append(mse_val)
                uaci_channels.append(uaci_val)
                npcr_channels.append(npcr_val)
                entropy_cover_channels.append(entropy_cov)
                entropy_dec_channels.append(entropy_dec)
            
            # Average values
            avg_psnr = np.mean(psnr_channels)
            avg_mse = np.mean(mse_channels)
            avg_uaci = np.mean(uaci_channels)
            avg_npcr = np.mean(npcr_channels)
            avg_entropy_cover = np.mean(entropy_cover_channels)
            avg_entropy_dec = np.mean(entropy_dec_channels)
            
            # Compact metrics display - 2 columns
            col_met1, col_met2 = st.columns(2)
            with col_met1:
                st.metric("PSNR", f"{avg_psnr:.1f}", help="dB")
                st.metric("MSE", f"{avg_mse:.1f}", help="Lower better")
                st.metric("UACI", f"{avg_uaci:.2f}%", help="Change intensity")
            with col_met2:
                st.metric("NPCR", f"{avg_npcr:.2f}%", help="Pixel change rate")
                st.metric("ENT-C", f"{avg_entropy_cover:.2f}", help="Cover entropy")
                st.metric("ENT-D", f"{avg_entropy_dec:.2f}", help="Dec entropy")

        # NCC dan BER jika watermark tersedia
        if st.session_state.get("extracted_wm") is not None and st.session_state.wm_input_rgb is not None:
            # Resize original watermark to match extracted watermark size
            wm_ref = Image.fromarray(st.session_state.wm_input_rgb.astype(np.uint8), mode="RGB").resize(st.session_state.extracted_wm.size, Image.BICUBIC)
            
            # Compute NCC and BER for each RGB channel
            ncc_channels = []
            ber_channels = []
            
            for channel in range(3):
                wm_ref_ch = np.array(wm_ref, dtype=np.float32)[:, :, channel]
                wm_ext_ch = np.array(st.session_state.extracted_wm, dtype=np.float32)[:, :, channel]
                
                ncc_val = compute_ncc(wm_ref_ch, wm_ext_ch)
                ncc_channels.append(ncc_val)
                
                # Convert to binary for BER calculation
                wm_ref_binary = (wm_ref_ch > 127).astype(np.uint8)
                wm_ext_binary = (wm_ext_ch > 127).astype(np.uint8)
                ber_val = compute_ber(wm_ref_binary, wm_ext_binary)
                ber_channels.append(ber_val)
            
            avg_ncc = np.mean(ncc_channels)
            avg_ber = np.mean(ber_channels)
            
            # Watermark metrics in compact layout
            col_wm1, col_wm2 = st.columns(2)
            with col_wm1:
                st.metric("NCC", f"{avg_ncc:.3f}", help="Correlation")
            with col_wm2:
                st.metric("BER", f"{avg_ber:.3f}", help="Bit error rate")
            
            # Channel details in small text
            st.caption(f"NCC: R:{ncc_channels[0]:.3f} G:{ncc_channels[1]:.3f} B:{ncc_channels[2]:.3f}")
            st.caption(f"BER: R:{ber_channels[0]:.3f} G:{ber_channels[1]:.3f} B:{ber_channels[2]:.3f}")
        
        # Compact histogram - smaller and simpler
        if cover_rgb is not None:
            st.caption("📈 Histogram")
            # Use only one channel (G) for compactness
            df_cover_g = make_histogram_df(cover_rgb[:, :, 1])  # Green channel
            df_wm_g = make_histogram_df(dec_rgb[:, :, 1])
            df_cover_g["type"] = "Cover"
            df_wm_g["type"] = "Watermarked"
            
            df_hist = pd.concat([df_cover_g, df_wm_g], ignore_index=True)
            chart = alt.Chart(df_hist).mark_line(opacity=0.8, strokeWidth=1).encode(
                x=alt.X("intensity:Q", title="", axis=alt.Axis(labels=False)),
                y=alt.Y("count:Q", title="", axis=alt.Axis(labels=False)),
                color=alt.Color("type:N", scale=alt.Scale(domain=["Cover", "Watermarked"], 
                                                         range=["#00ff00", "#ff6666"]))
            ).properties(width=180, height=60)
            st.altair_chart(chart, use_container_width=True)

# Keys display moved to sidebar
