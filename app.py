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


def bytes_from_image(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def image_from_bytes(b: bytes) -> Image.Image:
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
    S_cover: np.ndarray
    Uw: np.ndarray
    Vtw: np.ndarray
    alpha: float
    cover_shape: Tuple[int, int]
    wm_shape: Tuple[int, int]


def dwt_svd_embed(cover_img: Image.Image, wm_img: Image.Image, alpha: float = 0.05):
    C = pil_to_array_gray(cover_img)
    LL, (LH, HL, HH) = pywt.dwt2(C, 'haar')

    LL_h, LL_w = LL.shape
    wm_resized = wm_img.convert("L").resize((LL_w, LL_h), Image.BICUBIC)
    W = np.array(wm_resized, dtype=np.float32)

    Uc, Sc, Vct = np.linalg.svd(LL, full_matrices=False)
    Uw, Sw, Vtw = np.linalg.svd(W, full_matrices=False)

    S_prime = Sc + alpha * Sw
    LL_prime = (Uc @ np.diag(S_prime) @ Vct).astype(np.float32)

    Cw = pywt.idwt2((LL_prime, (LH, HL, HH)), 'haar')
    watermarked = array_to_pil_gray(Cw)

    side = WatermarkSideInfo(
        S_cover=Sc.astype(np.float32),
        Uw=Uw.astype(np.float32),
        Vtw=Vtw.astype(np.float32),
        alpha=float(alpha),
        cover_shape=C.shape,
        wm_shape=W.shape,
    )
    return watermarked, side


def dwt_svd_extract(wm_image: Image.Image, side: WatermarkSideInfo) -> Image.Image:
    Cw = pil_to_array_gray(wm_image).astype(np.float32)
    LLw, _ = pywt.dwt2(Cw, 'haar')
    Ucw, Scw, Vtcw = np.linalg.svd(LLw.astype(np.float32), full_matrices=False)

    # Robust alignment in case of slight size/rounding mismatches
    len_scw = Scw.shape[0]
    len_sc = side.S_cover.shape[0]
    uw_cols = side.Uw.shape[1]
    vtw_rows = side.Vtw.shape[0]
    k = min(len_scw, len_sc, uw_cols, vtw_rows)
    if k <= 0:
        raise ValueError("Invalid SVD dimensions for extraction")

    Scw_k = Scw[:k]
    Sc_k = side.S_cover[:k]
    Uw_k = side.Uw[:, :k]
    Vtw_k = side.Vtw[:k, :]

    Sw_est = (Scw_k - Sc_k) / max(side.alpha, 1e-8)
    W_est = (Uw_k @ np.diag(Sw_est) @ Vtw_k)
    W_est = np.clip(W_est, 0, 255).astype(np.uint8)

    img = Image.fromarray(W_est, mode="L").resize((side.wm_shape[1], side.wm_shape[0]), Image.NEAREST)
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
        wm_shape=np.array(side.wm_shape, dtype=np.int32),
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


st.set_page_config(page_title="ECIES + DWT–SVD Watermarking (Single Page)", layout="wide")

if "ecc_priv_pem" not in st.session_state:
    priv_pem, pub_pem = generate_ecc_keypair()
    st.session_state.ecc_priv_pem = priv_pem
    st.session_state.ecc_pub_pem = pub_pem

if "sideinfo" not in st.session_state:
    st.session_state.sideinfo = None

st.title("Implementasi ECIES (ECC) + Watermarking DWT–SVD")
st.caption("Single-page: embed watermark (DWT–SVD), enkripsi ECIES, dekripsi, dan ekstraksi.")

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
    st.header("Input")
    cover_file = st.file_uploader("Cover Image (host)", type=["png", "jpg", "jpeg", "bmp"])
    wm_file = st.file_uploader("Watermark Image (logo/grayscale)", type=["png", "jpg", "jpeg", "bmp"])
    alpha = st.slider("Alpha (kekuatan watermark)",  0.01, 0.50, 0.30, 0.01)
    st.divider()
    st.subheader("ECC Keys")
    if st.button("Generate New ECC Keypair"):
        priv_pem, pub_pem = generate_ecc_keypair()
        st.session_state.ecc_priv_pem = priv_pem
        st.session_state.ecc_pub_pem = pub_pem
        st.success("Keypair baru dibuat.")

    # Download ECC keys as proper PEM files
    st.download_button("Download ECC Public Key (.pem)", data=st.session_state.ecc_pub_pem, file_name="ecc_public.pem", mime="application/x-pem-file")
    st.download_button("Download ECC Private Key (.pem)", data=st.session_state.ecc_priv_pem, file_name="ecc_private.pem", mime="application/x-pem-file")

    # (Paste PEM inputs removed as requested)

    st.divider()
    st.subheader("Kunci Pihak Lain")
    up_pub = st.file_uploader("Upload Recipient Public Key (PEM) untuk Enkripsi", type=["pem"])
    if up_pub is not None:
        try:
            new_pub = up_pub.read()
            st.session_state.ecc_pub_pem = new_pub
            st.success("Public key diupdate dari upload (menimpa current).")
        except Exception as e:
            st.error(f"Gagal memuat public key: {e}")
    up_priv = st.file_uploader("Upload Recipient Private Key (PEM) untuk Dekripsi", type=["pem"])
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

col1, col2 = st.columns(2)

with col1:
    st.subheader("1) Embedding Watermark")
    if cover_file and wm_file:
        cover_img = Image.open(cover_file).convert("RGB")
        wm_img = Image.open(wm_file).convert("RGB")

        st_image_compat(cover_img, caption="Cover Image")
        st_image_compat(wm_img, caption="Watermark Image")

        if st.button("Embed Watermark (DWT–SVD)"):
            wm_out, side = dwt_svd_embed(cover_img, wm_img, alpha=alpha)
            st.session_state.watermarked_img = wm_out
            st.session_state.sideinfo = side
            st.success("Watermark embedded.")

    if "watermarked_img" in st.session_state:
        st_image_compat(st.session_state.watermarked_img, caption="Watermarked (Grayscale)")

        st.download_button(
            "Download Watermarked Image (PNG)",
            data=bytes_from_image(st.session_state.watermarked_img, "PNG"),
            file_name="watermarked.png",
            mime="image/png",
        )

with col2:
    st.subheader("2) Enkripsi Hasil (ECIES)")
    if "watermarked_img" in st.session_state and st.session_state.sideinfo is not None:
        recipient_pub_pem = st.session_state.ecc_pub_pem

        watermarked_bytes = bytes_from_image(st.session_state.watermarked_img, "PNG")
        side_bytes = sideinfo_to_npz_bytes(st.session_state.sideinfo)
        package = {
            "image_png_b64": b64e(watermarked_bytes),
            "sideinfo_npz_b64": b64e(side_bytes),
        }
        package_bytes = json.dumps(package).encode("utf-8")

        if st.button("Encrypt Package (Image + SideInfo)"):
            enc = ecies_encrypt(package_bytes, recipient_pub_pem)
            enc_json = json.dumps(enc, indent=2).encode("utf-8")
            st.download_button("Download Encrypted Package (JSON)", data=enc_json, file_name="package.enc.json", mime="application/json")
            st.success("Package terenkripsi siap diunduh.")

    st.subheader("3) Dekripsi & Ekstraksi")
    enc_file = st.file_uploader("Upload Encrypted Package (JSON)", type=["json"], key="enc_upl")
    if enc_file is not None:
        try:
            enc_payload = json.loads(enc_file.read().decode("utf-8"))
        except Exception as e:
            st.error(f"File terenkripsi tidak valid: {e}")
            enc_payload = None

        if enc_payload is not None:
            recipient_priv_pem = st.session_state.ecc_priv_pem

            if st.button("Decrypt Package"):
                try:
                    pt = ecies_decrypt(enc_payload, recipient_priv_pem)
                    package = json.loads(pt.decode("utf-8"))

                    img_b = b64d(package["image_png_b64"]) 
                    side_b = b64d(package["sideinfo_npz_b64"]) 
                    wm_img = image_from_bytes(img_b).convert("L")

                    # Pastikan dimensi konsisten untuk ekstraksi
                    try:
                        side = npz_bytes_to_sideinfo(side_b)
                        # Pastikan ukuran gambar sama dengan cover_shape yang tersimpan
                        expected_size = (side.cover_shape[1], side.cover_shape[0])
                        if wm_img.size != expected_size:
                            wm_img = wm_img.resize(expected_size, Image.BICUBIC)

                        # Simpan ke session state
                        st.session_state.decrypted_img = wm_img
                        st.session_state.decrypted_side = side
                        st.info(f"Ukuran decrypted: {wm_img.size}, cover_shape: {expected_size}, alpha: {side.alpha}")
                        st.success("Dekripsi berhasil! Silakan klik tombol Extract di bawah.")
                    except Exception as e:
                        st.error(f"Side-info tidak valid: {e}")
                        st.session_state.decrypted_img = wm_img
                        st.session_state.decrypted_side = None

                except Exception as e:
                    st.error(f"Dekripsi gagal: {e}")

    # Tampilkan hasil dekripsi jika ada
    if "decrypted_img" in st.session_state:
        st_image_compat(st.session_state.decrypted_img, caption="Decrypted Watermarked Image")
        
        if "decrypted_side" in st.session_state and st.session_state.decrypted_side is not None:
            if st.button("Extract Watermark (from Decrypted Image)"):
                try:
                    extracted = dwt_svd_extract(st.session_state.decrypted_img, st.session_state.decrypted_side)
                    st.session_state.extracted_wm = extracted
                    st.success("Ekstraksi watermark berhasil!")
                except Exception as e:
                    st.error(f"Ekstraksi gagal: {e}")
            
            # Tampilkan watermark yang diekstrak jika ada
            if "extracted_wm" in st.session_state:
                st_image_compat(st.session_state.extracted_wm, caption="Extracted Watermark (Grayscale)")
                st.download_button("Download Extracted Watermark (PNG)", data=bytes_from_image(st.session_state.extracted_wm, "PNG"), file_name="extracted_watermark.png", mime="image/png")

# -------------------- Keys Display (Bottom) --------------------
st.divider()
st.subheader("ECC Keys (Current Values)")
col_k1, col_k2 = st.columns(2)
with col_k1:
    st.caption("Public Key (PEM)")
    try:
        st.code(st.session_state.ecc_pub_pem.decode("utf-8"))
    except Exception:
        st.code("<invalid public key bytes>")
with col_k2:
    st.caption("Private Key (PEM)")
    try:
        st.code(st.session_state.ecc_priv_pem.decode("utf-8"))
    except Exception:
        st.code("<invalid private key bytes>")
