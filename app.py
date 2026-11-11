import io
import time

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="Binary Image Steganography Workbench",
    layout="wide"
)

HEADER_BITS = 32
BLOCK_SIZE = 2

METHODS = [
    "Pixel Flipping",
    "Block Parity",
    "Pattern Substitution"
]


# ============================================================
# IMAGE HELPERS
# ============================================================

def to_binary_image(uploaded_file, threshold=128):
    """
    Convert uploaded image to RGB for display and binary 0/1 array
    for all steganography algorithms.
    """

    img = Image.open(uploaded_file).convert("RGB")

    gray = img.convert("L")
    gray_arr = np.array(gray)

    binary = (gray_arr >= threshold).astype(np.uint8)

    binary_img = Image.fromarray(
        binary * 255,
        mode="L"
    )

    return img, binary_img, binary


def binary_image_from_stego(uploaded_file):
    """
    Stego images are saved as 0/255 PNGs.
    Therefore extraction always uses threshold 128.
    """

    img = Image.open(uploaded_file).convert("L")

    arr = np.array(img)

    binary = (arr >= 128).astype(np.uint8)

    return binary


def image_bytes(arr):
    """
    Convert binary 0/1 NumPy array to downloadable PNG bytes.
    """

    img = Image.fromarray(
        (arr * 255).astype(np.uint8),
        mode="L"
    )

    buffer = io.BytesIO()

    img.save(
        buffer,
        format="PNG"
    )

    return buffer.getvalue()


# ============================================================
# MESSAGE <-> BITS
# ============================================================

def bytes_to_bits(data):
    """
    Convert bytes to an array of 0/1 bits.
    """

    if not data:
        return np.array([], dtype=np.uint8)

    return np.unpackbits(
        np.frombuffer(data, dtype=np.uint8)
    ).astype(np.uint8)


def bits_to_bytes(bits):
    """
    Convert 0/1 bits back to bytes.
    """

    bits = np.asarray(bits, dtype=np.uint8)

    usable = (len(bits) // 8) * 8

    if usable == 0:
        return b""

    return np.packbits(
        bits[:usable]
    ).tobytes()


def make_payload(message):
    """
    Payload format:

    32-bit header = UTF-8 message length in bytes
    Remaining bits = actual message
    """

    raw = message.encode("utf-8")

    header = len(raw).to_bytes(
        HEADER_BITS // 8,
        byteorder="big"
    )

    return bytes_to_bits(
        header + raw
    )


def read_payload(bits):
    """
    Read the 32-bit length header and recover the message.
    """

    bits = np.asarray(bits, dtype=np.uint8)

    if len(bits) < HEADER_BITS:
        raise ValueError(
            "Not enough data to contain the message header."
        )

    header_bytes = bits_to_bytes(
        bits[:HEADER_BITS]
    )

    message_length = int.from_bytes(
        header_bytes,
        byteorder="big"
    )

    required_bits = HEADER_BITS + message_length * 8

    if required_bits > len(bits):
        raise ValueError(
            "Incomplete or invalid hidden message."
        )

    message_bits = bits[
        HEADER_BITS:required_bits
    ]

    try:
        return bits_to_bytes(
            message_bits
        ).decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(
            "Hidden data is not valid UTF-8. "
            "The image may not belong to this technique."
        )


# ============================================================
# METRICS
# ============================================================

def changed_pixels(original, stego):
    """
    Return:
    - number of changed pixels
    - percentage of changed pixels
    """

    diff = original != stego

    count = int(diff.sum())

    percentage = float(
        diff.mean() * 100
    )

    return count, percentage


def psnr(original, stego):
    """
    PSNR for binary images.

    Binary values are 0 and 1, so MAX_I = 1.
    """

    original = original.astype(np.float32)
    stego = stego.astype(np.float32)

    mse = np.mean(
        (original - stego) ** 2
    )

    if mse == 0:
        return float("inf")

    return float(
        20 * np.log10(
            1.0 / np.sqrt(mse)
        )
    )


# ============================================================
# PIXEL FLIPPING
# ============================================================

def pixel_storage_capacity(arr):
    """
    Total number of bits that can physically be stored.
    """

    return arr.size


def pixel_message_capacity(arr):
    """
    Message capacity after reserving 32 bits for header.
    """

    return max(
        0,
        pixel_storage_capacity(arr) - HEADER_BITS
    )


def embed_pixel_flipping(arr, bits):
    """
    Pixels are scanned row-wise.

    Each selected pixel is forced to equal
    the corresponding secret bit.
    """

    flat = arr.flatten().copy()

    if len(bits) > pixel_storage_capacity(arr):
        max_bytes = pixel_message_capacity(arr) // 8

        raise ValueError(
            f"Capacity exceeded. "
            f"Maximum message size is approximately "
            f"{max_bytes} bytes."
        )

    flat[:len(bits)] = bits

    return flat.reshape(arr.shape)


def extract_pixel_flipping(arr):
    """
    Read bits row-wise and recover message.
    """

    flat = arr.flatten()

    return read_payload(flat)


# ============================================================
# BLOCK PARITY
# ============================================================

def block_positions(arr):
    """
    Generate non-overlapping 2x2 blocks
    in row-major order.
    """

    h = (arr.shape[0] // BLOCK_SIZE) * BLOCK_SIZE
    w = (arr.shape[1] // BLOCK_SIZE) * BLOCK_SIZE

    for r in range(0, h, BLOCK_SIZE):
        for c in range(0, w, BLOCK_SIZE):
            yield r, c


def block_count(arr):
    """
    Number of usable 2x2 blocks.
    """

    return (
        arr.shape[0] // BLOCK_SIZE
    ) * (
        arr.shape[1] // BLOCK_SIZE
    )


def block_message_capacity(arr):
    """
    One 2x2 block stores one bit.

    32 bits are reserved for the header.
    """

    return max(
        0,
        block_count(arr) - HEADER_BITS
    )


def embed_block_parity(arr, bits):
    """
    For each 2x2 block:

    parity = (# of 1s) mod 2

    If parity != secret bit,
    flip the bottom-right pixel.
    """

    out = arr.copy()

    positions = list(
        block_positions(out)
    )

    if len(bits) > len(positions):
        max_bytes = block_message_capacity(arr) // 8

        raise ValueError(
            f"Capacity exceeded. "
            f"Maximum message size is approximately "
            f"{max_bytes} bytes."
        )

    for bit, (r, c) in zip(bits, positions):

        block = out[
            r:r + 2,
            c:c + 2
        ]

        parity = int(
            block.sum() % 2
        )

        if parity != int(bit):

            # Agreed pixel:
            # bottom-right of the 2x2 block
            out[r + 1, c + 1] ^= 1

    return out


def extract_block_parity(arr):
    """
    Compute parity of each 2x2 block
    and recover the message.
    """

    bits = []

    for r, c in block_positions(arr):

        block = arr[
            r:r + 2,
            c:c + 2
        ]

        parity = int(
            block.sum() % 2
        )

        bits.append(parity)

    return read_payload(bits)


# ============================================================
# PATTERN SUBSTITUTION
# ============================================================

# Fixed project dictionary
#
# Pattern A -> 0
#
# 1 0
# 0 1
#
#
# Pattern B -> 1
#
# 0 1
# 1 0

PATTERN_A = np.array(
    [
        [1, 0],
        [0, 1]
    ],
    dtype=np.uint8
)

PATTERN_B = np.array(
    [
        [0, 1],
        [1, 0]
    ],
    dtype=np.uint8
)


def pattern_value(block):
    """
    Return:

    0 -> Pattern A
    1 -> Pattern B
    None -> not an eligible pattern
    """

    if np.array_equal(
        block,
        PATTERN_A
    ):
        return 0

    if np.array_equal(
        block,
        PATTERN_B
    ):
        return 1

    return None


def pattern_positions(arr):
    """
    Find all naturally occurring A/B patterns.

    Scan is row-wise using non-overlapping 2x2 blocks.
    """

    positions = []

    for r, c in block_positions(arr):

        block = arr[
            r:r + 2,
            c:c + 2
        ]

        if pattern_value(block) is not None:
            positions.append(
                (r, c)
            )

    return positions


def pattern_storage_capacity(arr):
    """
    Number of eligible A/B blocks.
    """

    return len(
        pattern_positions(arr)
    )


def pattern_message_capacity(arr):
    """
    32 bits are reserved for the message header.
    """

    return max(
        0,
        pattern_storage_capacity(arr)
        - HEADER_BITS
    )


def embed_pattern_substitution(arr, bits):
    """
    Encode bits using the fixed A/B dictionary.

    Existing A/B blocks are used.
    Non-A/B blocks are skipped.
    """

    out = arr.copy()

    positions = pattern_positions(out)

    if len(bits) > len(positions):

        max_bytes = (
            pattern_message_capacity(arr)
            // 8
        )

        raise ValueError(
            f"Capacity exceeded. "
            f"This image has {len(positions)} "
            f"eligible pattern blocks. "
            f"Maximum message size is approximately "
            f"{max_bytes} bytes."
        )

    for bit, (r, c) in zip(
        bits,
        positions
    ):

        desired = int(bit)

        current = pattern_value(
            out[
                r:r + 2,
                c:c + 2
            ]
        )

        if current == desired:
            continue

        if desired == 0:
            out[
                r:r + 2,
                c:c + 2
            ] = PATTERN_A

        else:
            out[
                r:r + 2,
                c:c + 2
            ] = PATTERN_B

    return out


def extract_pattern_substitution(arr):
    """
    Scan the same eligible A/B blocks
    and convert:

    A -> 0
    B -> 1
    """

    bits = []

    for r, c in block_positions(arr):

        block = arr[
            r:r + 2,
            c:c + 2
        ]

        value = pattern_value(block)

        if value is not None:
            bits.append(value)

    if len(bits) < HEADER_BITS:
        raise ValueError(
            "Not enough eligible pattern blocks "
            "for the 32-bit message header."
        )

    return read_payload(bits)


# ============================================================
# GENERAL METHOD HELPERS
# ============================================================

def get_capacity(method, arr):
    """
    Return maximum message capacity in bits.
    """

    if method == "Pixel Flipping":
        return pixel_message_capacity(arr)

    if method == "Block Parity":
        return block_message_capacity(arr)

    return pattern_message_capacity(arr)


def embed_method(method, arr, bits):

    if method == "Pixel Flipping":
        return embed_pixel_flipping(
            arr,
            bits
        )

    if method == "Block Parity":
        return embed_block_parity(
            arr,
            bits
        )

    return embed_pattern_substitution(
        arr,
        bits
    )


def extract_method(method, arr):

    if method == "Pixel Flipping":
        return extract_pixel_flipping(arr)

    if method == "Block Parity":
        return extract_block_parity(arr)

    return extract_pattern_substitution(arr)


def run_method(method, arr, message):
    """
    Embed + extract + calculate metrics.
    """

    bits = make_payload(message)

    start = time.perf_counter()

    stego = embed_method(
        method,
        arr,
        bits
    )

    elapsed = (
        time.perf_counter() - start
    )

    recovered = extract_method(
        method,
        stego
    )

    changed, percentage = changed_pixels(
        arr,
        stego
    )

    return (
        stego,
        recovered,
        changed,
        percentage,
        elapsed
    )


# ============================================================
# VISUALIZATION
# ============================================================

def make_change_overlay(original, stego):
    """
    Create a visual map where changed pixels
    are highlighted in red.

    This is much easier to see than a black/white
    difference image when only a few pixels change.
    """

    base = (
        original * 255
    ).astype(np.uint8)

    overlay = np.stack(
        [base, base, base],
        axis=-1
    )

    changed = (
        original != stego
    )

    # Changed pixels -> red
    overlay[changed] = [
        255,
        0,
        0
    ]

    return overlay


def make_enlarged_change_map(original, stego, scale=4):
    """
    Enlarged binary change map.

    Changed pixels become white and are enlarged.
    """

    diff = (
        original != stego
    ).astype(np.uint8)

    enlarged = np.kron(
        diff,
        np.ones(
            (scale, scale),
            dtype=np.uint8
        )
    )

    return enlarged * 255


def plot_changed_coordinates(original, stego, title):
    """
    Scatter plot of changed pixel locations.
    Useful when only a small number of pixels change.
    """

    changed = (
        original != stego
    )

    y, x = np.where(changed)

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    if len(x) > 0:
        ax.scatter(
            x,
            y,
            s=25
        )

    ax.set_title(title)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")

    ax.invert_yaxis()

    ax.set_xlim(
        0,
        original.shape[1]
    )

    ax.set_ylim(
        original.shape[0],
        0
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# ROBUSTNESS TEST
# ============================================================

def robustness_test(
    method,
    stego,
    expected_message,
    trials=20
):
    """
    Flip one random pixel in the stego image
    and check whether the message can still
    be recovered.

    Fixed random seed makes comparison reproducible.
    """

    rng = np.random.default_rng(42)

    successful = 0

    for _ in range(trials):

        attacked = stego.copy()

        index = rng.integers(
            0,
            attacked.size
        )

        attacked.flat[index] ^= 1

        try:

            recovered = extract_method(
                method,
                attacked
            )

            if recovered == expected_message:
                successful += 1

        except Exception:
            pass

    return (
        successful / trials
    ) * 100


# ============================================================
# HEADER
# ============================================================

st.title(
    "Binary Image Steganography Workbench"
)

st.caption(
    "Pixel Flipping • Block Parity • Pattern Substitution"
)

st.markdown(
    """
This application compares three binary-image steganography
techniques using the **same binary image and the same secret message**.
"""
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded = st.file_uploader(
    "Upload a cover image",
    type=[
        "png",
        "jpg",
        "jpeg",
        "bmp",
        "webp"
    ]
)

threshold = st.slider(
    "Binary threshold",
    min_value=0,
    max_value=255,
    value=128,
    help=(
        "Pixels with grayscale value greater than or equal "
        "to this threshold become white (1). "
        "Pixels below it become black (0)."
    )
)


if uploaded:

    original, binary_img, binary = (
        to_binary_image(
            uploaded,
            threshold
        )
    )

    # ========================================================
    # IMAGE DISPLAY
    # ========================================================

    c1, c2 = st.columns(2)

    with c1:

        st.subheader(
            "Original image"
        )

        st.image(
            original,
            use_container_width=True
        )

    with c2:

        st.subheader(
            "Binary image used by algorithms"
        )

        st.image(
            binary_img,
            use_container_width=True
        )

    st.info(
        "All three algorithms operate only on the binary image. "
        "Color and grayscale images are automatically converted "
        "using the selected threshold."
    )

    st.divider()

    # ========================================================
    # TABS
    # ========================================================

    tab_embed, tab_extract, tab_compare = st.tabs(
        [
            "Embed",
            "Extract",
            "Compare"
        ]
    )


    # ========================================================
    # EMBED TAB
    # ========================================================

    with tab_embed:

        st.subheader(
            "Embed a secret message"
        )

        method = st.selectbox(
            "Technique",
            METHODS
        )

        message = st.text_area(
            "Secret message",
            placeholder=(
                "Enter the message you want to hide..."
            )
        )

        if method == "Pixel Flipping":

            st.info(
                "Pixels are scanned row-wise. "
                "Each selected pixel is forced to equal "
                "the corresponding secret bit."
            )

        elif method == "Block Parity":

            st.info(
                "Non-overlapping 2×2 blocks are scanned row-wise. "
                "Parity = (# of 1s) mod 2. "
                "If required, the bottom-right pixel is flipped."
            )

        else:

            st.info(
                "Pattern A = 0 and Pattern B = 1. "
                "Both are fixed 2×2 diagonal patterns. "
                "Only naturally occurring A/B blocks are used."
            )

            st.code(
                """Pattern A → 0
[[1, 0],
 [0, 1]]

Pattern B → 1
[[0, 1],
 [1, 0]]"""
            )

        capacity_bits = get_capacity(
            method,
            binary
        )

        capacity_bytes = (
            capacity_bits // 8
        )

        st.metric(
            "Maximum message capacity",
            f"{capacity_bytes:,} bytes"
        )

        st.caption(
            "The 32-bit message-length header is already "
            "excluded from this displayed capacity."
        )

        if st.button(
            "Embed message",
            type="primary"
        ):

            if not message:

                st.warning(
                    "Enter a secret message first."
                )

            else:

                try:

                    (
                        stego,
                        recovered,
                        changed,
                        percentage,
                        elapsed
                    ) = run_method(
                        method,
                        binary,
                        message
                    )

                    if recovered == message:

                        st.success(
                            "Message embedded and recovered successfully."
                        )

                    else:

                        st.error(
                            "Embedding completed, but recovery verification failed."
                        )

                    st.image(
                        Image.fromarray(
                            stego * 255
                        ),
                        caption=(
                            f"{method} — stego image"
                        ),
                        use_container_width=True
                    )

                    m1, m2, m3 = st.columns(3)

                    with m1:
                        st.metric(
                            "Changed pixels",
                            f"{changed:,}"
                        )

                    with m2:
                        st.metric(
                            "Changed %",
                            f"{percentage:.4f}%"
                        )

                    with m3:
                        st.metric(
                            "Embedding time",
                            f"{elapsed * 1000:.2f} ms"
                        )

                    st.download_button(
                        "Download stego PNG",
                        data=image_bytes(stego),
                        file_name=(
                            "stego_"
                            + method.lower().replace(
                                " ",
                                "_"
                            )
                            + ".png"
                        ),
                        mime="image/png"
                    )

                except Exception as e:

                    st.error(
                        str(e)
                    )


    # ========================================================
    # EXTRACT TAB
    # ========================================================

    with tab_extract:

        st.subheader(
            "Extract a hidden message"
        )

        stego_upload = st.file_uploader(
            "Upload a stego PNG",
            type=["png"],
            key="stego_upload"
        )

        extract_method_name = st.selectbox(
            "Technique used during embedding",
            METHODS,
            key="extract_method"
        )

        if st.button(
            "Extract message",
            key="extract_message_button"
        ):

            if not stego_upload:

                st.warning(
                    "Upload a stego PNG first."
                )

            else:

                try:

                    stego_arr = (
                        binary_image_from_stego(
                            stego_upload
                        )
                    )

                    message = extract_method(
                        extract_method_name,
                        stego_arr
                    )

                    st.success(
                        "Message recovered successfully."
                    )

                    st.text_area(
                        "Recovered message",
                        message,
                        height=150
                    )

                except Exception as e:

                    st.error(
                        str(e)
                    )


    # ========================================================
    # COMPARE TAB
    # ========================================================

    with tab_compare:

        st.subheader(
            "Compare all three techniques"
        )

        compare_message = st.text_area(
            "Message for fair comparison",
            value="HELLO",
            key="compare_message"
        )

        st.caption(
            "The same binary image and same message are used "
            "for all three techniques."
        )

        if st.button(
            "Run comparison",
            type="primary"
        ):

            if not compare_message:

                st.warning(
                    "Enter a message for comparison."
                )

            else:

                rows = []

                stegos = {}

                robustness = {}

                # --------------------------------------------
                # Run every technique
                # --------------------------------------------

                for method_name in METHODS:

                    try:

                        (
                            stego,
                            recovered,
                            changed,
                            percentage,
                            elapsed
                        ) = run_method(
                            method_name,
                            binary,
                            compare_message
                        )

                        stegos[
                            method_name
                        ] = stego

                        robustness[
                            method_name
                        ] = robustness_test(
                            method_name,
                            stego,
                            compare_message
                        )

                        rows.append(
                            {
                                "Technique":
                                    method_name,

                                "Capacity (bits)":
                                    get_capacity(
                                        method_name,
                                        binary
                                    ),

                                "Capacity (bytes)":
                                    get_capacity(
                                        method_name,
                                        binary
                                    ) // 8,

                                "Changed Pixels":
                                    changed,

                                "Changed %":
                                    round(
                                        percentage,
                                        4
                                    ),

                                "PSNR (dB)":
                                    (
                                        "∞"
                                        if changed == 0
                                        else round(
                                            psnr(
                                                binary,
                                                stego
                                            ),
                                            2
                                        )
                                    ),

                                "Recovery":
                                    recovered
                                    == compare_message,

                                "Robustness %":
                                    round(
                                        robustness[
                                            method_name
                                        ],
                                        1
                                    ),

                                "Time (ms)":
                                    round(
                                        elapsed * 1000,
                                        2
                                    )
                            }
                        )

                    except Exception as e:

                        rows.append(
                            {
                                "Technique":
                                    method_name,

                                "Error":
                                    str(e)
                            }
                        )

                # --------------------------------------------
                # Metrics table
                # --------------------------------------------

                st.subheader(
                    "Performance comparison"
                )

                df = pd.DataFrame(
                    rows
                )

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )

                # --------------------------------------------
                # Visual comparison
                # --------------------------------------------

                st.subheader(
                    "Visual comparison"
                )

                cols = st.columns(4)

                with cols[0]:

                    st.image(
                        binary_img,
                        caption="Original binary",
                        use_container_width=True
                    )

                for i, method_name in enumerate(
                    METHODS,
                    start=1
                ):

                    with cols[i]:

                        if method_name in stegos:

                            st.image(
                                Image.fromarray(
                                    stegos[
                                        method_name
                                    ] * 255
                                ),
                                caption=method_name,
                                use_container_width=True
                            )

                # --------------------------------------------
                # Changed pixel overlays
                # --------------------------------------------

                st.subheader(
                    "Changed-pixel maps"
                )

                st.caption(
                    "Red pixels show exactly where the "
                    "stego image differs from the original binary image."
                )

                cols = st.columns(3)

                for i, method_name in enumerate(
                    METHODS
                ):

                    with cols[i]:

                        if method_name in stegos:

                            overlay = (
                                make_change_overlay(
                                    binary,
                                    stegos[
                                        method_name
                                    ]
                                )
                            )

                            st.image(
                                overlay,
                                caption=(
                                    method_name
                                    + " — changed pixels"
                                ),
                                use_container_width=True
                            )

                            changed_count, changed_pct = (
                                changed_pixels(
                                    binary,
                                    stegos[
                                        method_name
                                    ]
                                )
                            )

                            st.caption(
                                f"{changed_count:,} changed pixels "
                                f"({changed_pct:.4f}%)"
                            )

                # --------------------------------------------
                # Changed pixel coordinate plots
                # --------------------------------------------

                st.subheader(
                    "Changed-pixel locations"
                )

                st.caption(
                    "These plots make small numbers of changed "
                    "pixels visible even when the image is very large."
                )

                cols = st.columns(3)

                for i, method_name in enumerate(
                    METHODS
                ):

                    with cols[i]:

                        if method_name in stegos:

                            plot_changed_coordinates(
                                binary,
                                stegos[
                                    method_name
                                ],
                                method_name
                            )

                # --------------------------------------------
                # Robustness
                # --------------------------------------------

                st.subheader(
                    "Robustness test"
                )

                st.write(
                    "A deterministic test flips one random pixel "
                    "20 times and checks whether the original "
                    "message can still be recovered."
                )

                robustness_df = pd.DataFrame(
                    [
                        {
                            "Technique":
                                method_name,

                            "Successful recoveries":
                                f"{round(robustness[method_name] / 100 * 20)} / 20",

                            "Robustness":
                                f"{robustness[method_name]:.1f}%"
                        }

                        for method_name in METHODS
                        if method_name in robustness
                    ]
                )

                st.dataframe(
                    robustness_df,
                    use_container_width=True,
                    hide_index=True
                )


# ============================================================
# NO IMAGE
# ============================================================

else:

    st.info(
        "Upload an image to begin. "
        "PNG, JPG, JPEG, BMP and WEBP images are accepted. "
        "Color and grayscale images are automatically converted "
        "to binary using the selected threshold."
    )