Binary Image Steganography Workbench

Run
pip install -r requirements.txt
streamlit run app.py

Techniques
- Pixel Flipping: selected pixels are forced to equal the secret bits; pixels are scanned row-wise.
- Block Parity: non-overlapping 2×2 blocks; parity is (# of 1s) mod 2; if needed, the bottom-right pixel is flipped.
- Pattern Substitution: Pattern A → 0 and Pattern B → 1, using the fixed 2×2 dictionary:
  - A = [[1,0],[0,1]]
  - B = [[0,1],[1,0]]
The dictionary, 2×2 block size, and row-wise scan are explicit implementation choices used to make the PPT's Pattern Substitution concept deterministic.

The application accepts color, grayscale, and binary images and automatically converts them to binary using a configurable grayscale threshold.