import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap

# --- Configuration ---
FILE_PATH = "/Users/noel/Documents/Research_Projects/segwater_v2/data/val_10pct.memmap"
DTYPE = 'float32'
CHANNELS, HEIGHT, WIDTH = 3, 224, 224
N_EXAMPLES = 5
OUTPUT_FILENAME = "memmap_analysis.png"

def visualize_samples():
    # 1. Calculate number of samples
    bytes_per_sample = np.dtype(DTYPE).itemsize * CHANNELS * HEIGHT * WIDTH
    n_samples = os.path.getsize(FILE_PATH) // bytes_per_sample
    
    # 2. Map the data
    data = np.memmap(FILE_PATH, dtype=DTYPE, mode='r', shape=(n_samples, CHANNELS, HEIGHT, WIDTH))
    indices = np.random.choice(n_samples, N_EXAMPLES, replace=False)

    # 3. Setup Plot (Rows = Samples, Cols = Ch0, Ch1, Mask)
    fig, axes = plt.subplots(N_EXAMPLES, 3, figsize=(12, 15))
    
    # Custom colormap for Mask: 0=Black, 1=White, 255=Red
    # We will map 255 to '2' for visualization purposes
    mask_cmap = ListedColormap(['black', 'white', 'red'])

    for i, idx in enumerate(indices):
        sample = data[idx]
        
        # --- Process Input Channels (Z-score to 0-1 range for display) ---
        # We use percentiles (2nd and 98th) to avoid outlier clipping issues
        for ch in range(2):
            img_ch = sample[ch]
            p2, p98 = np.percentile(img_ch, (2, 98))
            img_ch_rescaled = np.clip((img_ch - p2) / (p98 - p2 + 1e-5), 0, 1)
            
            axes[i, ch].imshow(img_ch_rescaled, cmap='gray')
            axes[i, ch].set_title(f"Idx {idx} | Ch {ch}")
            axes[i, ch].axis('off')

        # --- Process Mask (Channel 2) ---
        mask = sample[2].copy()
        # Remap 255 to 2 so the colormap (0, 1, 2) works
        mask_display = np.where(mask == 255, 2, mask)
        
        im_mask = axes[i, 2].imshow(mask_display, cmap=mask_cmap, vmin=0, vmax=2)
        axes[i, 2].set_title(f"Idx {idx} | Mask (Red=255)")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_FILENAME, dpi=150)
    print(f"Saved visualization to {OUTPUT_FILENAME}")

if __name__ == "__main__":
    if os.path.exists(FILE_PATH):
        visualize_samples()
    else:
        print(f"File not found.")