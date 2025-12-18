import numpy as np
import matplotlib.pyplot as plt
import torch

@torch.no_grad()
def save_attention_plot(model, loader, device, out_path="results/figures/attention_example.png"):
    model.eval()
    x, y_class, y_hr = next(iter(loader))
    x = x.to(device)

    logits, hr_hat, attn = model(x)  # attn: (B,T)
    attn = attn[0].cpu().numpy()
    sig = x[0, 0].cpu().numpy()

    # Stretch attention to signal length for visualization
    attn_up = np.interp(
        np.linspace(0, len(attn)-1, num=len(sig)),
        np.arange(len(attn)),
        attn
    )

    plt.figure()
    plt.plot(sig, label="ECG window")
    plt.plot(attn_up * (sig.max() - sig.min()) + sig.min(), label="Attention (scaled)")
    plt.legend()
    plt.title("Attention over ECG window (example)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
