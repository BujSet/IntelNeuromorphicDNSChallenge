import matplotlib.pyplot as plt

baseline_epochs = [5, 10, 25, 50]
baseline_sisnr = [-2.8961, -1.3876, -0.2253, 0.3425]

cipic_epochs = [5, 25, 50, 100]
cipic_sisnr = [0.7229, 3.9968, 4.6532, 5.354]

hybrid_demucs_sisnr = 9.0685

fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

ax.axhline(
    hybrid_demucs_sisnr,
    color="#898781",
    linestyle="--",
    linewidth=2,
    label="Hybrid Demucs",
    zorder=1,
)

ax.scatter(
    baseline_epochs, baseline_sisnr,
    s=70, color="#2a78d6", edgecolor="#fcfcfb", linewidth=1.5,
    label="Baseline SNN", zorder=3,
)
ax.scatter(
    cipic_epochs, cipic_sisnr,
    s=70, color="#eb6834", edgecolor="#fcfcfb", linewidth=1.5,
    label="CIPIC + SNN", zorder=3,
)

ax.set_xlabel("Epochs")
ax.set_ylabel("Test SI-SNR (dB)")
ax.set_title("Vocals SI-SNR vs. Training Epochs")
ax.set_xticks([5, 10, 25, 50, 100])
ax.set_ylim(top=12)

ax.grid(True, axis="y", color="#e1e0d9", linewidth=1, zorder=0)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("#c3c2b7")

ax.legend(frameon=False, loc="lower right")

fig.tight_layout()
out_path = "/home/esong32/IntelNeuromorphicDNSChallenge/chtc_files/data/si_snr_vs_epochs.png"
fig.savefig(out_path)
print(out_path)
