import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def draw_architecture():
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.axis("off")

    # Colors
    c_frozen_bg = "#EBF3FB"
    c_frozen_border = "#2B6CB0"
    c_trainable_bg = "#FEF3C7"
    c_trainable_border = "#D97706"
    c_retrieval_bg = "#E6FFFA"
    c_retrieval_border = "#0D9488"
    c_data_bg = "#F3F4F6"
    c_data_border = "#4B5563"
    c_arrow = "#1F2937"

    # Helper function for rounded boxes
    def draw_box(x, y, w, h, title, subtitle="", bg="#FFFFFF", border="#000000", lw=1.5, ls="-", title_color="#000000", sub_color="#4B5563", fontsize=9):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                    facecolor=bg, edgecolor=border, linewidth=lw, linestyle=ls)
        ax.add_patch(box)
        if subtitle:
            ax.text(x + w/2, y + h*0.62, title, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=title_color)
            ax.text(x + w/2, y + h*0.28, subtitle, ha="center", va="center", fontsize=fontsize-1.5, color=sub_color)
        else:
            ax.text(x + w/2, y + h*0.5, title, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=title_color)

    # 1. Document & Input Section (Left)
    draw_box(4, 55, 20, 10, "Source Document", "Paragraph: S_1, S_2, ..., S_t", c_data_bg, c_data_border)
    draw_box(4, 38, 20, 9, "Current Sentence (S_t)", "e.g., 'Google Chrome'", c_data_bg, c_data_border)
    draw_box(4, 21, 20, 9, "Preceding Context (S_<t)", "k <= 2 prior sentences", c_data_bg, c_data_border)

    # 2. Semantic Context Retrieval (Middle-Left)
    draw_box(29, 18, 23, 14, "Semantic Retrieval & Encoding", "all-MiniLM-L6-v2 (Frozen)\nCosine Sim Top-k -> c in R^384", 
             c_retrieval_bg, c_retrieval_border, ls="--", title_color="#0F766E")

    # 3. NLLB Encoder (Middle-Left Top)
    draw_box(29, 36, 23, 12, "NLLB-200 Encoder", "distilled-600M (Frozen)\nH_enc in R^(B x L x 1024)", 
             c_frozen_bg, c_frozen_border, ls="--", title_color="#1E40AF")

    # 4. Trainable Context Adapter Module (Center Highlighted Region)
    # Background region for Adapter
    adapter_bg = patches.FancyBboxPatch((56, 10), 38, 44, boxstyle="round,pad=1.0,rounding_size=2.0",
                                        facecolor="#FFFBEB", edgecolor="#F59E0B", linewidth=2.0, linestyle="-")
    ax.add_patch(adapter_bg)
    ax.text(75, 52, "TRAINABLE CONTEXT ADAPTER (2.49M Params)", ha="center", va="center", fontsize=10, fontweight="bold", color="#92400E")

    # Inside Adapter
    draw_box(58, 14, 34, 8, "Context Projection", "Linear(384->1024) + LN + GELU", c_trainable_bg, c_trainable_border, title_color="#92400E")
    draw_box(58, 26, 34, 8, "Sequence Expansion & Concat", "Expand to [B, L, 1024] -> [H_enc ; C_seq]", c_trainable_bg, c_trainable_border, title_color="#92400E")
    draw_box(58, 38, 34, 8, "Gated Residual Fusion", "Gate G = Sigmoid(W [H_enc ; C_seq] + b), b_0=-4.0\nH_fused = H_enc + G (*) C_seq", c_trainable_bg, c_trainable_border, title_color="#92400E")

    # 5. NLLB Decoder (Right Top)
    draw_box(58, 57, 34, 10, "NLLB-200 Decoder (Frozen)", "Cross-Attention over H_fused\nAutoregressive Target Generation (te)", 
             c_frozen_bg, c_frozen_border, ls="--", title_color="#1E40AF")

    # 6. Target Output
    draw_box(30, 57, 21, 10, "Target Translation (y)", "e.g., Telugu Tokens", c_data_bg, c_data_border)

    # Connecting Arrows
    def arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.6, shrinkA=3, shrinkB=3))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 1.2, label, ha="center", va="bottom", fontsize=7.5, color="#374151")

    # Inputs to components
    arrow(14, 55, 14, 47)
    arrow(14, 38, 14, 30)
    arrow(24, 42.5, 29, 42.5, "S_t")
    arrow(24, 25.5, 29, 25.5, "S_<t")
    arrow(14, 38, 29, 23, "query S_t")

    # Context to Adapter
    arrow(52, 25, 58, 18, "c in R^384")
    arrow(75, 22, 75, 26, "C_proj")
    arrow(75, 34, 75, 38, "X_fusion")

    # Encoder to Adapter & Fusion
    arrow(52, 42, 58, 30, "H_enc")
    arrow(52, 43, 58, 42, "H_enc")

    # Adapter to Decoder
    arrow(75, 46, 75, 57, "H_fused in R^(B x L x 1024)")

    # Decoder to Output
    arrow(58, 62, 51, 62, "greedy / beam")

    # Legend / Annotation
    leg_x = 4
    leg_y = 6
    draw_box(leg_x, leg_y, 11, 4, "Frozen Module", "", c_frozen_bg, c_frozen_border, ls="--", fontsize=7.5, title_color="#1E40AF")
    draw_box(leg_x + 13, leg_y, 13, 4, "Trainable Adapter", "", c_trainable_bg, c_trainable_border, fontsize=7.5, title_color="#92400E")
    draw_box(leg_x + 28, leg_y, 13, 4, "Retrieval Module", "", c_retrieval_bg, c_retrieval_border, ls="--", fontsize=7.5, title_color="#0F766E")
    ax.text(leg_x + 44, leg_y + 2, "* NLLB-200 and MiniLM weights are strictly frozen during training and inference", 
            ha="left", va="center", fontsize=7.5, fontstyle="italic", color="#4B5563")

    out_dir = Path("research_paper_output/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / "system_architecture.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / "system_architecture.pdf", bbox_inches="tight")
    plt.close()
    print("Architecture diagram saved successfully to png and pdf.")

if __name__ == "__main__":
    draw_architecture()
