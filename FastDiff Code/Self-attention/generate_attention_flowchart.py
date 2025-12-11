import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import os

# Set up figure - horizontal layout (adjusted aspect ratio)
fig, ax = plt.subplots(1, 1, figsize=(20, 11), facecolor='white')
ax.set_facecolor('white')
ax.set_xlim(0, 20)
ax.set_ylim(0, 11)
ax.axis('off')

# Colors
BLUE = '#2563eb'
PINK = '#db2777'
GREEN = '#16a34a'
PURPLE = '#7c3aed'
ORANGE = '#ea580c'
GRAY = '#6b7280'
DARK = '#1f2937'
CYAN = '#0891b2'

LIGHT_BLUE = '#dbeafe'
LIGHT_PINK = '#fce7f3'
LIGHT_GREEN = '#dcfce7'
LIGHT_PURPLE = '#ede9fe'
LIGHT_ORANGE = '#fff7ed'
LIGHT_CYAN = '#cffafe'

def draw_box(ax, x, y, w, h, text, fill_color, edge_color, fontsize=11, bold=False):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.2",
                         facecolor=fill_color, edgecolor=edge_color, linewidth=2.5)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=DARK, fontweight=weight)

def draw_circle(ax, x, y, r, text, fill_color, edge_color, fontsize=11):
    circle = plt.Circle((x, y), r, facecolor=fill_color, edgecolor=edge_color, linewidth=2.5)
    ax.add_patch(circle)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=DARK, fontweight='bold')

def draw_arrow(ax, x1, y1, x2, y2, color=GRAY, lw=2.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))

# ==================== TITLE ====================
ax.text(10, 10.3, 'ContextUnet with Self-Attention', fontsize=26, fontweight='bold',
        ha='center', color=DARK)

# ==================== MAIN FLOW ====================
main_y = 6.0

# Input
draw_circle(ax, 1.0, main_y, 0.55, 'x', LIGHT_GREEN, GREEN, 16)
ax.text(1.0, main_y - 1.0, 'Input\n[3,32,32]', fontsize=11, ha='center', color=GRAY)

draw_arrow(ax, 1.6, main_y, 2.2, main_y, GRAY)

# Encoder
draw_box(ax, 3.1, main_y, 1.6, 1.3, 'init_conv', LIGHT_BLUE, BLUE, 13)
ax.text(3.1, main_y - 1.0, '[256,32,32]', fontsize=10, ha='center', color=GRAY)

draw_arrow(ax, 3.95, main_y, 4.55, main_y, GRAY)

draw_box(ax, 5.4, main_y, 1.5, 1.3, 'down1', LIGHT_BLUE, BLUE, 13)
ax.text(5.4, main_y - 1.0, '[256,16,16]', fontsize=10, ha='center', color=GRAY)

draw_arrow(ax, 6.2, main_y, 6.8, main_y, GRAY)

draw_box(ax, 7.6, main_y, 1.5, 1.3, 'down2', LIGHT_BLUE, BLUE, 13)
ax.text(7.6, main_y - 1.0, '[512,8,8]', fontsize=10, ha='center', color=GRAY)

draw_arrow(ax, 8.4, main_y, 9.0, main_y, GRAY)

# ==================== ATTENTION (HIGHLIGHTED) ====================
attn_box = FancyBboxPatch((9.0, main_y - 0.9), 2.4, 1.8, boxstyle="round,pad=0.02,rounding_size=0.2",
                          facecolor='#fef3c7', edgecolor=ORANGE, linewidth=3)
ax.add_patch(attn_box)
ax.text(10.2, main_y + 0.35, 'Self-Attention', fontsize=14, fontweight='bold', ha='center', color=ORANGE)
ax.text(10.2, main_y - 0.2, '(4 heads)', fontsize=12, ha='center', color=ORANGE)
ax.text(10.2, main_y - 1.0, '[512,8,8]', fontsize=10, ha='center', color=GRAY)

ax.text(10.2, main_y + 1.4, 'NEW', fontsize=12, fontweight='bold', ha='center', color='white',
        bbox=dict(boxstyle='round', facecolor=ORANGE, edgecolor=ORANGE, pad=0.3))

draw_arrow(ax, 11.45, main_y, 12.05, main_y, GRAY)

# Decoder
draw_box(ax, 12.9, main_y, 1.5, 1.3, 'up0', LIGHT_PURPLE, PURPLE, 13)

draw_arrow(ax, 13.7, main_y, 14.3, main_y, GRAY)

draw_box(ax, 15.1, main_y, 1.4, 1.3, 'up1', LIGHT_PURPLE, PURPLE, 13)
ax.text(15.1, main_y - 1.0, '[256,16,16]', fontsize=10, ha='center', color=GRAY)

draw_arrow(ax, 15.85, main_y, 16.45, main_y, GRAY)

draw_box(ax, 17.2, main_y, 1.4, 1.3, 'up2', LIGHT_PURPLE, PURPLE, 13)
ax.text(17.2, main_y - 1.0, '[256,32,32]', fontsize=10, ha='center', color=GRAY)

draw_arrow(ax, 17.95, main_y, 18.4, main_y, GRAY)

# Output
draw_circle(ax, 19.0, main_y, 0.55, 'e', LIGHT_GREEN, GREEN, 16)
ax.text(19.0, main_y - 1.0, 'Noise\n[3,32,32]', fontsize=11, ha='center', color=GRAY)

# ==================== SKIP CONNECTIONS ====================
skip_y = main_y + 2.5

# Skip 1: init_conv -> up2
ax.annotate('', xy=(17.2, main_y + 0.7), xytext=(3.1, main_y + 0.7),
            arrowprops=dict(arrowstyle='->', color=BLUE, lw=2, linestyle='--',
                           connectionstyle='arc3,rad=-0.25'))
ax.text(10, skip_y + 0.6, 'skip connections', fontsize=12, color=BLUE, ha='center', style='italic')

# Skip 2: down1 -> up1
ax.annotate('', xy=(15.1, main_y + 0.7), xytext=(5.4, main_y + 0.7),
            arrowprops=dict(arrowstyle='->', color=BLUE, lw=2, linestyle='--',
                           connectionstyle='arc3,rad=-0.2'))

# Skip 3: down2 -> up0
ax.annotate('', xy=(12.9, main_y + 0.7), xytext=(7.6, main_y + 0.7),
            arrowprops=dict(arrowstyle='->', color=BLUE, lw=2, linestyle='--',
                           connectionstyle='arc3,rad=-0.15'))

# ==================== TIME & CONTEXT EMBEDDINGS ====================
embed_y = main_y - 2.8

draw_box(ax, 12.9, embed_y, 2.4, 0.9, 'Time Embed (t)', LIGHT_CYAN, CYAN, 12)
draw_box(ax, 15.8, embed_y, 2.4, 0.9, 'Class Embed (c)', LIGHT_PINK, PINK, 12)

draw_arrow(ax, 12.9, embed_y + 0.5, 12.9, main_y - 0.7, CYAN, 2)
draw_arrow(ax, 15.8, embed_y + 0.5, 15.8, main_y - 0.7, PINK, 2)

ax.text(14.35, embed_y - 0.9, 'Conditioning: h = cemb * h + temb', fontsize=11, 
        ha='center', color=GRAY, style='italic')

# ==================== ATTENTION DETAIL BOX ====================
detail_box = FancyBboxPatch((0.5, 0.5), 9, 2.8, boxstyle="round,pad=0.02,rounding_size=0.15",
                            facecolor=LIGHT_ORANGE, edgecolor=ORANGE, linewidth=2)
ax.add_patch(detail_box)

ax.text(5, 3.05, 'Self-Attention Detail', fontsize=14, fontweight='bold', ha='center', color=ORANGE)

# Simple flow
draw_box(ax, 1.8, 1.6, 1.8, 0.7, 'Flatten\n[B,64,512]', 'white', ORANGE, 11)
draw_arrow(ax, 2.75, 1.6, 3.35, 1.6, ORANGE, 2)

draw_box(ax, 4.2, 1.6, 1.5, 0.7, 'LayerNorm', 'white', ORANGE, 11)
draw_arrow(ax, 4.95, 1.6, 5.55, 1.6, ORANGE, 2)

draw_box(ax, 6.5, 1.6, 1.7, 0.7, 'MHA\n(4 heads)', '#ffedd5', ORANGE, 11, True)
draw_arrow(ax, 7.4, 1.6, 8.0, 1.6, ORANGE, 2)

draw_box(ax, 8.7, 1.6, 1.2, 0.7, 'FFN', 'white', ORANGE, 11)

ax.text(5, 0.85, 'MHA: Q=K=V (self-attention), + residual connections', fontsize=11, 
        ha='center', color=GRAY)

# ==================== LEGEND ====================
legend_y = 9.3
ax.add_patch(Rectangle((0.5, legend_y), 0.4, 0.4, facecolor=LIGHT_BLUE, edgecolor=BLUE, linewidth=2))
ax.text(1.1, legend_y + 0.2, 'Encoder', fontsize=12, color=DARK, va='center')

ax.add_patch(Rectangle((3, legend_y), 0.4, 0.4, facecolor=LIGHT_PURPLE, edgecolor=PURPLE, linewidth=2))
ax.text(3.6, legend_y + 0.2, 'Decoder', fontsize=12, color=DARK, va='center')

ax.add_patch(Rectangle((5.5, legend_y), 0.4, 0.4, facecolor='#fef3c7', edgecolor=ORANGE, linewidth=2))
ax.text(6.1, legend_y + 0.2, 'Self-Attention (NEW)', fontsize=12, color=DARK, va='center')

ax.plot([9.5, 10.3], [legend_y + 0.2, legend_y + 0.2], color=BLUE, linestyle='--', linewidth=2)
ax.text(10.5, legend_y + 0.2, 'Skip', fontsize=12, color=DARK, va='center')

# ==================== SAVE ====================
script_dir = os.path.dirname(os.path.abspath(__file__))
png_path = os.path.join(script_dir, 'Attention_Model_Flowchart.png')
pdf_path = os.path.join(script_dir, 'Attention_Model_Flowchart.pdf')

plt.tight_layout()
plt.savefig(png_path, dpi=200, facecolor='white', 
            edgecolor='none', bbox_inches='tight', pad_inches=0.3)
plt.savefig(pdf_path, facecolor='white', 
            edgecolor='none', bbox_inches='tight', pad_inches=0.3)
print("Attention Model Flowchart saved:")
print(f"   - {png_path}")
print(f"   - {pdf_path}")
plt.close()
