import pandas as pd
import torch
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib as mpl
import platform
from dataclasses import dataclass

# Get the absolute path to the src directory
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)

# Import necessary modules
import transformer
from transformer.model.transformer import make_model, subsequent_mask
from transformer.training.util import TrainState
from transformer.data.dataset import tokenize, load_tokenizers
from transformer.evaluation.util import greedy_decode


# Set font based on your operating system
def setup_font_for_cjk():
    system = platform.system()

    if system == 'Windows':
        # Common CJK fonts on Windows
        cjk_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong', 'KaiTi', 'Arial Unicode MS']
    elif system == 'Darwin':  # macOS
        # Common CJK fonts on macOS
        cjk_fonts = ['Hiragino Sans GB', 'Apple SD Gothic Neo', 'Heiti SC', 'Apple LiGothic', 'Arial Unicode MS']
    else:  # Linux and others
        # Common CJK fonts on Linux
        cjk_fonts = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'Noto Sans CJK TC', 'WenQuanYi Micro Hei',
                     'Droid Sans Fallback']

    # Add more generic fonts as fallbacks
    cjk_fonts.extend(['Noto Sans CJK', 'Source Han Sans', 'WenQuanYi Zen Hei', 'Arial Unicode MS'])

    # Check for any font that supports CJK
    for font in cjk_fonts:
        if any([font.lower() in f.name.lower() for f in fm.fontManager.ttflist]):
            print(f"Using font: {font}")
            plt.rcParams['font.family'] = font
            return font

    # Add custom font file if needed (comment this out if not using)
    try:
        # Replace this with your font file path
        # custom_font_path = '/path/to/your/font.ttf'
        # fm.fontManager.addfont(custom_font_path)
        # plt.rcParams['font.family'] = 'Your Font Name'
        # return 'Your Font Name'
        pass
    except:
        pass

    print("Warning: No suitable CJK font found. Chinese characters may not display correctly.")
    return None


setup_font_for_cjk()


# Define the function to get model translation
def get_model_translation(model_path, src_text, config):
    """
    Generate a translation using the model.

    Args:
        model_path: Path to the model checkpoint
        src_text: Source text to translate
        config: Configuration

    Returns:
        Translation text
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    src_vocab, tgt_vocab = checkpoint["src_vocab"], checkpoint["tgt_vocab"]

    model = make_model(
        len(src_vocab),
        len(tgt_vocab),
        N=config.model_layers,
        d_model=config.d_model,
        d_ff=config.d_ff,
        h=config.h,
        dropout=0.0
    )

    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    # Load tokenizers
    tokenizers = load_tokenizers(config.spacy_models)

    # Tokenize source text
    src_tokens = ["<s>"] + tokenize(src_text, tokenizers[config.src_lang]) + ["</s>"]
    print("Source tokens:", src_tokens)

    # Convert to indices
    src_indices = [src_vocab[token] for token in src_tokens]

    # Create tensors
    src = torch.tensor([src_indices], dtype=torch.long).to(device)
    src_mask = (src != src_vocab["<blank>"]).unsqueeze(-2).to(device)

    # Generate translation with greedy decode
    output_ids = greedy_decode(
        model,
        src,
        src_mask,
        max_len=100,  # Set a reasonable max length
        start_symbol=tgt_vocab["<s>"]
    )[0]

    # Convert to tokens
    pad_id = tgt_vocab["<blank>"]
    eos_token = "</s>"
    out_tokens = [tgt_vocab.lookup_token(x) for x in output_ids if x != pad_id]

    if eos_token in out_tokens:
        out_tokens = out_tokens[:out_tokens.index(eos_token) + 1]

    out_text = " ".join([t for t in out_tokens if t not in ["<s>", "</s>", "<blank>"]])

    print("Model output:", out_text)

    return out_text


def plot_attention_map(attn, layer, head, row_tokens, col_tokens, ax=None, title=None, max_dim=30):
    """
    Plots an attention map using matplotlib

    Args:
        attn: Attention matrix
        layer: Layer index
        head: Head index
        row_tokens: Tokens for rows (target)
        col_tokens: Tokens for columns (source)
        ax: Matplotlib axis (optional)
        title: Plot title (optional)
        max_dim: Maximum dimension to display
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    # Extract the attention weights for the specific head
    attn_data = attn[0, head].data.cpu().numpy()

    # Limit dimensions if needed
    max_rows = min(len(row_tokens), max_dim)
    max_cols = min(len(col_tokens), max_dim)
    attn_data = attn_data[:max_rows, :max_cols]


    # Create a heatmap
    sns.heatmap(
        attn_data,
        ax=ax,
        cmap="viridis",
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"shrink": 0.75}
    )

    # Set labels
    ax.set_xticks(np.arange(max_cols) + 0.5)
    ax.set_yticks(np.arange(max_rows) + 0.5)

    # Format tokens for display (add indices)
    col_labels = [f"{i}: {token}" for i, token in enumerate(col_tokens[:max_cols])]
    row_labels = [f"{i}: {token}" for i, token in enumerate(row_tokens[:max_rows])]

    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(row_labels, rotation=0, fontsize=10)

    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Layer {layer + 1}, Head {head + 1}")

    return ax


def visualize_with_matplotlib(viz_data, src_text, tgt_text, model_output_text, output_prefix="attention"):
    """
    Creates and saves matplotlib visualizations for attention data

    Args:
        viz_data: Dictionary with attention matrices
        src_text: Source text
        tgt_text: Target text
        output_prefix: Prefix for output filenames
    """

    print("output_prefix:", output_prefix)
    # Get data from viz_data
    layers_to_viz = [0, 2, 4]  # Same as in your original code
    heads_to_viz = [0, 3, 5, 7]  # Same as in your original code

    # 1. Encoder Self-Attention
    fig, axes = plt.subplots(
        len(layers_to_viz),
        len(heads_to_viz),
        figsize=(16, 12)
    )
    fig.suptitle(f"Encoder Self-Attention: {src_text}", fontsize=16)

    for i, layer_idx in enumerate(layers_to_viz):
        for j, head_idx in enumerate(heads_to_viz):
            ax = axes[i, j]
            plot_attention_map(
                viz_data[f"encoder_layer{layer_idx}_self"],
                layer_idx,
                head_idx,
                viz_data["src_tokens"],
                viz_data["src_tokens"],
                ax=ax
            )

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for title
    plt.savefig(f"./src/visualization/{output_prefix}_encoder_self.png", dpi=200)
    plt.close(fig)

    # 2. Decoder Self-Attention Training
    fig, axes = plt.subplots(
        len(layers_to_viz),
        len(heads_to_viz),
        figsize=(16, 12)
    )
    fig.suptitle(f"Decoder Self-Attention in Training: {tgt_text}", fontsize=16)

    for i, layer_idx in enumerate(layers_to_viz):
        for j, head_idx in enumerate(heads_to_viz):
            ax = axes[i, j]
            plot_attention_map(
                viz_data[f"decoder_layer{layer_idx}_self"],
                layer_idx,
                head_idx,
                viz_data["tgt_tokens"][1:],
                viz_data["tgt_tokens"][:-1],
                ax=ax
            )

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for title
    plt.savefig(f"./src/visualization/{output_prefix}_decoder_self_train.png", dpi=200)
    plt.close(fig)

    # 3. Decoder-Encoder Attention Training
    fig, axes = plt.subplots(
        len(layers_to_viz),
        len(heads_to_viz),
        figsize=(16, 12)
    )
    fig.suptitle(f"Decoder-Encoder Cross-Attention in Training: {tgt_text} ← {src_text}", fontsize=16)

    for i, layer_idx in enumerate(layers_to_viz):
        for j, head_idx in enumerate(heads_to_viz):
            ax = axes[i, j]
            plot_attention_map(
                viz_data[f"decoder_layer{layer_idx}_src"],
                layer_idx,
                head_idx,
                viz_data["tgt_tokens"][1:],
                viz_data["src_tokens"],
                ax=ax
            )

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for title
    plt.savefig(f"./src/visualization/{output_prefix}_decoder_cross_train.png", dpi=200)
    plt.close(fig)

    # 4. Decoder Self-Attention Inference
    fig, axes = plt.subplots(
        len(layers_to_viz),
        len(heads_to_viz),
        figsize=(16, 12)
    )
    fig.suptitle(f"Decoder Self-Attention in Inference: {model_output_text}", fontsize=16)

    for i, layer_idx in enumerate(layers_to_viz):
        for j, head_idx in enumerate(heads_to_viz):
            ax = axes[i, j]
            plot_attention_map(
                viz_data[f"decoder_layer{layer_idx}_self"],
                layer_idx,
                head_idx,
                viz_data["model_output_tokens"][1:],
                viz_data["model_output_tokens"][:-1],
                ax=ax
            )

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for title
    plt.savefig(f"./src/visualization/{output_prefix}_decoder_self_inference.png", dpi=200)
    plt.close(fig)

    # 5. Encoder Cross-Attention Inference
    fig, axes = plt.subplots(
        len(layers_to_viz),
        len(heads_to_viz),
        figsize=(16, 12)
    )
    fig.suptitle(f"Decoder-Encoder Cross-Attention in Inference: {model_output_text} ← {src_text}", fontsize=16)

    for i, layer_idx in enumerate(layers_to_viz):
        for j, head_idx in enumerate(heads_to_viz):
            ax = axes[i, j]
            plot_attention_map(
                viz_data[f"decoder_layer{layer_idx}_src"],
                layer_idx,
                head_idx,
                viz_data["model_output_tokens"][1:],
                viz_data["src_tokens"],
                ax=ax
            )

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for title
    plt.savefig(f"./src/visualization/{output_prefix}_decoder_cross_inference.png", dpi=200)
    plt.close(fig)




def get_attention_matrices(model, layer_idx):
    """Get attention matrices for all parts of the model"""
    return {
        "encoder_self": model.encoder.layers[layer_idx].self_attn.attn,
        "decoder_self": model.decoder.layers[layer_idx].self_attn.attn,
        "decoder_src": model.decoder.layers[layer_idx].src_attn.attn,
    }


def visualize_single_example(model_path, src_text, tgt_text, model_output, config):
    """Visualize attention for a single source-target sentence pair"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    src_vocab, tgt_vocab = checkpoint["src_vocab"], checkpoint["tgt_vocab"]

    model = make_model(
        len(src_vocab),
        len(tgt_vocab),
        N=config.model_layers,
        d_model=config.d_model,
        d_ff=config.d_ff,
        h=config.h,
        dropout=0.0  # No dropout for visualization
    )

    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    # 2. Load tokenizers
    tokenizers = load_tokenizers(config.spacy_models)

    # 3. Tokenize source and target
    src_tokens = ["<s>"] + tokenize(src_text, tokenizers[config.src_lang]) + ["</s>"]
    tgt_tokens = ["<s>"] + tokenize(tgt_text, tokenizers[config.tgt_lang]) + ["</s>"]
    model_output_tokens = ["<s>"] + tokenize(model_output, tokenizers[config.tgt_lang]) + ["</s>"]

    print(f"Source tokens: {src_tokens}")
    print(f"Target tokens: {tgt_tokens}")
    print(f"model_output_tokens tokens: {model_output_tokens}")

    # 4. Convert to indices
    src_indices = [src_vocab[token] for token in src_tokens]
    tgt_indices = [tgt_vocab[token] for token in tgt_tokens]
    model_output_indices = [tgt_vocab[token] for token in model_output_tokens]

    # 5. Create tensors
    src = torch.tensor([src_indices], dtype=torch.long).to(device)
    tgt = torch.tensor([tgt_indices], dtype=torch.long).to(device)
    output = torch.tensor([model_output_indices], dtype=torch.long).to(device)

    # 6. Create masks
    src_mask = (src != src_vocab["<blank>"]).unsqueeze(-2).to(device)
    tgt_mask = subsequent_mask(tgt.size(1)).type_as(src_mask.data).to(device)

    # 7. Forward pass to get attention
    with torch.no_grad():
        model(src, tgt, src_mask, tgt_mask)

    # 8. Collect attention matrices
    layers_to_viz = [0, 2, 4]  # Visualize these layers

    # Create a dictionary to hold all the attention matrices and tokens
    viz_data = {
        "src_tokens": src_tokens,
        "tgt_tokens": tgt_tokens,
        "model_output_tokens": model_output_tokens
    }

    # For each layer we want to visualize
    for layer_idx in layers_to_viz:
        # Get attention matrices
        attn_matrices = get_attention_matrices(model, layer_idx)

        # Store attention matrices in viz_data
        viz_data[f"encoder_layer{layer_idx}_self"] = attn_matrices["encoder_self"]
        viz_data[f"decoder_layer{layer_idx}_self"] = attn_matrices["decoder_self"]
        viz_data[f"decoder_layer{layer_idx}_src"] = attn_matrices["decoder_src"]

    # 9. Create and save matplotlib visualizations
    visualize_with_matplotlib(viz_data, src_text, tgt_text, model_output)

    return viz_data



@dataclass
class TranslationConfig:
    src_lang: str = "zh"
    tgt_lang: str = "en"
    spacy_models: dict = None
    model_layers: int = 6
    d_model: int = 512
    d_ff: int = 2048
    h: int = 8

    def __post_init__(self):
        if self.spacy_models is None:
            self.spacy_models = {
                "zh": "zh_core_web_sm",
                "en": "en_core_web_sm"
            }


# Create config
config = TranslationConfig()

# Define the source text and model path
src_text = "我長大後想當國王。"
checkpoint_path = "./src/visualization/epoch_40_bs128_acc21_warm3000.pt"

# Get model translation
model_translation = get_model_translation(checkpoint_path, src_text, config)

# Visualize a single example
viz_data = visualize_single_example(
    model_path=checkpoint_path,
    src_text=src_text,
    tgt_text="When I grow up, I want to be a king.",
    model_output=model_translation,
    config=config
)

print("Visualization complete!")