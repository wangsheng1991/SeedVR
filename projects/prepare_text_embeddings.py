# // Copyright (c) 2025 Custom Training Implementation
# //
# // Licensed under the Apache License, Version 2.0 (the "License")

"""
Script to pre-compute text embeddings for SeedVR2 training

Uses T5-XXL encoder to generate embeddings for positive/negative prompts
"""

import os
import torch
from transformers import T5EncoderModel, T5Tokenizer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print("Loading T5-XXL encoder...")

    # Load T5-XXL (or use the encoder from your setup)
    model_name = "google/t5-v1_1-xxl"

    try:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        encoder = T5EncoderModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(device).eval()
    except Exception as e:
        print(f"Failed to load T5-XXL: {e}")
        print("Creating dummy embeddings instead...")

        # Create dummy embeddings
        embed_dim = 5120
        seq_len = 256

        pos_emb = torch.zeros(1, seq_len, embed_dim)
        neg_emb = torch.zeros(1, seq_len, embed_dim)

        torch.save(pos_emb, "pos_emb.pt")
        torch.save(neg_emb, "neg_emb.pt")

        print("Saved dummy embeddings to pos_emb.pt and neg_emb.pt")
        return

    # Positive prompt (quality enhancement)
    positive_text = """Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera,
    hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing,
    skin pore detailing, hyper sharpness, perfect without deformations."""

    # Negative prompt (artifacts to avoid)
    negative_text = """painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon,
    CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark,
    signature, jpeg artifacts, deformed, lowres, over-smooth"""

    print("Encoding positive prompt...")
    with torch.no_grad():
        pos_tokens = tokenizer(
            positive_text,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)

        pos_emb = encoder(
            input_ids=pos_tokens.input_ids,
            attention_mask=pos_tokens.attention_mask,
        ).last_hidden_state

    print("Encoding negative prompt...")
    with torch.no_grad():
        neg_tokens = tokenizer(
            negative_text,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)

        neg_emb = encoder(
            input_ids=neg_tokens.input_ids,
            attention_mask=neg_tokens.attention_mask,
        ).last_hidden_state

    # Save embeddings
    torch.save(pos_emb.cpu(), "pos_emb.pt")
    torch.save(neg_emb.cpu(), "neg_emb.pt")

    print(f"Saved embeddings:")
    print(f"  pos_emb.pt: {pos_emb.shape}")
    print(f"  neg_emb.pt: {neg_emb.shape}")


if __name__ == "__main__":
    main()
