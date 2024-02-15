"""
Preprocessing script before training a smaller DistilBERT.
Extracts weights from base DistilBERT to initialize the smaller
"""
import argparse

import torch

from transformers import DistilBertForMaskedLM


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Extraction some layers of the full BertForMaskedLM or RObertaForMaskedLM for Transfer Learned"
            " Distillation"
        )
    )
    parser.add_argument("--model_type", default="bert", choices=["bert", "distilbert"])
    parser.add_argument("--model_name", default="distilbert-base-uncased", type=str)
    parser.add_argument("--dump_checkpoint", default="serialization_dir/pw_db.pth", type=str)
    parser.add_argument("--vocab_transform", action="store_true")
    parser.add_argument("--n_layers", default="4")

    args = parser.parse_args()

    if args.model_type == "bert":
        model = DistilBertForMaskedLM.from_pretrained(args.model_name)
        prefix = "distilbert"
    else:
        raise ValueError('args.model_type should be "bert".')

    state_dict = model.state_dict()
    compressed_sd = {}

    for w in ["word_embeddings", "position_embeddings"]:
        compressed_sd[f"distilbert.embeddings.{w}.weight"] = state_dict[f"{prefix}.embeddings.{w}.weight"]
    for w in ["weight", "bias"]:
        compressed_sd[f"distilbert.embeddings.LayerNorm.{w}"] = state_dict[f"{prefix}.embeddings.LayerNorm.{w}"]
    
    std_idx = 0

    for teacher_idx in [0, 2, 3, 5]:
        for w in ["weight", "bias"]:
            compressed_sd[f"distilbert.transformer.layer.{std_idx}.attention.q_lin.{w}"] = state_dict[
                f"distilbert.transformer.layer.{std_idx}.attention.q_lin.{w}"
            ]
            compressed_sd[f"distilbert.transformer.layer.{std_idx}.attention.k_lin.{w}"] = state_dict[
                f"distilbert.transformer.layer.{std_idx}.attention.k_lin.{w}"
            ]
            compressed_sd[f"distilbert.transformer.layer.{std_idx}.attention.v_lin.{w}"] = state_dict[
                f"distilbert.transformer.layer.{std_idx}.attention.v_lin.{w}"
            ]

            compressed_sd[f"distilbert.transformer.layer.{std_idx}.attention.out_lin.{w}"] = state_dict[
                f"distilbert.transformer.layer.{std_idx}.attention.out_lin.{w}"
            ]
            compressed_sd[f"distilbert.transformer.layer.{std_idx}.sa_layer_norm.{w}"] = state_dict[
                f"distilbert.transformer.layer.{std_idx}.sa_layer_norm.{w}"
            ]

            compressed_sd[f"distilbert.transformer.layer.{std_idx}.ffn.lin1.{w}"] = state_dict[
                f"distilbert.transformer.layer.{std_idx}.ffn.lin1.{w}"
            ]
            compressed_sd[f"distilbert.transformer.layer.{std_idx}.ffn.lin2.{w}"] = state_dict[
                f"distilbert.transformer.layer.{std_idx}.ffn.lin2.{w}"
            ]
            compressed_sd[f"distilbert.transformer.layer.{std_idx}.output_layer_norm.{w}"] = state_dict[
                f"distilbert.transformer.layer.{std_idx}.output_layer_norm.{w}"
            ]
        std_idx += 1

    compressed_sd["vocab_projector.weight"] = state_dict["vocab_projector.weight"]
    compressed_sd["vocab_projector.bias"] = state_dict["vocab_projector.bias"]
    if args.vocab_transform:
        for w in ["weight", "bias"]:
            compressed_sd[f"vocab_transform.{w}"] = state_dict[f"vocab_transform.{w}"]
            compressed_sd[f"vocab_layer_norm.{w}"] = state_dict[f"vocab_layer_norm.{w}"]

    print(f"N layers selected for distillation: {std_idx}")
    print(f"Number of params transferred for distillation: {len(compressed_sd.keys())}")

    print(f"Save transferred checkpoint to {args.dump_checkpoint}.")
    torch.save(compressed_sd, args.dump_checkpoint)
