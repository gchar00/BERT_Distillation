import torch

from transformers import DistilBertForMaskedLM


if __name__ == "__main__":
    
    model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
    #prefix = "bert"

    state_dict = model.state_dict()
    compressed_sd = {}

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
    #if args.vocab_transform:
    for w in ["weight", "bias"]:
        compressed_sd[f"vocab_transform.{w}"] = state_dict[f"vocab_transform.{w}"]
        compressed_sd[f"vocab_layer_norm.{w}"] = state_dict[f"vocab_layer_norm.{w}"]

    print(f"N layers selected for distillation: {std_idx}")
    print(f"Number of params transferred for distillation: {len(compressed_sd.keys())}")

    torch.save(compressed_sd, "serialization_dir/distilbert_four.pth")