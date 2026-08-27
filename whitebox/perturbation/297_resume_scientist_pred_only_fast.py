#!/usr/bin/env python3
"""Fast runner for experiment 296: score only the pred class once."""
import argparse
import importlib


base = importlib.import_module("296_scientist_known_pred_only")


def score(att, prep, spans):
    import torch
    zero = torch.zeros(len(prep.prompt_ids), device=att.device)
    alpha = torch.stack([zero, *[att.alpha_from_spans(prep, [i])
                                 for i in range(len(spans))]])
    values = []
    for start in range(0, len(alpha), att.max_rows):
        with torch.no_grad():
            embeds = att._embeds(prep, alpha[start:start + att.max_rows])
            value = att._class_logprob(embeds, prep.pred_variant_ids)
        values.append(value.detach().float().cpu())
    return torch.cat(values).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("collect", "evaluate", "all"))
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    base.score = score
    if args.stage in ("collect", "all"):
        base.collect(args)
    if args.stage in ("evaluate", "all"):
        base.evaluate()


if __name__ == "__main__":
    main()
