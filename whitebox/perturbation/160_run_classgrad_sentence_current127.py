#!/usr/bin/env python3
"""CLI runner for the class-gradient, sentence-aware current127 experiment."""
import argparse
import importlib


experiment = importlib.import_module("159_scientist_classgrad_sentence_current127")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("collect", "evaluate", "all"))
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--saliency-mass", type=float, default=.75)
    parser.add_argument("--max-candidate-fraction", type=float, default=.60)
    parser.add_argument("--sentence-topk", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 < args.saliency_mass <= 1:
        parser.error("--saliency-mass must be in (0, 1]")
    if not 0 < args.max_candidate_fraction <= 1:
        parser.error("--max-candidate-fraction must be in (0, 1]")

    def shortlist(att, prep, spans, keep=None, blocks=None):
        return experiment.sentence_shortlist(
            att, prep, spans, saliency_mass=args.saliency_mass,
            max_candidate_fraction=args.max_candidate_fraction,
            topk=args.sentence_topk,
        )

    experiment.m.shortlist = shortlist
    # Compatibility attributes read by the inherited collector before they are
    # discarded by the sentence-aware shortlist above.
    args.keep = args.blocks = 0
    if args.stage in ("collect", "all"):
        experiment.m.collect(args)
    if args.stage in ("evaluate", "all"):
        experiment.evaluate()


if __name__ == "__main__":
    main()
