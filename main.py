import argparse
import logging

from src.config import cfg
from src.inference.competition import evaluate_checkpoint_by_class, generate_submission
from src.logging_utils import setup_logging
from src.pipeline import build_training_artifacts, print_run_summary
from src.runtime import set_global_seed
from src.training.train import train


def run_training() -> None:
    logger = logging.getLogger(__name__)
    set_global_seed(cfg.data.seed)
    logger.info("Starting training run")

    artifacts = build_training_artifacts(cfg)
    print_run_summary(artifacts, cfg)
    train(
        model=artifacts.model,
        train_data=artifacts.train_data,
        dev_data=artifacts.dev_data,
        mapping=artifacts.mapping,
        config=cfg,
    )
    logger.info("Training run completed")


def run_submission(checkpoint_path: str | None, mapping_path: str | None) -> None:
    logger = logging.getLogger(__name__)
    set_global_seed(cfg.data.seed)
    logger.info(
        "Generating competition submission | checkpoint=%s regex_labels=%s",
        checkpoint_path or cfg.model_path,
        cfg.regex.enabled_labels,
    )
    generate_submission(
        config=cfg,
        checkpoint_path=checkpoint_path,
        mapping_path=mapping_path,
    )


def run_evaluation(
    dataset_path: str | None,
    split_name: str,
    checkpoint_path: str | None,
    mapping_path: str | None,
) -> None:
    logger = logging.getLogger(__name__)
    set_global_seed(cfg.data.seed)
    logger.info(
        "Evaluating checkpoint by class | checkpoint=%s dataset_path=%s split=%s regex_labels=%s",
        checkpoint_path or cfg.model_path,
        dataset_path or cfg.paths.train_data_file,
        split_name,
        cfg.regex.enabled_labels,
    )
    evaluate_checkpoint_by_class(
        config=cfg,
        dataset_path=dataset_path,
        split_name=split_name,
        checkpoint_path=checkpoint_path,
        mapping_path=mapping_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Competition NER pipeline")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("train", "submit", "evaluate"),
        default="train",
        help="train the model, generate a submission, or evaluate a labeled set",
    )
    parser.add_argument(
        "--dataset-path",
        help="optional path to a labeled TSV dataset for evaluation; defaults to competition/train_dataset.tsv",
    )
    parser.add_argument(
        "--split",
        choices=("train", "dev", "all"),
        default="dev",
        help="which split to evaluate when using a labeled dataset",
    )
    parser.add_argument(
        "--checkpoint-path",
        help="checkpoint to load for submit/evaluate; defaults to models/competition_bilstm_crf",
    )
    parser.add_argument(
        "--mapping-path",
        help="optional mapping file for legacy checkpoints that do not contain mappings",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(cfg.train_log_path)
    if args.command == "submit":
        run_submission(args.checkpoint_path, args.mapping_path)
        return
    if args.command == "evaluate":
        run_evaluation(args.dataset_path, args.split, args.checkpoint_path, args.mapping_path)
        return

    run_training()


if __name__ == "__main__":
    main()
