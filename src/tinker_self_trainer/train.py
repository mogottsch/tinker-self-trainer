from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import chz
import datasets
import tinker
from tinker_cookbook import hyperparam_utils
from tinker_cookbook.display import colorize_example
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train as supervised_train
from tinker_cookbook.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilder,
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
)

logger = logging.getLogger(__name__)


@chz.chz
class UserStyleDatasetBuilder(ChatDatasetBuilder):
    data_path: str = "data/training_data.jsonl"
    test_split: float = 0.1
    seed: int = 42

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        # Load conversations manually to handle the specific JSON structure
        # (each line is a list of dicts)
        conversations = []
        path = Path(self.data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    conversations.append({"messages": json.loads(line)})
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line in {path}")

        ds = datasets.Dataset.from_list(conversations)

        if self.test_split > 0 and len(ds) > 0:
            ds = ds.train_test_split(test_size=self.test_split, seed=self.seed)
            train_ds = ds["train"]
            test_ds = ds["test"]
        else:
            train_ds = ds
            test_ds = None

        def flatmap_fn(row: dict[str, Any]) -> list[tinker.Datum]:
            conversation = row["messages"]
            datums = []
            prefix = []
            for message in conversation:
                # Original code expected "message" key for content
                content = message.get("message", message.get("content", ""))
                entry = {"role": message["role"], "content": content}
                prefix.append(entry)

                # We only train on user messages
                if entry["role"] != "user":
                    continue

                # Construct annotated conversation where only the last message (user) is trainable
                annotated = []
                for idx, msg in enumerate(prefix):
                    annotated.append(
                        {
                            "role": msg["role"],
                            "content": msg["content"],
                            "trainable": idx == len(prefix) - 1,
                        }
                    )

                # Render
                datum = conversation_to_datum(
                    annotated,
                    self.renderer,
                    self.common_config.max_length,
                    train_on_what=TrainOnWhat.CUSTOMIZED,
                )
                if datum:
                    datums.append(datum)
            return datums

        supervised_dataset = SupervisedDatasetFromHFDataset(
            train_ds,
            batch_size=self.common_config.batch_size,
            flatmap_fn=flatmap_fn,
        )

        test_dataset = None
        if test_ds is not None:
            test_dataset = SupervisedDatasetFromHFDataset(
                test_ds,
                batch_size=self.common_config.batch_size,  # Use same batch size for eval
                flatmap_fn=flatmap_fn,
            )

        return supervised_dataset, test_dataset


def get_learning_rate(model_name: str, learning_rate: float | None) -> float:
    if learning_rate is not None:
        return learning_rate
    try:
        lr = hyperparam_utils.get_lr(model_name, is_lora=True)
        logger.info(f"Using recommended learning rate: {lr:.2e}")
        return lr
    except Exception as e:
        logger.warning(f"Could not determine recommended LR: {e}. Defaulting to 1e-4")
        return 1e-4


class TrainingArgs(argparse.Namespace):
    data_path: str
    model_name: str
    log_dir: str
    batch_size: int
    learning_rate: float | None
    max_length: int
    epochs: int
    lora_rank: int
    eval_every: int
    save_every: int
    test_split: float
    base_url: str | None
    preview_only: bool


def parse_args() -> TrainingArgs:
    parser = argparse.ArgumentParser(
        description="Train user-style responder with Tinker."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/training_data.jsonl",
        help="Path to training data",
    )
    parser.add_argument(
        "--model-name", type=str, default="meta-llama/Llama-3.2-1B", help="Model name"
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs/self_trainer", help="Log directory"
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=None, help="Learning rate"
    )
    parser.add_argument(
        "--max-length", type=int, default=4096, help="Max sequence length"
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    parser.add_argument(
        "--eval-every", type=int, default=100, help="Eval every N steps"
    )
    parser.add_argument(
        "--save-every", type=int, default=200, help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--test-split", type=float, default=0.1, help="Test split fraction"
    )
    parser.add_argument(
        "--base-url", type=str, default=None, help="Tinker service base URL"
    )
    parser.add_argument(
        "--preview-only", action="store_true", help="Print sample and exit"
    )
    args = parser.parse_args(namespace=TrainingArgs())
    return args


async def run(args: TrainingArgs):
    logging.basicConfig(level=logging.INFO)

    # Determine learning rate
    lr = get_learning_rate(args.model_name, args.learning_rate)

    # Configure Dataset Builder
    dataset_builder = UserStyleDatasetBuilder(
        data_path=args.data_path,
        test_split=args.test_split,
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=args.model_name,
            renderer_name="llama3",
            max_length=args.max_length,
            batch_size=args.batch_size,
            train_on_what=TrainOnWhat.CUSTOMIZED,
        ),
    )

    # Preview logic
    if args.preview_only:
        dataset, _ = dataset_builder()
        if len(dataset) == 0:
            logger.error("Dataset is empty.")
            return

        logger.info("Previewing first batch...")
        batch = dataset.get_batch(0)
        if not batch:
            logger.error("First batch is empty.")
            return

        tokenizer = dataset_builder.tokenizer
        print(colorize_example(batch[0], tokenizer))
        return

    # Configure Training
    config = supervised_train.Config(
        log_path=str(Path(args.log_dir).expanduser()),
        model_name=args.model_name,
        dataset_builder=dataset_builder,
        learning_rate=lr,
        num_epochs=args.epochs,
        lora_rank=args.lora_rank,
        eval_every=args.eval_every,
        save_every=args.save_every,
        base_url=args.base_url,
    )

    await supervised_train.main(config)


def main():
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
