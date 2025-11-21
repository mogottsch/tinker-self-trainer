# Tinker Self Trainer

Train a model to act like you using your OpenWebUI chat history with Tinker.

## Setup

1.  Install dependencies:

    ```bash
    uv sync
    ```

2.  Place your `chats.json` export in the project root.

## Running the Pipeline

We use Dagster for data processing.
To materialize all assets and generate the training data:

```bash
uv run dagster asset materialize --select training_data_jsonl -f src/tinker_self_trainer/dagster_pipeline.py
```

This will output the final training data to `data/training_data.jsonl`.

## Fine-Tuning with Tinker

1. Ensure `data/training_data.jsonl` exists (materialize the Dagster assets first).
2. Export your `TINKER_API_KEY`.
3. Run the supervised fine-tuning loop:

```bash
uv run python -m src.tinker_self_trainer.train
```

Metrics and checkpoints are written to `logs/self_trainer/`.

The training defaults (batch size 128, LR from the closed-form estimate) follow Tinker’s supervised learning hyperparameter guidance ([docs](https://tinker-docs.thinkingmachines.ai/supervised-learning/sl-hyperparams)).

### Preview Mode

To inspect a sample conversation (roles + trainable flag) without starting training:

```bash
uv run python -m src.tinker_self_trainer.train --preview-only
```

## Testing Your Clone

After training, you can chat with your fine-tuned model using the `chat.py` script.

1.  Find your checkpoint URI in `logs/self_trainer/checkpoints.jsonl`. It will look like `tinker://UUID:train:0/sampler_weights/000060`.
2.  Run the chat script:

```bash
uv run python src/tinker_self_trainer/chat.py "YOUR_CHECKPOINT_URI"
```
