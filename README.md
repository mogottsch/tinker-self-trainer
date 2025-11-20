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

### Via CLI (Headless)

To materialize all assets and generate the training data:

```bash
uv run dagster asset materialize --select training_data_jsonl -f src/tinker_self_trainer/dagster_pipeline.py
```

This will output the final training data to `data/training_data.jsonl`.

### Via UI (Visual)

To launch the Dagster UI:

```bash
uv run dagster dev -f src/tinker_self_trainer/dagster_pipeline.py
```

Open [http://localhost:3000](http://localhost:3000) to view and launch runs.
