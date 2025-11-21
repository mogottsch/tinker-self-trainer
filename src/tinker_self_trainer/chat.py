import argparse
import asyncio
import logging
from typing import List, Dict

import tinker
from tinker_cookbook import renderers, model_info
from tinker_cookbook.tokenizer_utils import get_tokenizer, Tokenizer

logger = logging.getLogger(__name__)


def get_model_tokenizer(model_name: str) -> Tokenizer:
    return get_tokenizer(model_name)


def get_model_renderer(model_name: str, tokenizer: Tokenizer) -> renderers.Renderer:
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    return renderers.get_renderer(renderer_name, tokenizer)


def get_stop_sequences_for_inverted_roles(
    renderer: renderers.Renderer,
) -> list[int] | list[str]:
    stop_sequences = list(renderer.get_stop_sequences())
    
    if isinstance(renderer, renderers.DeepSeekV3Renderer) or isinstance(
        renderer, renderers.DeepSeekV3DisableThinkingRenderer
    ):
        assistant_token = renderer._get_special_token("Assistant")
        stop_sequences.append(assistant_token)
    
    return stop_sequences


async def generate_response(
    sampling_client: tinker.SamplingClient,
    tokenizer: Tokenizer,
    renderer: renderers.Renderer,
    messages: List[Dict[str, str]],
) -> str:
    model_input = renderer.build_generation_prompt(messages)  # type: ignore
    stop_sequences = get_stop_sequences_for_inverted_roles(renderer)

    response = await sampling_client.sample_async(
        model_input,
        sampling_params=tinker.SamplingParams(
            max_tokens=512,
            temperature=0.7,
            stop=stop_sequences,
        ),
        num_samples=1,
    )
    return tokenizer.decode(response.sequences[0].tokens)


async def run_chat_loop(
    sampling_client: tinker.SamplingClient,
    tokenizer: Tokenizer,
    renderer: renderers.Renderer,
) -> None:
    messages: List[Dict[str, str]] = []
    print("\nChat with your clone! (Type 'quit' to exit)\n")

    while True:
        user_input = input("Assistant (You): ")
        if user_input.lower() in ["quit", "exit"]:
            break

        messages.append({"role": "assistant", "content": user_input})
        response_text = await generate_response(
            sampling_client, tokenizer, renderer, messages
        )
        print(f"User (Clone): {response_text}")
        messages.append({"role": "user", "content": response_text})


async def chat(args: argparse.Namespace) -> None:
    service_client = tinker.ServiceClient()
    logger.info(f"Loading model from {args.checkpoint}...")
    sampling_client = service_client.create_sampling_client(
        base_model=args.model_name,
        model_path=args.checkpoint,
    )
    tokenizer = get_model_tokenizer(args.model_name)
    renderer = get_model_renderer(args.model_name, tokenizer)
    await run_chat_loop(sampling_client, tokenizer, renderer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with your trained model.")
    parser.add_argument(
        "checkpoint", type=str, help="Tinker URI of the checkpoint (sampler_path)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Base model name",
    )
    args = parser.parse_args()

    asyncio.run(chat(args))


if __name__ == "__main__":
    main()
