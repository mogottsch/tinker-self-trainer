import argparse
import asyncio
import logging
from typing import List, Dict

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer, Tokenizer

logger = logging.getLogger(__name__)


def get_llama3_tokenizer() -> Tokenizer:
    return get_tokenizer("meta-llama/Llama-3.2-1B")


def get_llama3_renderer(tokenizer: Tokenizer) -> renderers.Renderer:
    return renderers.get_renderer("llama3", tokenizer)


def encode_messages(tokenizer: Tokenizer, messages: List[Dict[str, str]]) -> List[int]:
    prompt_tokens = tokenizer.encode("<|begin_of_text|>", add_special_tokens=False)
    for msg in messages:
        prompt_tokens.extend(
            tokenizer.encode(
                f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>",
                add_special_tokens=False,
            )
        )
    prompt_tokens.extend(
        tokenizer.encode(
            "<|start_header_id|>user<|end_header_id|>\n\n",
            add_special_tokens=False,
        )
    )
    return prompt_tokens


async def generate_response(
    sampling_client: tinker.SamplingClient,
    tokenizer: Tokenizer,
    renderer: renderers.Renderer,
    messages: List[Dict[str, str]],
) -> str:
    prompt_tokens = encode_messages(tokenizer, messages)
    response = await sampling_client.sample_async(
        tinker.ModelInput.from_ints(prompt_tokens),
        sampling_params=tinker.SamplingParams(
            max_tokens=512,
            temperature=0.7,
            stop=renderer.get_stop_sequences(),
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
        base_model="meta-llama/Llama-3.2-1B",
        model_path=args.checkpoint,
    )
    tokenizer = get_llama3_tokenizer()
    renderer = get_llama3_renderer(tokenizer)
    await run_chat_loop(sampling_client, tokenizer, renderer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with your trained model.")
    parser.add_argument(
        "checkpoint", type=str, help="Tinker URI of the checkpoint (sampler_path)"
    )
    args = parser.parse_args()

    asyncio.run(chat(args))


if __name__ == "__main__":
    main()
