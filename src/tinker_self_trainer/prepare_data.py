import json
from pydantic import BaseModel
from tinker import types
from tinker_cookbook import renderers, tokenizer_utils

class ChatMessage(BaseModel):
    role: str
    content: str

def load_chats(file_path: str) -> list[dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_root_message_id(messages_map: dict) -> str | None:
    for message_id, message_data in messages_map.items():
        if message_data.get("parentId") is None:
            return message_id
    return None

def get_conversations_from_tree(messages_map: dict, current_id: str) -> list[list[ChatMessage]]:
    if not current_id or current_id not in messages_map:
        return [[]]
    
    msg_data = messages_map[current_id]
    try:
        msg = ChatMessage(
            role=msg_data.get("role", "unknown"),
            content=msg_data.get("content", "")
        )
    except Exception:
        return [[]]

    children_ids = msg_data.get("childrenIds", [])
    
    if not children_ids:
        return [[msg]]
    
    all_paths = []
    for child_id in children_ids:
        child_paths = get_conversations_from_tree(messages_map, child_id)
        for path in child_paths:
            all_paths.append([msg] + path)
            
    return all_paths

def extract_examples(chats: list[dict], limit: int = 100) -> list[dict]:
    examples = []

    for chat_entry in chats:
        if len(examples) >= limit:
            break
            
        messages_map = chat_entry.get("chat", {}).get("history", {}).get("messages", {})
        if not messages_map:
            continue
            
        root_id = find_root_message_id(messages_map)
        if not root_id:
            continue

        conversations = get_conversations_from_tree(messages_map, root_id)

        for messages in conversations:
            for i in range(len(messages) - 1):
                msg = messages[i]
                next_msg = messages[i+1]

                if msg.role != 'user':
                    continue
                if next_msg.role != 'assistant':
                    continue
                    
                user_content = msg.content.strip()
                assistant_content = next_msg.content.strip()

                if not user_content or not assistant_content:
                    continue

                examples.append({
                    "input": user_content,
                    "output": assistant_content
                })
                
                if len(examples) >= limit:
                    break
            if len(examples) >= limit:
                break
    
    print(f"Extracted {len(examples)} examples.")
    return examples

def process_example(example: dict, renderer) -> types.Datum:
    messages = [
        {'role': 'user', 'content': example['input']},
        {'role': 'assistant', 'content': example['output']}
    ]
    
    tokens, weights = renderer.build_supervised_example(messages)

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=tokens[:-1]),
        loss_fn_inputs=dict(weights=weights[1:], target_tokens=tokens[1:])
    )

def main():
    export_file = "chats.json"
    model_name = "meta-llama/Llama-3.2-1B"
    
    chats = load_chats(export_file)
    examples = extract_examples(chats, limit=100)
    
    if not examples:
        print("No examples found.")
        return

    print(f"Initializing tokenizer for {model_name}...")
    tokenizer = tokenizer_utils.get_tokenizer(model_name)
    renderer = renderers.get_renderer('llama3', tokenizer)

    processed_examples = [process_example(ex, renderer) for ex in examples]

    if not processed_examples:
        return

    datum0 = processed_examples[0]
    print("\n--- Processed Datum 0 ---")
    print(f"{'Input Token':<20} {'Target Token':<20} {'Weight':<10}")
    print("-" * 50)
    
    input_ints = datum0.model_input.to_ints()
    target_ints = datum0.loss_fn_inputs['target_tokens']
    weights_list = datum0.loss_fn_inputs['weights']

    for inp, tgt, wgt in zip(input_ints[:20], target_ints[:20], weights_list[:20]):
        inp_str = repr(tokenizer.decode([inp]))
        tgt_str = repr(tokenizer.decode([tgt]))
        print(f"{inp_str:<20} {tgt_str:<20} {wgt:<10}")

if __name__ == "__main__":
    main()
