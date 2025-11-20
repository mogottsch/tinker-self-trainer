import json
import os
from pathlib import Path
from typing import Optional

from dagster import (
    asset,
    AssetCheckResult,
    asset_check,
    Definitions,
    FilesystemIOManager,
    AssetExecutionContext,
)
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    id: str
    parent_id: Optional[str] = Field(alias="parentId")
    children_ids: list[str] = Field(default_factory=list, alias="childrenIds")
    role: str
    content: str
    children: list["ChatMessage"] = Field(default_factory=list)


class ChatTree(BaseModel):
    id: str
    title: str
    root: ChatMessage
    size: int


class ChatBranch(BaseModel):
    chat_id: str
    messages: list[ChatMessage]


@asset
def raw_chats(context: AssetExecutionContext) -> list[dict]:
    root_path = Path("chats.json")
    alt_path = Path(__file__).resolve().parents[2] / "chats.json"
    target = root_path if root_path.exists() else alt_path
    if not target.exists():
        raise FileNotFoundError("chats.json not found.")

    with target.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@asset
def message_trees(raw_chats: list[dict]) -> list[ChatTree]:
    trees = []
    for entry in raw_chats:
        chat_data = entry.get("chat", {})
        history = chat_data.get("history", {})
        raw_messages = history.get("messages", {})

        if not raw_messages:
            continue

        messages_map: dict[str, ChatMessage] = {}
        for msg_id, msg_data in raw_messages.items():
            if "id" not in msg_data:
                msg_data["id"] = msg_id
            messages_map[msg_id] = ChatMessage(**msg_data)

        root: Optional[ChatMessage] = None
        for msg in messages_map.values():
            if msg.parent_id is None:
                root = msg
                break

        if not root:
            raise ValueError(f"Chat {entry.get('id')} has no root message.")

        _build_recursive_tree(root, messages_map)

        trees.append(
            ChatTree(
                id=entry.get("id", "unknown"),
                title=entry.get("title", "Untitled"),
                root=root,
                size=len(messages_map),
            )
        )

    return trees


def _build_recursive_tree(
    node: ChatMessage, messages_map: dict[str, ChatMessage]
) -> None:
    for child_id in node.children_ids:
        if child_id not in messages_map:
            raise ValueError(f"Message {node.id} references missing child {child_id}")

        child = messages_map[child_id]
        node.children.append(child)
        _build_recursive_tree(child, messages_map)


@asset
def message_branches(message_trees: list[ChatTree]) -> list[ChatBranch]:
    all_branches = []

    for tree in message_trees:
        paths = _traverse_recursive_tree(tree.root)
        for path in paths:
            all_branches.append(ChatBranch(chat_id=tree.id, messages=path))

    return all_branches


@asset
def training_data_jsonl(message_branches: list[ChatBranch]) -> None:
    output_path = os.path.join("data", "training_data.jsonl")
    os.makedirs("data", exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for branch in message_branches:
            sequence = [
                {"role": msg.role, "message": msg.content.strip()}
                for msg in branch.messages
                if msg.content.strip()
            ]
            if not sequence:
                continue
            if sequence[-1]["role"] == "assistant":
                sequence = sequence[:-1]
                if not sequence:
                    continue
            if sequence[0]["role"] == "user":
                sequence.insert(
                    0, {"role": "assistant", "message": "Hey! How can I help you?"}
                )
            if len(sequence) < 2:
                continue
            f.write(json.dumps(sequence) + "\n")
            count += 1

    print(f"Wrote {count} conversation sequences to {output_path}")


def _traverse_recursive_tree(node: ChatMessage) -> list[list[ChatMessage]]:
    if not node.children:
        return [[node]]

    paths = []
    for child in node.children:
        child_paths = _traverse_recursive_tree(child)
        for p in child_paths:
            paths.append([node] + p)

    return paths


@asset_check(asset=message_trees)
def check_is_tree(message_trees: list[ChatTree]) -> AssetCheckResult:
    failures = []
    for tree in message_trees:
        visited: set[str] = set()
        node_count = _count_nodes(tree.root, visited)
        if node_count != tree.size:
            failures.append(
                f"{tree.id}: expected {tree.size} nodes, found {node_count} (visited={len(visited)})."
            )
    if failures:
        return AssetCheckResult(passed=False, metadata={"failures": failures[:10]})
    return AssetCheckResult(passed=True)


def _count_nodes(node: ChatMessage, visited: set[str]) -> int:
    if node.id in visited:
        raise ValueError(f"Cycle detected at message {node.id}")
    visited.add(node.id)
    total = 1
    for child in node.children:
        total += _count_nodes(child, visited)
    return total


@asset_check(asset=raw_chats)
def check_raw_chats_structure(raw_chats: list[dict]) -> AssetCheckResult:
    failures = []
    tree_count = 0
    for entry in raw_chats:
        chat_data = entry.get("chat", {})
        history = chat_data.get("history", {})
        raw_messages = history.get("messages", {})
        if not raw_messages:
            continue
        roots = [
            msg_id
            for msg_id, msg in raw_messages.items()
            if msg.get("parentId") is None
        ]
        if len(roots) != 1:
            failures.append(
                f"{entry.get('id', 'unknown')}: expected 1 root, found {len(roots)}"
            )
            continue
        tree_count += 1
    if tree_count == 0:
        failures.append("No chat with a tree-like structure found.")
    if failures:
        return AssetCheckResult(passed=False, metadata={"failures": failures[:10]})
    return AssetCheckResult(passed=True, metadata={"tree_chats": tree_count})


defs = Definitions(
    assets=[raw_chats, message_trees, message_branches, training_data_jsonl],
    asset_checks=[check_raw_chats_structure, check_is_tree],
    resources={"io_manager": FilesystemIOManager(base_dir="data")},
)
