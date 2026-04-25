"""
CustomJSON 任务：从 JSONL 文件加载对话数据，用于 SFT 训练。

每行是一个 JSON 数组，包含交替的 user/assistant 消息对象。
示例行：[{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]

典型用途：
- 加载身份对话数据（identity_conversations.jsonl）
- 加载自定义合成数据
- 加载任何 JSONL 格式的对话语料
"""

import os
import json
from tasks.common import Task

class CustomJSON(Task):
    """
    从 JSONL 文件加载对话。

    文件格式要求：
    - 每行一个 JSON 数组
    - 消息对象包含 'role' 和 'content' 字段
    - user 和 assistant 必须严格交替出现
    - 至少包含 2 条消息（一问一答）
    - content 必须是字符串类型

    加载时进行结构校验，遇到格式错误会直接断言失败。
    """

    def __init__(self, filepath, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.conversations = []

        # Load all conversations from the JSONL file
        if not os.path.exists(filepath):
            # Helpful error message due to recent change. Will be removed in the future.
            print("-" * 80)
            print(f"Warning: File {filepath} does not exist")
            print("HINT (Oct 21 2025)")
            print("If you recently did a git pull and suddenly see this, it might be due to the new addition of identity conversations")
            print("See this discussion for more details: https://github.com/karpathy/nanochat/discussions/139")
            print("Quick fix: simply run the following command to download the file and you're done:")
            print(f"curl -L -o {filepath} https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl")
            print("-" * 80)

        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:  # skip empty lines
                        continue
                    messages = json.loads(line)
                    # Validate the conversation structure
                    assert isinstance(messages, list), f"Expected list of messages, got {type(messages)}"
                    assert len(messages) >= 2, f"Conversation must have at least 2 messages, got {len(messages)}"
                    # Validate message structure and alternating roles
                    for i, message in enumerate(messages):
                        assert "role" in message, f"Message {i} missing 'role' field"
                        assert "content" in message, f"Message {i} missing 'content' field"
                        expected_role = "user" if i % 2 == 0 else "assistant"
                        assert message["role"] == expected_role, f"Message {i} has role {message['role']} but should be {expected_role}"
                        assert isinstance(message["content"], str), f"Message {i} content must be a string"

                    self.conversations.append(messages)

        self.length = len(self.conversations)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        messages = self.conversations[index]
        conversation = {
            "messages": messages,
        }
        return conversation

