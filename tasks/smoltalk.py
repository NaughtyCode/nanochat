"""
SmolTalk — HuggingFace 发布的通用对话数据集，适合 SFT 训练。
https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk

我们使用 "smol" 版本（460K 训练 / 24K 测试），
规模更适合小模型的 SFT 微调。
数据集格式：每条记录含 messages 列表，user/assistant 交替，
可选 system 消息作为首条。
"""

from datasets import load_dataset
from tasks.common import Task

class SmolTalk(Task):
    """
    smol-smoltalk 对话数据集。训练集约 460K 条，测试集约 24K 条。

    数据校验规则：
    - 每条对话至少 2 条消息（一问一答）
    - user 和 assistant 必须严格交替
    - system 消息（可选）只能出现在对话开头
    - 所有 content 必须是字符串类型
    """

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SmolTalk split must be train|test"
        self.ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = row["messages"]
        # ---------------------------------------------------------------------
        # sanity checking asserts here
        # TODO: we could remove these asserts later, for now just don't want any footguns
        # there is an optional system message at the beginning
        assert len(messages) >= 1
        first_message = messages[0]
        if first_message["role"] == "system":
            rest_messages = messages[1:] # optional system message is OK
        else:
            rest_messages = messages
        assert len(rest_messages) >= 2, "SmolTalk messages must have at least 2 messages"
        for i, message in enumerate(rest_messages):
            # user and assistant alternate as user,assistant,user,assistant,...
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role, f"Message {i} has role {message['role']} but should be {expected_role}"
            assert isinstance(message["content"], str), "Content must be a string"
        # ---------------------------------------------------------------------
        # create and return the Conversation object (ok to emit the system message too)
        conversation = {
            "messages": messages,
        }
        return conversation
