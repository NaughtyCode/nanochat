"""
让 nanochat 在拼写和计数方面变得更强的任务，例如：

"How many r are in strawberry?" → 3

设计的巧妙之处在于：让 assistant 同时使用人工计数和 Python 工具来解决问题。
这种"双重验证"的解题思路是希望模型学会的良好习惯，后续 RL 训练可以进一步
优化模型在两种方法之间的信任度选择。

本文件包含两个任务：
1. SpellingBee — 统计单词中某个字母的出现次数（核心任务）
2. SimpleSpelling — 简单地逐个字母拼出单词（辅助任务）

(2) 的存在是因为它浓缩了 (1) 中最困难的部分：让 LLM 学会每个 token
（语义上的小片段/原子）如何映射到构成它的字符序列。大模型最终会自己学会这一点，
但如果想让小模型也具备这个能力，就必须在训练数据中大量添加相关内容来主动鼓励它。
SFT 阶段是引入这种能力的最佳时机。

运行预览：python -m tasks.spellingbee
"""

import re
import random
from tasks.common import Task
from nanochat.common import download_file_with_lock

# Letters of the alphabet
LETTERS = "abcdefghijklmnopqrstuvwxyz"
# A list of 370K English words of large variety
WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"
# A number bigger than 370K to separate train and test random seeds
TEST_RANDOM_SEED_OFFSET = 10_000_000

# 与 gsm8k 使用相同的答案提取逻辑
ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
def extract_answer(completion):
    """从 #### 标记后提取数值答案。"""
    match = ANSWER_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None

# 用户消息模板 — 通过多样化的提问方式做数据增强
# 包含英语、西班牙语、中文、韩语、法语、德语、日语共 7 种语言
USER_MSG_TEMPLATES = [
    "How many {letter} are in the word {word}",
    "How many {letter} are in {word}",
    "Count the number of {letter} in {word}",
    "How many times does {letter} appear in {word}",
    "What's the count of {letter} in {word}",
    "In the word {word}, how many {letter} are there",
    "How many letter {letter} are in the word {word}",
    "Count how many {letter} appear in {word}",
    "Tell me the number of {letter} in {word}",
    "How many occurrences of {letter} are in {word}",
    "Find the count of {letter} in {word}",
    "Can you count the {letter} letters in {word}",
    "What is the frequency of {letter} in {word}",
    "How many {letter}s are in {word}",
    "How many {letter}'s are in {word}",
    "Count all the {letter} in {word}",
    "How many times is {letter} in {word}",
    "Number of {letter} in {word}",
    "Total count of {letter} in {word}",
    "How many {letter} does {word} have",
    "How many {letter} does {word} contain",
    "What's the number of {letter} in {word}",
    "{word} has how many {letter}",
    "In {word}, count the {letter}",
    "How many {letter} appear in {word}",
    "Count the {letter} in {word}",
    "Give me the count of {letter} in {word}",
    "How many instances of {letter} in {word}",
    "Show me how many {letter} are in {word}",
    "Calculate the number of {letter} in {word}",
    # Spanish
    "¿Cuántas {letter} hay en {word}?",
    "¿Cuántas veces aparece {letter} en {word}?",
    "Cuenta las {letter} en {word}",
    "¿Cuántas letras {letter} tiene {word}?",
    # Chinese (Simplified)
    "{word}中有多少个{letter}",
    "{word}里有几个{letter}",
    "数一下{word}中的{letter}",
    "{word}这个词里有多少{letter}",
    # Korean
    "{word}에 {letter}가 몇 개 있나요",
    "{word}에서 {letter}의 개수는",
    "{word}에 {letter}가 몇 번 나오나요",
    "{word}라는 단어에 {letter}가 몇 개",
    # French
    "Combien de {letter} dans {word}",
    "Combien de fois {letter} apparaît dans {word}",
    "Compte les {letter} dans {word}",
    # German
    "Wie viele {letter} sind in {word}",
    "Wie oft kommt {letter} in {word} vor",
    "Zähle die {letter} in {word}",
    # Japanese
    "{word}に{letter}は何個ありますか",
    "{word}の中に{letter}がいくつ",
    "{word}に{letter}が何回出てくる",
]

class SpellingBee(Task):
    """
    "拼写蜜蜂"任务：统计单词中某个字母的出现次数。

    解题流程（assistant 的理想回复）：
    1. 手动逐字拼写单词，逐个计数
    2. 使用 Python 工具 `word.count(letter)` 进行双重验证
    3. 以 #### N 格式给出最终答案

    数据增强策略：
    - 7 种语言的提问模板随机选择
    - 30% 概率全小写（模拟不按 shift 的用户）
    - 50% 概率不加问号
    - 随机给字母/单词添加引号
    - 90% 概率选词中存在的字母，10% 概率选随机字母（处理字母不在词中的情况）
    """

    def __init__(self, size=1000, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SpellingBee split must be train|test"
        self.size = size
        self.split = split
        filename = WORD_LIST_URL.split("/")[-1]
        word_list_path = download_file_with_lock(WORD_LIST_URL, filename)
        with open(word_list_path, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f]
        self.words = words

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return self.size

    def get_example(self, index):
        seed = index if self.split == 'train' else TEST_RANDOM_SEED_OFFSET + index
        rng = random.Random(seed)

        # pick a random word
        word = rng.choice(self.words)
        # pick a letter from it (90%) or a random letter (10%)
        letter = rng.choice(word) if rng.random() < 0.9 else rng.choice(LETTERS)

        # get the correct answer by simply counting
        count = word.count(letter)

        # create a user message, with a bunch of variations as data augmentation
        template = rng.choice(USER_MSG_TEMPLATES)
        # 30% chance to lowercase the template (lazy people don't use shift)
        if rng.random() < 0.3:
            template = template.lower()
        quote_options = ['', "'", '"']
        letter_quote = rng.choice(quote_options) # is the letter quoted?
        word_quote = rng.choice(quote_options) # is the word quoted?
        letter_wrapped = f"{letter_quote}{letter}{letter_quote}"
        word_wrapped = f"{word_quote}{word}{word_quote}"
        user_msg = template.format(letter=letter_wrapped, word=word_wrapped)
        if rng.random() < 0.5: # 50% of people don't even use question marks
            user_msg += "?"

        # Now create the ideal assistant response - build as parts (text + tool calls)
        assistant_parts = []
        word_letters = ",".join(list(word))
        manual_text = f"""We are asked to find the number '{letter}' in the word '{word}'. Let me try a manual approach first.

First spell the word out:
{word}:{word_letters}

Then count the occurrences of '{letter}':
"""
        # Little simulated loop of the solution process
        # TODO: This is where the fun starts, we could simulate cute little mistakes
        # and get the model to review its work and recover from them.
        # You might of course hope this could arise in RL too, but realistically you'd want to help it out a bit.
        running_count = 0
        for i, char in enumerate(word, 1):
            if char == letter:
                running_count += 1
                # note: there deliberately cannot be a space here between i and char
                # because this would create a different token! (e.g. " a" and "a" are different tokens)
                manual_text += f"{i}:{char} hit! count={running_count}\n"
            else:
                manual_text += f"{i}:{char}\n"

        manual_text += f"\nThis gives us {running_count}."
        assistant_parts.append({"type": "text", "text": manual_text})
        # Part 2: Python verification
        assistant_parts.append({"type": "text", "text": "\n\nLet me double check this using Python:\n\n"})
        # Part 3: Python tool call
        python_expr = f"'{word}'.count('{letter}')"
        assistant_parts.append({"type": "python", "text": python_expr})
        # Part 4: Python output
        assistant_parts.append({"type": "python_output", "text": str(count)})
        # Part 5: Final answer
        assistant_parts.append({"type": "text", "text": f"\n\nPython gives us {count}.\n\nMy final answer is:\n\n#### {count}"})

        # return the full conversation
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_parts}
        ]
        conversation = {
            "messages": messages,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        评估：从对话和模型补全中提取 #### 答案，比较是否正确。
        与 gsm8k 的评估逻辑相同：基于 #### 标记提取数值并比较。
        """
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        # First extract the ground truth answer from the conversation
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "Last message must be from the Assistant"
        assert isinstance(assistant_message['content'], list), "This is expected to be a list of parts"
        # The last text part contains the final answer with ####
        last_text_part = assistant_message['content'][-1]['text']
        # Extract both the ground truth answer and the predicted answer
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        # Compare and return the success as int
        is_correct = int(pred_num == ref_num)
        return is_correct

    def reward(self, conversation, assistant_response):
        """简单 0-1 奖励，与 gsm8k 一致。用于 RL 训练。"""
        is_correct = self.evaluate(conversation, assistant_response)
        is_correct_float = float(is_correct)
        return is_correct_float


class SimpleSpelling(Task):
    """
    简单拼写任务：让模型练习逐个字母拼出单词。

    与 SpellingBee 使用不同的词序（独立 shuffle），避免两个任务产生
    相同的词序模式。这个任务作为 SpellingBee 的基础能力训练，
    帮助模型建立 token→字符 的映射能力。
    """

    def __init__(self, size=1000, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SpellingBee split must be train|test"
        self.size = size
        self.split = split
        filename = WORD_LIST_URL.split("/")[-1]
        word_list_path = download_file_with_lock(WORD_LIST_URL, filename)
        with open(word_list_path, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f]
        rng = random.Random(42)
        rng.shuffle(words) # use a different word order than the SpellingBee task
        self.words = words

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return self.size

    def get_example(self, index):
        seed = index if self.split == 'train' else TEST_RANDOM_SEED_OFFSET + index
        rng = random.Random(seed)
        # pick a random word
        word = rng.choice(self.words)
        word_letters = ",".join(list(word))
        # return the full conversation
        messages = [
            {"role": "user", "content": f"Spell the word: {word}"},
            {"role": "assistant", "content": f"{word}:{word_letters}"}
        ]
        conversation = {
            "messages": messages,
        }
        return conversation


if __name__ == "__main__":

    # preview the SpellingBee task, first 10 examples
    task = SpellingBee()
    for i in range(10):
        ex = task.get_example(i)
        print("=" * 100)
        print(ex['messages'][0]['content'])
        print("-" * 100)
        # Assistant content is now a list of parts
        assistant_parts = ex['messages'][1]['content']
        for part in assistant_parts:
            if part['type'] == 'text':
                print(part['text'], end='')
            elif part['type'] == 'python':
                print(f"<<{part['text']}=", end='')
            elif part['type'] == 'python_output':
                print(f"{part['text']}>>", end='')
        print()
        print("-" * 100)

    # # preview the SimpleSpelling task, first 10 examples
    # task = SimpleSpelling()
    # for i in range(10):
    #     ex = task.get_example(i)
    #     print("=" * 100)
    #     print(ex['messages'][0]['content'])
    #     print("-" * 100)
    #     print(ex['messages'][1]['content'])

    # # also scrutinize the tokenization (last example only)
    # from nanochat.tokenizer import get_tokenizer
    # tokenizer = get_tokenizer()
    # ids, mask = tokenizer.render_conversation(ex)
    # print(tokenizer.visualize_tokenization(ids, mask, with_token_id=True))
