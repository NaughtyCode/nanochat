"""
在 HumanEval 数据集上评估 Chat 模型。
注意：这个数据集的名字有误导性，和"人类"无关——
它是一个编程能力基准测试（coding benchmark）。

评估流程：
1. 从模型补全中提取 Python 代码（支持 ```python``` 代码块）
2. 自动提取 prompt 中的 import 语句并前置到补全代码前
3. 拼接完整程序：imports + 补全代码 + 测试用例 + check(entry_point)
4. 在沙箱中执行代码，根据测试结果返回成功/失败
"""

import re
from datasets import load_dataset
from nanochat.execution import execute_code
from tasks.common import Task

def extract_imports(prompt):
    """从代码块开头提取 import 语句。遇到第一个非 import/非注释行即停止。"""
    imports = []
    for line in prompt.split('\n'):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            imports.append(stripped)
        elif stripped and not stripped.startswith('#'):
            # Stop at first non-import, non-comment line
            break
    return '\n'.join(imports)

def extract_program(completion):
    """
    从 LLM 补全中提取 Python 代码。

    处理多种输出格式：
    - ```python ... ``` 或 ``` ... ``` 包裹的代码块
    - 无 markdown 标记的纯代码
    - 代码块前后的额外文字

    返回找到的第一个代码块，如果没有则返回整个补全文本。
    """
    # Try to find markdown code blocks (```python or just ```)
    # Match ```python\n...\n``` or ```\n...\n```
    pattern = r'```(?:python)?\s*\n(.*?)\n```'
    matches = re.findall(pattern, completion, re.DOTALL)

    if matches:
        # Return the first code block found
        return matches[0].strip()

    # No code blocks found, return the whole completion
    return completion.strip()

class HumanEval(Task):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ds = load_dataset("openai/openai_humaneval", split="test").shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        """ Get a single problem from the dataset. """
        row = self.ds[index]
        prompt = row['prompt'] # prompts in HumanEval are the beginning of the program
        solution = row['canonical_solution'] # the correct continuation of the program
        entry_point = row['entry_point'] # the function to check
        test = row['test'] # the test cases
        complete_solution = f"{prompt}\n{solution}"
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": complete_solution},
        ]
        conversation = {
            "messages": messages,
            "entry_point": entry_point, # needed during evaluation
            "test": test, # needed during evaluation
        }
        return conversation

    def evaluate(self, conversation, completion):
        """ Given (conversation, completion), return boolean success of the completion. """
        # the prompt will contain the imports and the function signature
        imports = extract_imports(conversation['messages'][0]['content'])
        # the completion will usually contain the whole function
        # but not always with the needed imports, so we manually append them
        completion_code = extract_program(completion)
        program = (
            imports
            + "\n\n"
            + completion_code
            + "\n\n"
            + conversation['test']
            + "\n"
            + f"check({conversation['entry_point']})"
        )
        result = execute_code(program)
        success = result.success
        return success
