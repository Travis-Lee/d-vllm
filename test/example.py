import os
import logging
from transformers import AutoTokenizer
from dvllm import LLM, SParams

def setup_logger(level: str = "DEBUG"):
    logging.basicConfig(
        level=getattr(logging, level),
        format="[%(levelname)s] %(name)s: %(message)s"
    )

def main():
    setup_logger()

    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    logging.info(f"Loading model from: {model_path}")

    # 初始化 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 初始化 LLM，指定 MPS
    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        device_type="mps",
    )

    # 生成参数
    sparams = SParams(temperature=0.7, max_tokens=128)

    # 一定要使用 chat template
    prompts = [
        "<|im_start|>user\nintroduce yourself<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nlist all prime numbers within 100<|im_end|>\n<|im_start|>assistant\n"
    ]

    logging.debug(f"Prompts for model: {prompts}")

    # 调用 generate
    outputs = llm.generate(prompts, sparams)

    # 输出结果
    for prompt, output in zip(prompts, outputs):
        print("\n==============================")
        print(f"Prompt:\n{prompt}")
        print(f"Completion:\n{output['text']}")
        print("==============================\n")

if __name__ == "__main__":
    main()
