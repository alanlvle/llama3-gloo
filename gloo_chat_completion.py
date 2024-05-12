import datetime

from llama import Dialog, Llama
import os
import torch
from typing import List, Optional

os.environ["HUGGINGFACE_CO_RESOLVE_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_ENDPOINT "] = "https://hf-mirror.com"


print(torch.cuda.is_available())
print(torch.backends.cudnn.is_available())
print(torch.cuda_version)
print(torch.backends.cudnn.version())
print(datetime.datetime.utcnow())

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

check_point_directory = "Meta-Llama-3-8B-Instruct"
tokenizer_path = "Meta-Llama-3-8B-Instruct/tokenizer.model"
temperature = 0.6  # How much randomness is included in model output
top_p = 0.9  # Top-p probability threshold for nucleus sampling
max_sequence_length = 512  # The llama 3 model model produces up to 8192 tokens so max sequence length needs to be <= 8192
max_batch_size = 4
model_parallel_size = 1

if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend='gloo')

# Initialize the model
generator = Llama.build(
    ckpt_dir=check_point_directory,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_sequence_length,
    max_batch_size=max_batch_size,
    model_parallel_size=model_parallel_size
)


def dialog_chat_user(content: str):
    dialogs_temp:[Dialog] = [ [{"role": "user", "content": content}]]
    dialog_chat(dialogs_temp)


def dialog_chat(dialogs_temp:[Dialog]):
    results = generator.chat_completion(
        dialogs_temp,
        max_gen_len=max_sequence_length,
        temperature=temperature,
        top_p=top_p,
    )
    for dialog, result in zip(dialogs_temp, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


def main():

    dialogs: List[Dialog] = [
        [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
        [
            {"role": "user", "content": "I am going to Paris, what should I see?"},
            {
                "role": "assistant",
                "content": """\
    Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

    1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
    2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
    3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

    These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
            },
            {"role": "user", "content": "What is so great about #1?"},
        ],
        [
            {"role": "system", "content": "Always answer with Haiku"},
            {"role": "user", "content": "I am going to Paris, what should I see?"},
        ],
        [
            {
                "role": "system",
                "content": "Always answer with emojis",
            },
            {"role": "user", "content": "How to go from Beijing to NY?"},
        ],
    ]
    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_sequence_length,
        temperature=temperature,
        top_p=top_p,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    main()