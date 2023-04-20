import click
import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM

class TextModel(object):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")

    def generate_text(self, prompt: str) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        pad_token_id = self.tokenizer.pad_token_id
        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            max_length=105,
            do_sample=True,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def __call__(self, prompt: str) -> str:
        return self.generate_text(prompt)


@click.command()
@click.option("--prompt", required=True)
def inference(prompt: str) -> str:
    model = TextModel()
    generated_text = model(prompt)
    print(generated_text)
    return generated_text

if __name__ == "__main__":
    inference()
