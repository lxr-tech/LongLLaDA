import os

import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer


def main(path):

    device = 'cuda:0'

    print(path)

    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    file = '/'.join(os.path.realpath(__file__).split('/')[:-1])
    file = f'{file}/gov_report_001.txt'

    text = open(file, mode='r').readline()

    input_ids = tokenizer(text)['input_ids']

    prompt_len = 16385 if len(input_ids) > 16384 else ((len(input_ids) // 64) * 64 + 1)

    prompt = torch.tensor([input_ids[:prompt_len]]).to(device)

    with torch.no_grad():
        outputs = model(prompt, labels=prompt)
        loss = F.cross_entropy(outputs.logits[0, :-1], prompt[0, 1:], reduction='none')  # / p_mask[mask_index]
        loss = torch.cumsum(loss, dim=-1) / (torch.arange(prompt_len-1).to(loss) + 1)
        perplexity = torch.exp(loss)

    print(f"Perplexity: {perplexity.float().detach().cpu().numpy().tolist()[64::64]}", flush=True)


if __name__ == '__main__':

    path = 'meta-llama/Meta-Llama-3-8B-Instruct'

    main(path)

    path = 'meta-llama/Meta-Llama-3-8B'

    main(path)
