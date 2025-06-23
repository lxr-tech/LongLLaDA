# copy from projects_xrliu/LLaDA/my_get_log_likelihood.py

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

import os
import random

from tqdm import tqdm

import numpy as np

seed = 2025
os.environ['PYTHONHASHSEED'] = str(seed) 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False # if benchmark=True, deterministic will be False
torch.backends.cudnn.deterministic = True # choose a deterministic algorithm 


def forward_process(batch, prompt_index, mask_id):
    b, l = batch.shape

    target_len = (l - prompt_index.sum()).item()
    k = torch.randint(1, target_len + 1, (), device=batch.device)

    x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
    x = ((x - 1) % target_len) + 1
    assert x.min() >= 1 and x.max() <= target_len

    indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
    is_mask = indices < x.unsqueeze(1)
    for i in range(b):
        is_mask[i] = is_mask[i][torch.randperm(target_len)]

    is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)
    noisy_batch = torch.where(is_mask, mask_id, batch)

    # Return the masked batch and the mask ratio
    return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)


def get_logits(model, batch, prompt_index, cfg_scale, mask_id):
    if cfg_scale > 0.:
        assert len(prompt_index) == batch.shape[1]
        prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
        un_batch = batch.clone()
        un_batch[prompt_index] = mask_id
        batch = torch.cat([batch, un_batch])

    input = batch
    logits = model(input).logits

    if cfg_scale > 0.:
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
    return logits


@ torch.no_grad()
def get_ppl(model, prompt, answer, mc_num=8, batch_size=1, cfg_scale=0., mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (l1).
        answer: A tensor of shape (l2).
        mc_num: Monte Carlo estimation times.
                As detailed in Appendix B.5. Since MMLU, CMMLU, and C-EVAL only require the likelihood of a single token, a
                single Monte Carlo estimate is sufficient for these benchmarks. For all other benchmarks, we find that 128
                Monte Carlo samples are adequate to produce stable results.
        batch_size: Mini batch size.
        cfg_scale: Unsupervised classifier-free guidance scale.
        mask_id: The toke id of [MASK] is 126336.
    '''
    seq = torch.concatenate([prompt, answer])[None, :]
    # seq = seq.repeat((batch_size, 1)).to(model.device)
    prompt_index = torch.arange(seq.shape[1], device=model.device) < len(prompt)

    loss_ = []
    for _ in range(mc_num):
        perturbed_seq, p_mask = forward_process(seq, prompt_index, mask_id)
        mask_index = perturbed_seq == mask_id

        logits = get_logits(model, perturbed_seq, prompt_index, cfg_scale, mask_id)

        loss = F.cross_entropy(logits[mask_index], seq[mask_index], reduction='none')
        loss = loss.mean()

        loss_.append(loss.item())

    return np.exp(sum(loss_) / len(loss_))


def main(path):
    device = 'cuda:0'

    print(path)

    model = AutoModel.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    file = '/'.join(os.path.realpath(__file__).split('/')[:-1])
    file = f'{file}/gov_report_001.txt'

    text = open(file, mode='r').readline()

    input_ids = tokenizer(text)['input_ids']
    print(f'{len(input_ids)}')

    context_len, chunk_size = 16384, 64

    perplexity = []

    for i in tqdm(list(range(int(context_len // chunk_size) - 1))):
        prompt_len = (i+1) * chunk_size
        prompt = torch.tensor(input_ids[:prompt_len]).to(device)
        answer = torch.tensor(input_ids[prompt_len:prompt_len+chunk_size]).to(device)
        ppl = get_ppl(model, prompt, answer, mc_num=8)
        print(prompt_len, flush=True)
        perplexity.append(ppl)

    num_sample = len(perplexity)
    perplexity = np.log(np.array(perplexity))
    perplexity = np.exp(np.cumsum(perplexity) / (np.arange(num_sample) + 1))

    print(f"Perplexity: {perplexity.tolist()}", flush=True)


if __name__ == '__main__':

    path = 'GSAI-ML/LLaDA-8B-Instruct'

    main(path)

    path = 'GSAI-ML/LLaDA-8B-Base'

    main(path)
