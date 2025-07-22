from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import LLaDACausalLM

import torch

with read_base():

    # longbench

    from opencompass.configs.datasets.longbench.longbenchnarrativeqa.longbench_narrativeqa_gen import LongBench_narrativeqa_datasets
    from opencompass.configs.datasets.longbench.longbenchqasper.longbench_qasper_gen import LongBench_qasper_datasets
    from opencompass.configs.datasets.longbench.longbenchmultifieldqa_en.longbench_multifieldqa_en_gen import LongBench_multifieldqa_en_datasets
    from opencompass.configs.datasets.longbench.longbenchmultifieldqa_zh.longbench_multifieldqa_zh_gen import LongBench_multifieldqa_zh_datasets

    from opencompass.configs.datasets.longbench.longbenchhotpotqa.longbench_hotpotqa_gen import LongBench_hotpotqa_datasets
    from opencompass.configs.datasets.longbench.longbench2wikimqa.longbench_2wikimqa_gen import LongBench_2wikimqa_datasets
    from opencompass.configs.datasets.longbench.longbenchmusique.longbench_musique_gen import LongBench_musique_datasets
    from opencompass.configs.datasets.longbench.longbenchdureader.longbench_dureader_gen import LongBench_dureader_datasets

    from opencompass.configs.datasets.longbench.longbenchgov_report.longbench_gov_report_gen import LongBench_gov_report_datasets
    from opencompass.configs.datasets.longbench.longbenchqmsum.longbench_qmsum_gen import LongBench_qmsum_datasets
    from opencompass.configs.datasets.longbench.longbenchmulti_news.longbench_multi_news_gen import LongBench_multi_news_datasets
    from opencompass.configs.datasets.longbench.longbenchvcsum.longbench_vcsum_gen import LongBench_vcsum_datasets

    from opencompass.configs.datasets.longbench.longbenchtrec.longbench_trec_gen import LongBench_trec_datasets
    from opencompass.configs.datasets.longbench.longbenchtriviaqa.longbench_triviaqa_gen import LongBench_triviaqa_datasets
    from opencompass.configs.datasets.longbench.longbenchsamsum.longbench_samsum_gen import LongBench_samsum_datasets
    from opencompass.configs.datasets.longbench.longbenchlsht.longbench_lsht_gen import LongBench_lsht_datasets

    from opencompass.configs.datasets.longbench.longbenchpassage_count.longbench_passage_count_gen import LongBench_passage_count_datasets
    from opencompass.configs.datasets.longbench.longbenchpassage_retrieval_en.longbench_passage_retrieval_en_gen import LongBench_passage_retrieval_en_datasets
    from opencompass.configs.datasets.longbench.longbenchpassage_retrieval_zh.longbench_passage_retrieval_zh_gen import LongBench_passage_retrieval_zh_datasets

    from opencompass.configs.datasets.longbench.longbenchlcc.longbench_lcc_gen import LongBench_lcc_datasets
    from opencompass.configs.datasets.longbench.longbenchrepobench.longbench_repobench_gen import LongBench_repobench_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

num_gpus = {
    'llama_3_8b_base': 1, 'llama_3_8b_chat': 1,

    'llada_8b_base': 1, 'llada_8b_chat': 1, 'llada_1_5_8b': 1, 

    'dream_v0_7b_base': 1, 'dream_v0_7b_chat': 1, 
}

path_dict = {
    'llama_3_8b_base': 'meta-llama/Meta-Llama-3-8B',
    'llama_3_8b_chat': 'meta-llama/Meta-Llama-3-8B-Instruct',

    'llada_8b_base': 'GSAI-ML/LLaDA-8B-Base', 
    'llada_8b_chat': 'GSAI-ML/LLaDA-8B-Instruct', 
    
    'llada_1_5_8b': 'GSAI-ML/LLaDA-1.5', 

    'dream_v0_7b_base': 'Dream-org/Dream-v0-Base-7B', 
    'dream_v0_7b_chat': 'Dream-org/Dream-v0-Instruct-7B', 
}

models = [
    # 8k
    ## llama series

    ('llama_3_8b_base-o512-8k', {}, {}, 7500, 512), 
    ('llama_3_8b_chat-o512-8k', {}, {}, 7500, 512), 

    ## llada series

    ('llada_8b_base-o512_b64_s512-8k', {}, {'steps': 512, 'block_length': 64, }, 7500, 512), 
    ('llada_8b_base-o512_b64_s512-ntk4-8k', {'scaling_factor': 4}, {'steps': 512, 'block_length': 64, }, 7500, 512), 
    ('llada_8b_chat-o512_b64_s512-8k', {}, {'steps': 512, 'block_length': 64, }, 7500, 512), 
    ('llada_8b_chat-o512_b64_s512-ntk4-8k', {'scaling_factor': 4}, {'steps': 512, 'block_length': 64, }, 7500, 512), 
    ('llada_1_5_8b-o512_b64_s512-8k', {}, {'steps': 512, 'block_length': 64, }, 7500, 512), 
    ('llada_1_5_8b-o512_b64_s512-ntk4-8k', {'scaling_factor': 4}, {'steps': 512, 'block_length': 64, }, 7500, 512), 

    ## dream series

    ('dream_v0_7b_base-o512_s512-8k', {}, {'steps': 512, }, 7500, 512), 
    ('dream_v0_7b_base-o512_s512-ntk5-8k', {'scaling_factor': 5}, {'steps': 512, }, 7500, 512), 
    ('dream_v0_7b_chat-o512_s512-8k', {}, {'steps': 512, }, 7500, 512), 
    ('dream_v0_7b_chat-o512_s512-ntk5-8k', {'scaling_factor': 5}, {'steps': 512, }, 7500, 512), 

    # 4k
    ## llama series

    ('llama_3_8b_base-o512-4k', {}, {}, 3500, 512), 
    ('llama_3_8b_chat-o512-4k', {}, {}, 3500, 512), 

    ## llada series

    ('llada_8b_base-o512_b64_s512-4k', {}, {'steps': 512, 'block_length': 64, }, 3500, 512), 
    ('llada_8b_base-o512_b64_s512-ntk4-4k', {'scaling_factor': 4}, {'steps': 512, 'block_length': 64, }, 3500, 512), 
    ('llada_8b_chat-o512_b64_s512-4k', {}, {'steps': 512, 'block_length': 64, }, 3500, 512), 
    ('llada_8b_chat-o512_b64_s512-ntk4-4k', {'scaling_factor': 4}, {'steps': 512, 'block_length': 64, }, 3500, 512), 
    ('llada_1_5_8b-o512_b64_s512-4k', {}, {'steps': 512, 'block_length': 64, }, 3500, 512), 
    ('llada_1_5_8b-o512_b64_s512-ntk4-4k', {'scaling_factor': 4}, {'steps': 512, 'block_length': 64, }, 3500, 512), 

    ## dream series

    ('dream_v0_7b_base-o512_s512-4k', {}, {'steps': 512, }, 3500, 512), 
    ('dream_v0_7b_base-o512_s512-ntk5-4k', {'scaling_factor': 5}, {'steps': 512, }, 3500, 512), 
    ('dream_v0_7b_chat-o512_s512-4k', {}, {'steps': 512, }, 3500, 512), 
    ('dream_v0_7b_chat-o512_s512-ntk5-4k', {'scaling_factor': 5}, {'steps': 512, }, 3500, 512), 

]

models = [
    dict(
        type=LLaDACausalLM, abbr=abbr, path=path_dict[abbr.split('-')[0]], drop_middle=True, 
        scaling_config=scaling_config, diffusion_config=diffusion_config, seed=2025, model_type=abbr.split('_')[0],
        model_kwargs={'flash_attention': True}, max_out_len=max_out_len, batch_size=1, max_seq_len=max_seq_len,
        run_cfg=dict(num_gpus=num_gpus[abbr.split('-')[0]], num_procs=num_gpus[abbr.split('-')[0]]),
    ) for abbr, scaling_config, diffusion_config, max_seq_len, max_out_len in models
]

work_dir = './outputs/llada_long/'

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000, gen_task_coef=15),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=32, retry=2, 
        task=dict(type=OpenICLEvalTask, dump_details=True),
    ),
)

# python run.py eval/eval_llada_long.py --dump-eval-details -r
# python run.py eval/eval_llada_long.py --dump-eval-details -r --debug
