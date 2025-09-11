import os
import json
from tools import load_dataset,load_input_ids,load_model, load_warm_up_input
import argparse
import torch
import time
from tqdm import tqdm
from transformers import AutoModelForCausalLM,AutoTokenizer
import os

# os.environ["CUDA_VISIBLE_DEVICES"]="6"

torch.cuda.reset_peak_memory_stats()


assistant_model = AutoModelForCausalLM.from_pretrained(
    "Qwen3-0.6B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager"
)


def main(args):
    if args.vanilla:
        print('use autogressive')
    else:
        print('use EAGLE3')
    model = load_model(args.base_model_path,args.ea_model_path,args.top_k,args.depth,args.total_token,args.use_tree, args.use_sps)
    if args.use_sps:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=False)
    else:
        tokenizer = model.tokenizer
    datas, origin_datas = load_dataset(args.dataset_name,args.use_all_data)

    # warmup
    for i in range(3):
        if args.use_sps:
            output_ids = model.generate(
                load_warm_up_input(tokenizer,args.model_type),
                do_sample=False,
                max_new_tokens=512,
                assistant_model=assistant_model,
            )
        elif args.vanilla:
            # input_ids = load_input_ids(datas[0], tokenizer, args.model_type, args.dataset_name, args.few_shots)
            output_ids, new_token, idx = model.naivegenerate(
                            load_warm_up_input(tokenizer,args.model_type),
                            temperature=0.0,
                            log=True,
                            is_llama3=args.model_type == "llama3",
                            max_new_tokens=40,
                        )
        else:
            # input_ids = load_input_ids(datas[0], tokenizer, args.model_type, args.dataset_name, args.few_shots)
            output_ids, new_token, idx, accept_len = model.eagenerate(
                                load_warm_up_input(tokenizer,args.model_type),
                                temperature=0.0,
                                log=True,
                                is_llama3=args.model_type == "llama3",
                                max_new_tokens=40,
                            )


    all_accept_len = 0.0
    all_generate_tokens = 0
    all_time = 0
    print('handle data:',len(datas))
    with open(args.output_path,'w') as f:
        data_nums = 0
        for data in tqdm(datas):
            input_ids = load_input_ids(data, tokenizer, args.model_type, args.dataset_name, args.few_shots)
            print('Question Length:',input_ids.shape)
            if args.use_sps:
                torch.cuda.synchronize()
                start_time = time.time()
                output_ids = model.generate(
                    torch.as_tensor(input_ids).cuda(),
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    assistant_model=assistant_model,
                )
                torch.cuda.synchronize()
                all_time += time.time() - start_time
                output_tokens = output_ids[0][len(input_ids[0]):]
                all_generate_tokens += output_tokens.shape[0]
            elif args.vanilla:
                torch.cuda.synchronize()
                start_time = time.time()
                output_ids, new_token, idx = model.naivegenerate(
                        torch.as_tensor(input_ids).cuda(),
                        temperature=args.temperature,
                        log=True,
                        is_llama3=args.model_type == "llama3",
                        max_new_tokens=args.max_new_tokens,
                    )
                torch.cuda.synchronize()
                all_time += time.time() - start_time
                output_tokens = output_ids[0][len(input_ids[0]):]
                all_generate_tokens += output_tokens.shape[0]
            else:
                torch.cuda.synchronize()
                start_time = time.time()
                output_ids, new_token, idx, accept_len = model.eagenerate(
                        torch.as_tensor(input_ids).cuda(),
                        temperature=args.temperature,
                        log=True,
                        is_llama3=args.model_type == "llama3",
                        max_new_tokens=args.max_new_tokens,
                    )
                torch.cuda.synchronize()
                output_tokens = output_ids[0][len(input_ids[0]):]
                if accept_len>0.85:
                    all_time += time.time() - start_time
                    all_accept_len += accept_len
                    all_generate_tokens += output_tokens.shape[0]
                    data_nums += 1
                
            origin_datas[data]['prediction'] = tokenizer.decode(output_tokens, skip_special_tokens=True)
            f.write(json.dumps(
                origin_datas[data]
            ) + "\n")
    if data_nums==0:
        data_nums=len(datas)
    print('AVG accept length:', all_accept_len/data_nums,'  AVG cost time:', all_time/data_nums, '  AVG generated tokens:', all_generate_tokens/data_nums)
    peak_memory = torch.cuda.max_memory_allocated()
    print(f"Max GPU used: {peak_memory / 1024**2:.2f} MB")




if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--base_model_path", type=str, default="Qwen3-8B")
    argparser.add_argument("--ea_model_path", type=str, default="qwen3-8b-eagle3")
    argparser.add_argument("--dataset_name", type=str, default="humaneval") # can choose gsm8k，humaneval, alpaca, sum, qa, longbench, repobench
    argparser.add_argument("--model_type", type=str, default="qwen3") # can choose qwen2，qwen3，llama3
    argparser.add_argument("--few_shots", action="store_true", default=False)
    argparser.add_argument("--vanilla",action="store_true", default=False) # use eagle or vanilla auto-regressive
    argparser.add_argument("--depth",type=int, default=3)
    argparser.add_argument("--top_k",type=int, default=1)
    argparser.add_argument("--total_token",type=int, default=4)
    argparser.add_argument("--temperature",type=float, default=0.0)
    argparser.add_argument("--use_tree",action="store_true", default=False) # use tree attention or not
    argparser.add_argument("--use_sps",action="store_true", default=False)  # use transformers'sps or not
    argparser.add_argument("--use_all_data",action="store_true", default=False) # use all the data of dataset
    argparser.add_argument("--max-new-tokens",type=int, default=4096)
    argparser.add_argument("--output_path",type=str,default=None)
    args = argparser.parse_args()

    main(args)

