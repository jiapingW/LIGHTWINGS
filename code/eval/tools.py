import json
import os
import torch


def load_model(base_model_path,ea_model_path,topk,depth,total_token,use_tree,use_sps):
    if use_sps:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
            attn_implementation="eager",
        )
        model.eval()
        return model
    if use_tree:
        from eagle.model.ea_model import EaModel
        model = EaModel.from_pretrained(
            base_model_path=base_model_path,
            ea_model_path=ea_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
            depth=depth,
            top_k=topk,
            total_token=total_token,
        )
    else:
        from eagle.model.ea_model_copy import EaModel
        model = EaModel.from_pretrained(
            base_model_path=base_model_path,
            ea_model_path=ea_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
            depth=depth,
            top_k=topk,
            total_token=total_token,
        )
        model.eval()
    return model



def load_dataset(dataset_name, use_all_data=False):
    datas = []
    origin_datas = {}
    if dataset_name=='gsm8k':
        if use_all_data:
            dataset_path = 'gsm8k.jsonl'
        else:
            dataset_path = 'gsm8k/question.jsonl'
    elif dataset_name=='humaneval':
        if use_all_data:
            dataset_path = 'humaneval.jsonl'
        else:
            dataset_path = 'humaneval/question.jsonl'
    elif dataset_name=='alpaca':
        dataset_path = 'alpaca/question.jsonl'
    elif dataset_name=='sum':
        if use_all_data:
            dataset_path = 'xsum.jsonl'
        else:
            dataset_path = 'xsum/question.jsonl'
    elif dataset_name=='qa':
        dataset_path = 'qa/question.jsonl'
    elif dataset_name=='longbench':
        dataset_path ='LongBench/hotpotqa_e_all.jsonl'
    elif dataset_name=='repobench':
        dataset_path ='LongBench/repobench-p_all.jsonl'
    else:
        raise ValueError('Invalid dataset name')
    if not use_all_data:
        with open(dataset_path, 'r') as f:
            for line in f:
                datas.append(json.loads(line)['turns'][0])
                origin_datas[json.loads(line)['turns'][0]] = json.loads(line)
    else: 
        try:
            with open(dataset_path, 'r') as f:
                for id,line in enumerate(f):
                    if id>2000:
                        break
                    if dataset_name=='gsm8k':
                        datas.append(json.loads(line)['question'])
                        origin_datas[json.loads(line)['question']] = json.loads(line)
                    elif dataset_name=='humaneval':
                        datas.append(json.loads(line)['prompt'])
                        origin_datas[json.loads(line)['prompt']] = json.loads(line)
                    elif dataset_name=='sum':
                        datas.append(json.loads(line)['document'])
                        origin_datas[json.loads(line)['document']] = json.loads(line)
                    else:
                        raise ValueError('dataset_name is not supported all data')
        except:
            print('Error:dataset file format is not match {}'.format(dataset_path))
            exit()


    return datas, origin_datas


def load_input_ids(data, tokenizer,model_type,dataset_name,few_shots=True):
    if model_type=='qwen2':
        if dataset_name=='gsm8k':
            if few_shots:
                msg = [
                        {'role':"system","content":"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                        dict(role='user', content="Question: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?\nLet's think step by step\nAnswer:"),
                        dict(role='assistant', content='Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.\nFor the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.\nAngelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.\nHowever, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.\nThey also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.\nAnd they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.\nSo Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.\nThey want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75\nThey will need to plan to study 4 days to allow for all the time they need.\nThe answer is 4\n'),
                        dict(role='user', content="Question: Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?\nLet's think step by step\nAnswer:"),
                        dict(role='assistant', content="Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in 2 pointers.\nHis team also scores 6 3 pointers, meaning they scored 8*3= 24 points in 3 pointers\nThey scored 10 free throws, and free throws count as one point so they scored 10*1=10 points in free throws.\nAll together his team scored 50+24+10= 84 points\nMark's opponents scored double his team's number of 2 pointers, meaning they scored 50*2=100 points in 2 pointers.\nHis opponents scored half his team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 pointers.\nThey also scored half Mark's team's points in free throws, meaning they scored 10/2=5 points in free throws.\nAll together Mark's opponents scored 100+12+5=117 points\nThe total score for the game is both team's scores added together, so it is 84+117=201 points\nThe answer is 201\n"),
                        dict(role='user', content="Question: Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?\nLet's think step by step\nAnswer:"),
                        dict(role='assistant', content="When Bella buys 2/5 times more marbles, she'll have increased the number of marbles by 2/5*60 = 24\nThe total number of marbles she'll have is 60+24 = 84\nIf Bella currently has 60 marbles, and she has two times as many marbles as frisbees, she has 60/2 = 30 frisbees.\nIf Bella buys 2/5 times more frisbees, she'll have 2/5*30 = 12 more frisbees.\nThe total number of frisbees she'll have will increase to 30+12 = 42\nBella also has 20 more frisbees than deck cards, meaning she has 30-20 = 10 deck cards\nIf she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck cards.\nThe total number of deck cards she'll have is 10+4 = 14\nTogether, Bella will have a total of 14+42+84 = 140 items\nThe answer is 140\n"),
                        dict(role='user', content="Question: A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?\nLet's think step by step\nAnswer:"),
                        dict(role='assistant', content='For the first three baskets, the number of apples and oranges in one basket is 9+15=24\nIn total, together with bananas, the number of fruits in one basket is 24+14=38 for the first three baskets.\nSince there are three baskets each having 38 fruits, there are 3*38=114 fruits in the first three baskets.\nThe number of apples in the fourth basket is 9-2=7\nThere are also 15-2=13 oranges in the fourth basket\nThe combined number of oranges and apples in the fourth basket is 13+7=20\nThe fourth basket also contains 14-2=12 bananas.\nIn total, the fourth basket has 20+12=32 fruits.\nThe four baskets together have 32+114=146 fruits.\nThe answer is 146\n'),
                        {'role':'user',"content":f"Question: {data}\nLet's think step by step\nAnswer:"}
                ]
            else:
                msg = [
                    {'role':"system","content":"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                    {'role':'user',"content":f"Question: {data}\nLet's think step by step\nAnswer:"}
                ]
            input_ids = tokenizer.apply_chat_template(
                msg,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt',
            ).cuda()
            return input_ids
        elif dataset_name=='humaneval':
            msg = [
                    {'role':"system","content":"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                    {'role':'user',"content":f"Complete the following python code:\n{data}"}
                ]
            input_ids = tokenizer.apply_chat_template(
                msg,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt',
            ).cuda()
            return input_ids
        elif dataset_name=='sum':
            msg = [
                    {'role':"system","content":"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                    {'role':'user',"content":f"{data}\nBased on the previous text, provide a brief single summary:"}
                ]
            input_ids = tokenizer.apply_chat_template(
                msg,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt',
            ).cuda()
            return input_ids
        elif dataset_name=='qa':
            msg = [
                    {'role':"system","content":"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                    {'role':'user',"content":f"{data}"}
                ]
            input_ids = tokenizer.apply_chat_template(
                msg,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt',
            ).cuda()
            return input_ids
        elif dataset_name=='alpaca':
            msg = [
                    {'role':"system","content":"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                    {'role':'user',"content":f"There are various questions below. Please answer them accurately according to the content of the questions. Do not deviate from the topic or answer irrelevant content. Question:\nQuestion:{data}"}
                ]
            input_ids = tokenizer.apply_chat_template(
                msg,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt',
            ).cuda()
            return input_ids
        elif dataset_name=='longbench' or dataset_name=='repobench':
            msg = [
                    {'role':"system","content":"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                    {'role':'user',"content":data}
                ]
            input_ids = tokenizer.apply_chat_template(
                msg,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt',
            ).cuda()
            return input_ids
        else:
            raise Exception("unsupported dataset")
    elif model_type=='qwen3':
        if dataset_name=='gsm8k':
            if few_shots:
                msg = [
                        {'role':"system","content":"You are a helpful assistant."},
                        dict(role='user', content="Question: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?\nLet's think step by step\nAnswer:"),
                        dict(role='assistant', content='Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.\nFor the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.\nAngelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.\nHowever, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.\nThey also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.\nAnd they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.\nSo Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.\nThey want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75\nThey will need to plan to study 4 days to allow for all the time they need.\nThe answer is 4\n'),
                        dict(role='user', content="Question: Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?\nLet's think step by step\nAnswer:"),
                        dict(role='assistant', content="Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in 2 pointers.\nHis team also scores 6 3 pointers, meaning they scored 8*3= 24 points in 3 pointers\nThey scored 10 free throws, and free throws count as one point so they scored 10*1=10 points in free throws.\nAll together his team scored 50+24+10= 84 points\nMark's opponents scored double his team's number of 2 pointers, meaning they scored 50*2=100 points in 2 pointers.\nHis opponents scored half his team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 pointers.\nThey also scored half Mark's team's points in free throws, meaning they scored 10/2=5 points in free throws.\nAll together Mark's opponents scored 100+12+5=117 points\nThe total score for the game is both team's scores added together, so it is 84+117=201 points\nThe answer is 201\n"),
                        dict(role='user', content="Question: Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?\nLet's think step by step\nAnswer:"),
                        dict(role='assistant', content="When Bella buys 2/5 times more marbles, she'll have increased the number of marbles by 2/5*60 = 24\nThe total number of marbles she'll have is 60+24 = 84\nIf Bella currently has 60 marbles, and she has two times as many marbles as frisbees, she has 60/2 = 30 frisbees.\nIf Bella buys 2/5 times more frisbees, she'll have 2/5*30 = 12 more frisbees.\nThe total number of frisbees she'll have will increase to 30+12 = 42\nBella also has 20 more frisbees than deck cards, meaning she has 30-20 = 10 deck cards\nIf she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck cards.\nThe total number of deck cards she'll have is 10+4 = 14\nTogether, Bella will have a total of 14+42+84 = 140 items\nThe answer is 140\n"),
                        dict(role='user', content="Question: A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?\nLet's think step by step\nAnswer:"),
                        dict(role='assistant', content='For the first three baskets, the number of apples and oranges in one basket is 9+15=24\nIn total, together with bananas, the number of fruits in one basket is 24+14=38 for the first three baskets.\nSince there are three baskets each having 38 fruits, there are 3*38=114 fruits in the first three baskets.\nThe number of apples in the fourth basket is 9-2=7\nThere are also 15-2=13 oranges in the fourth basket\nThe combined number of oranges and apples in the fourth basket is 13+7=20\nThe fourth basket also contains 14-2=12 bananas.\nIn total, the fourth basket has 20+12=32 fruits.\nThe four baskets together have 32+114=146 fruits.\nThe answer is 146\n'),
                        {'role':'user',"content":f"Question: {data}\nLet's think step by step\nAnswer:"}
                ]
            else:
                msg = [
                    {'role':"system","content":"You are a helpful assistant."},
                    {'role':'user',"content":f"Question: {data}\nLet's think step by step\nAnswer:"}
                ]
            input_ids = tokenizer.apply_chat_template(
                msg,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt',
                enable_thinking=False,
            ).cuda()
            return input_ids
        elif dataset_name=='humaneval':
            msg = [
                    {'role':"system","content":"You are a helpful assistant."},
                    {'role':'user',"content":f"Complete the following python code:\n{data}"}
                ]
            input_ids = tokenizer.apply_chat_template(
                msg,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt',
                enable_thinking=False,
            ).cuda()
            return input_ids
        elif dataset_name=='sum':
            msg = [
                    {'role':"system","content":"You are a helpful assistant."},
                    {'role':'user',"content":f"{data}\nBased on the previous text, provide a brief single summary:"}
                ]
            input_ids = tokenizer.apply_chat_template(
                msg,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt',
                enable_thinking=False,
            ).cuda()
            return input_ids
        elif dataset_name=='qa':
            msg = [
                    {'role':"system","content":"You are a helpful assistant."},
                    {'role':'user',"content":f"{data}"}
                ]
            input_ids = tokenizer.apply_chat_template(
                msg,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt',
            ).cuda()
            return input_ids
        elif dataset_name == "alpaca":
            msg = [
                    {'role':"system","content":"You are a helpful assistant."},
                    {'role':'user',"content":f"There are various questions below. Please answer them accurately according to the content of the questions. Do not deviate from the topic or answer irrelevant content. Question:\nQuestion:{data}"}
                ]
            input_ids = tokenizer.apply_chat_template(
                msg,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt',
            ).cuda()
            return input_ids
        elif dataset_name=='longbench' or dataset_name=='repobench':
            msg = [
                    {'role':"system","content":"You are a helpful assistant."},
                    {'role':'user',"content":data}
                ]
            input_ids = tokenizer.apply_chat_template(
                msg,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt',
                enable_thinking=False,
            ).cuda()
            return input_ids
        else:
            raise Exception("unsupported dataset")

    elif model_type=='llama3':
        if dataset_name=='gsm8k':
            if few_shots:
                msg = [
                        {'role':"system","content":"You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
                        dict(role='user', content="Question: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?\nLet's think step by step\nAnswer:"),
                        dict(role='assistant', content='Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.\nFor the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.\nAngelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.\nHowever, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.\nThey also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.\nAnd they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.\nSo Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.\nThey want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75\nThey will need to plan to study 4 days to allow for all the time they need.\nThe answer is 4\n'),
                        dict(role='user', content="Question: Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?\nLet's think step by step\nAnswer:"),
                        dict(role='assistant', content="Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in 2 pointers.\nHis team also scores 6 3 pointers, meaning they scored 8*3= 24 points in 3 pointers\nThey scored 10 free throws, and free throws count as one point so they scored 10*1=10 points in free throws.\nAll together his team scored 50+24+10= 84 points\nMark's opponents scored double his team's number of 2 pointers, meaning they scored 50*2=100 points in 2 pointers.\nHis opponents scored half his team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 pointers.\nThey also scored half Mark's team's points in free throws, meaning they scored 10/2=5 points in free throws.\nAll together Mark's opponents scored 100+12+5=117 points\nThe total score for the game is both team's scores added together, so it is 84+117=201 points\nThe answer is 201\n"),
                        dict(role='user', content="Question: Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?\nLet's think step by step\nAnswer:"),
                        dict(role='assistant', content="When Bella buys 2/5 times more marbles, she'll have increased the number of marbles by 2/5*60 = 24\nThe total number of marbles she'll have is 60+24 = 84\nIf Bella currently has 60 marbles, and she has two times as many marbles as frisbees, she has 60/2 = 30 frisbees.\nIf Bella buys 2/5 times more frisbees, she'll have 2/5*30 = 12 more frisbees.\nThe total number of frisbees she'll have will increase to 30+12 = 42\nBella also has 20 more frisbees than deck cards, meaning she has 30-20 = 10 deck cards\nIf she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck cards.\nThe total number of deck cards she'll have is 10+4 = 14\nTogether, Bella will have a total of 14+42+84 = 140 items\nThe answer is 140\n"),
                        dict(role='user', content="Question: A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?\nLet's think step by step\nAnswer:"),
                        dict(role='assistant', content='For the first three baskets, the number of apples and oranges in one basket is 9+15=24\nIn total, together with bananas, the number of fruits in one basket is 24+14=38 for the first three baskets.\nSince there are three baskets each having 38 fruits, there are 3*38=114 fruits in the first three baskets.\nThe number of apples in the fourth basket is 9-2=7\nThere are also 15-2=13 oranges in the fourth basket\nThe combined number of oranges and apples in the fourth basket is 13+7=20\nThe fourth basket also contains 14-2=12 bananas.\nIn total, the fourth basket has 20+12=32 fruits.\nThe four baskets together have 32+114=146 fruits.\nThe answer is 146\n'),
                        {'role':'user',"content":f"Question: {data}\nLet's think step by step\nAnswer:"}
                ]
            else:
                msg = [
                    {'role':"system","content":"You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
                    {'role':'user',"content":f"Question: {data}\nLet's think step by step\nAnswer:"}
                ]
            input_ids = tokenizer.apply_chat_template(
                msg,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt',
            ).cuda()
            return input_ids

    




def load_warm_up_input(tokenizer, model_type):
    data = 'How does the concept of torsion fields help explain the observed changes in gamma radiation during energy healing sessions? Answer according to: Recently replicated studies by Benford et al. Identified statistically significant decreases in extremely high-frequency electromagnetic fields, termed gamma rays, during alternative healing sessions with a trained practitioner and multiple subjects.4 During these experiments, external gamma radiation counts were recorded to obtain: (1) background control values, (2) resting control values (subjects in supine position), (3) no treatment baseline control values (with a person standing along side the subjects), and (4) treatment values (with therapist’s hands gently placed on the subjects). During the initial preliminary tests, total counts were recorded in 100-second trials separately over the crown, heart, abdomen and pelvic regions using a NaI(TI) crystal scintillator that detected gamma radiation from approximately 100 KeV to 3 MeV. Later tests involved 300-second counts over the heart, abdomen, right lateral side of the subject.\nEarly results demonstrated that gamma radiation levels markedly decreased during therapy sessions of 100% of subjects and at every body site tested regardless of which therapist performed the treatment. In many instances, the gamma counts fluctuated by thousands which represented a 3 – 6% difference in gross gamma count between the untrained control practitioners and trained healing energy practitioners within the short time periods analyzed. T-tests were used to determine that there was a significant statistical significance with p-values between 0.035 and <0.0001 in the 100-second trials, and p-values less than 0.0001 in the 300-second trials.\nThe consistent results from these controlled experiments may result, in part, from the increased electromagnetic fields surrounding the hands of healers. Dr. John Zimmerman discovered large frequency-pulsing biomagnetic fields emanating from hands of therapeutic touch practitioners during therapy as measured by an extremely sensitive magnetometer called a SQUID (Superconducting Quantum Interference Device).6 Similar frequency-pulsing biomagnetic fields were later measured and quantified from the hands of meditators and practitioners of yoga and QiGong using a simple magnetometer. These fields were 1000 times greater than the strongest human biomagnetic field7 and were found to be in the same frequency range as those being tested in medical research laboratories for use in speeding the healing process of certain biological tissues.3,8 This range is low energy and extremely low frequency and spans from 2 Hz to 50 Hz.8 However, still missing from this research is a mechanism of action that can explain the connection between electromagnetic field increases and observed healing benefits reported in the current therapy literature.\nIII. Adding “spin” to the healing equation.\nRecently released Russian research posits an approach for describing many, if not all, the biopositive and parapsychological phenomena related to bioenergy healing and conscious intention. This research has postulated and tested the existence of “torsion fields.” Russian scientist Akimov and his team consider the quantum vacuum as a universal torsion wave carrying medium. The torsion field is said to fill all of space isotropically, including its matter component. It has a quantal structure that is unobservable in non-disturbed states. However, violations of vacuum symmetry and invariance create different, and in principle observable, states. These theorized “violations” and “invariances” may be what the energy practitioners are creating, which induces changes in the gamma radiation previously described.'
    if model_type=='qwen2':
        msg = [
                {'role':"system","content":"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {'role':'user',"content":f"Question: {data}\nLet's think step by step\nAnswer:"}
        ]
        input_ids = tokenizer.apply_chat_template(
            msg,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors='pt',
        ).cuda()
        return input_ids
    elif model_type=='qwen3':
        msg = [
                {'role':"system","content":"You are a helpful assistant."},
                {'role':'user',"content":f"Question: {data}\nLet's think step by step\nAnswer:"}
        ]
        input_ids = tokenizer.apply_chat_template(
            msg,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors='pt',
        ).cuda()
        return input_ids
    else:
        raise ValueError(f'model_type {model_type} not supported')