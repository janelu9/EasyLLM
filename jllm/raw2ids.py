#!/usr/bin/env python
# coding: utf-8
# Created on Thur Jun 29 09:36:49 2023
# @author: Jian Lu
# Email:janelu@live.cn;

from functools import partial
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
import pyarrow.parquet
import numpy as np
import argparse
import json
import tqdm
import os
import gc

IGNORE_TOKEN_ID: int = -100
OVERLAPPING_LENGTH: int  = 1
SPLIT_LENGTH: int = 131072
NUM_ELEMENT_LIMIT = 2e9

def split_doc(doc,bos=True,eos=True):
    p=0;l=len(doc)
    while p<l:
        yield p == 0 and bos,doc[p:p+SPLIT_LENGTH],(p+SPLIT_LENGTH) >= l and eos
        p += SPLIT_LENGTH - 1

def pretrain_generator(file,sep="\n\n"):
    with open(file,"r") as f:
        doc=""
        line = f.readline()
        try:
            while line:
                doc = json.loads(line.strip())['text']
                for block in split_doc(doc):
                    yield block
                line = f.readline()  
        except:
            bos=True
            while line:
                doc+=line
                if doc[-2:]==sep:
                    for block in split_doc(doc[:-1],bos):
                        yield block
                    doc=""
                    bos=True
                elif len(doc)>=SPLIT_LENGTH:
                    for block in split_doc(doc,bos,False):
                        yield block
                    doc=""
                    bos=False
                line = f.readline()  
            if doc:
                for block in split_doc(doc,bos):
                    yield block
            
def token_pretrain(file,tokenizer,MAX_SEQ_LENGTH,stack=True):
    if stack:
        ids = []
        cu_seqlens = [0]
        for bos,doc,eos in pretrain_generator(file):
            if bos:
                if len(ids)>0:cu_seqlens.append(len(ids))
                ids.append(tokenizer.bos_token_id)
            ids.extend(tokenizer.encode(doc))
            if eos:
                ids.append(tokenizer.eos_token_id)
            p = 0
            n = len(ids)
            while p<n-OVERLAPPING_LENGTH:
                input_ids=ids[p:p+MAX_SEQ_LENGTH]
                l = len(input_ids)
                if l==MAX_SEQ_LENGTH:
                    cu_seqlens.append(MAX_SEQ_LENGTH-1)
                    yield {'input_ids':np.array(input_ids, dtype=np.int32),'cu_seqlens':np.array(cu_seqlens, dtype=np.int32)}
                    cu_seqlens = [0]
                p += MAX_SEQ_LENGTH-OVERLAPPING_LENGTH
                
            if l==MAX_SEQ_LENGTH: # all ids yielded, clear ids
                ids = []   
            elif l>=MAX_SEQ_LENGTH-3: # pad few then yield  
                input_ids.extend([tokenizer.pad_token_id]*(MAX_SEQ_LENGTH-l))
                cu_seqlens.append(MAX_SEQ_LENGTH-1)
                yield {'input_ids':np.array(input_ids, dtype=np.int32),'cu_seqlens':np.array(cu_seqlens, dtype=np.int32)}
                cu_seqlens = [0]
                ids = []
            else: # join to next doc
                ids=input_ids
 
        if 1<l<MAX_SEQ_LENGTH:
            cu_seqlens.append(MAX_SEQ_LENGTH-1)
            input_ids.extend([tokenizer.pad_token_id]*(MAX_SEQ_LENGTH-l))
            yield {'input_ids':np.array(input_ids, dtype=np.int32),'cu_seqlens':np.array(cu_seqlens, dtype=np.int32)}
    else:
        for bos,doc,eos in pretrain_generator(file):
            ids = []
            if bos:
                ids.append(tokenizer.bos_token_id)
            ids.extend(tokenizer.encode(doc))
            if eos:
                ids.append(tokenizer.eos_token_id)
            p = 0
            n = len(ids)
            while p<n-OVERLAPPING_LENGTH:
                input_ids=ids[p:p+MAX_SEQ_LENGTH]
                l = len(input_ids)
                if l==MAX_SEQ_LENGTH:
                    yield {'input_ids':np.array(input_ids, dtype=np.int32)}
                p += MAX_SEQ_LENGTH-OVERLAPPING_LENGTH
                
            if 1<l<MAX_SEQ_LENGTH:
                input_ids.extend([tokenizer.pad_token_id]*(MAX_SEQ_LENGTH-l))
                yield {'input_ids':np.array(input_ids, dtype=np.int32)}
        
def finetune_generator(file):
    with open(file,"r") as f:
        line = f.readline()
        while line:
            yield line
            line = f.readline()

def token_finetune(file,tokenizer,MAX_SEQ_LENGTH,ROLE = {},PREFIX = [],ADAPT = []):
    from jllm.data.utils import qa_inputs_generator
    for sample in finetune_generator(file):
        js = json.loads(sample.strip())
        pmt_anses = js['conversation'] if 'conversation' in js else js
        if len(pmt_anses) > 1:
            msgs = (PREFIX + pmt_anses) if 'system' not in pmt_anses[0] else pmt_anses
            ids = []; divide = [0]; 
            for start,msg in enumerate(msgs):
                k,v = next(iter(msg.items()))
                if k != "assistant":
                    ids.extend(ROLE[k] if k == 'system' or len(ROLE[k])==1 else ROLE[k][1:])
                    ids.extend(tokenizer.encode(v))
                    pre_k = k
                    break
                    
            if k != 'system':
                ids = ADAPT + ids
            
            for msg in msgs[start+1:]:
                k,v = next(iter(msg.items()))
                if k != pre_k:
                    if k != "assistant":
                        ids.append(tokenizer.im_end_id)
                        if pre_k != "system":
                            divide.append(len(ids))
                    ids.extend(ROLE[k])
                    if k == "assistant":
                        divide.append(len(ids))
                    ids.extend(tokenizer.encode(v))         
                else:
                    ids.extend(tokenizer.encode(v))
                pre_k = k

            if k == "assistant":
                ids.append(tokenizer.im_end_id)

            if len(divide)%2==1:
                ids=ids[:divide[-1]]
            else:
                divide.append(len(ids))
                
            if len(divide)>2 :
                for qa_inputs,_ in qa_inputs_generator(ids,
                                                       divide,
                                                       MAX_SEQ_LENGTH,
                                                       MAX_HISTORY_LENGTH = MAX_SEQ_LENGTH//2,
                                                       pad_token_id = tokenizer.pad_token_id,
                                                       IGNORE_TOKEN_ID = -100):
                    if 'category' in js:
                        qa_inputs.update({"prompt_len":divide[1],"classes":int(js['category'])})
                    yield qa_inputs
   
def write_parquet(filename,output_dir,tokenizer,MAX_SEQ_LENGTH=2048,dtype='ft',batch_size=2**15,compression='gzip',stack=False,max_num=1,image_path='',sep=False):
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer,use_fast=True,trust_remote_code=True,add_bos_token = False)
    tokenizer_class = tokenizer.__class__.__name__ 
    tokenizer,ROLE,PREFIX,ADAPT = TOKENIZER[tokenizer_class](tokenizer)
    auto_batch_size = False
    if dtype == 'ft':
        if not hasattr(tokenizer,'get_image_tokens'):
            token = partial(token_finetune, ROLE=ROLE, PREFIX=PREFIX, ADAPT=ADAPT)
        else:
            token = partial(token_vl, ROLE=ROLE, PREFIX=PREFIX, ADAPT=ADAPT, 
                            img_reader=ImageReaderCV2(max_num=max_num),
                            image_path=image_path,
                            sep = output_dir if sep else None
                            )
            auto_batch_size = True if not sep else False
    else:
        token = partial(token_pretrain,stack = stack)
    item_iter = token(filename,tokenizer,MAX_SEQ_LENGTH)
    file = os.path.splitext(os.path.basename(filename))[0]
    partition_dir = os.path.join(output_dir , file)
    partition_file = os.path.join(partition_dir , f"{file}-%05d.{compression}.parquet")
    check_file = os.path.join(output_dir , "." + file + ".crc")
    if os.path.exists(check_file):
        print(f"{filename} converted, continue!")
        return
    if not os.path.exists(partition_dir):
        os.makedirs(partition_dir)

    if not auto_batch_size:
        pbar = tqdm.tqdm(float("inf"))
        data = next(item_iter)
        data_batch={k:[data[k]] for k in data}
        i=0
        pbar.update(1)
        try:
            while True:
                for _ in range(batch_size-1):
                    data = next(item_iter)
                    for k in data:data_batch[k].append(data[k])
                    i+=1
                    pbar.update(1)
                    
                pyarrow.parquet.write_table(pyarrow.table(data_batch),
                                            partition_file % (i//batch_size), 
                                            compression=compression)
                del data_batch
                gc.collect()
                data_batch = None
                
                data = next(item_iter)
                data_batch={k:[data[k]] for k in data}
                i+=1
                pbar.update(1)
                
        except StopIteration:
            pbar.close()
                
        if data_batch :
            pyarrow.parquet.write_table(pyarrow.table(data_batch),
                                        partition_file % (i//batch_size), 
                                        compression=compression)
    else:
        pbar = tqdm.tqdm(float("inf"))
        data = next(item_iter)
        data_batch={k:[data[k]] for k in data}
        element_num = sum(data[k].size for k in data)
        i=0
        p=0
        pbar.update(1)
        for data in item_iter:
            i+=1
            pbar.update(1)
            for k in data: 
                data_batch[k].append(data[k])
                element_num+=data[k].size 
            if element_num > NUM_ELEMENT_LIMIT:
                pyarrow.parquet.write_table(pyarrow.table(data_batch),
                                            partition_file % (p),
                                            compression=compression)
                p+=1
                del data_batch
                gc.collect()
                data_batch={k:[] for k in data}
                element_num = 0
        pbar.close()       
        if data_batch[k] :
            pyarrow.parquet.write_table(pyarrow.table(data_batch),
                                        partition_file % (p), 
                                        compression=compression) 

    del data_batch                                
    os.system(f"echo '{i+1} {MAX_SEQ_LENGTH} {batch_size} {len(data.keys())}' > {check_file}")
    print(f"{filename} stored in parquet with {i+1} samples")
    gc.collect()

def token_vl(file,tokenizer,MAX_SEQ_LENGTH,ROLE = {},PREFIX = [],ADAPT = []
             ,img_reader=None,image_path='',sep = None):

    from jllm.data.utils import qa_inputs_generator,img_token_alignment
    
    if sep:
        images_database = {}
        def replace_image_token(v):
            if not isinstance(v,str):
                v,*imgs = v
                pixes = []
                for img in imgs:
                    try:
                        if img not in images_database:
                            image = cv2.imread(os.path.join(image_path,img))
                            target_aspect_ratio,_ = img_reader.find_closest_aspect_ratio(image)
                            num_patches = target_aspect_ratio[0]*target_aspect_ratio[1]
                            num_patches = num_patches + int(num_patches>1)
                            # key =len(images_database)
                            image_tokens = tokenizer.get_image_tokens(num_patches)
                            images_database[img] = (image_tokens,target_aspect_ratio)
                        else:
                            image_tokens,_ = images_database[img]
                        v=v.replace('<image>', image_tokens,1)
                        pixes.append(img)
                    except:
                        v=v.replace('<image>', '<图片>',1)
                        pixes.append('')
                return v,pixes if len(pixes) else ['']
            return v,['']
    else:
        def replace_image_token(v):
            if not isinstance(v,str):
                v,*imgs = v
                pixes = []
                for img in imgs:
                    try:
                        pix_v = img_reader(os.path.join(image_path,img))
                        image_tokens = tokenizer.get_image_tokens(pix_v.shape[0])
                        v=v.replace('<image>', image_tokens,1)
                        pixes.append(pix_v.reshape(-1))
                    except:
                        v=v.replace('<image>', '<图片>',1)
                return v,pixes if pixes else [np.empty(0,dtype=np.uint8)]
            return v,[np.empty(0,dtype=np.uint8)]

    for sample in finetune_generator(file):
        js = json.loads(sample.strip())
        pmt_anses = js['conversation'] if 'conversation' in js else js
        if len(pmt_anses) > 1:
            msgs = (PREFIX + pmt_anses) if 'system' not in pmt_anses[0] else pmt_anses
            ids = []; divide = [0]; pixes = []
            for start,msg in enumerate(msgs):
                k,v = next(iter(msg.items()))
                v,p = replace_image_token(v)
                if k != "assistant":
                    ids.extend(ROLE[k] if k == 'system' or len(ROLE[k])==1 else ROLE[k][1:])
                    ids.extend(tokenizer.encode(v))
                    pre_k = k
                    break
                    
            if k != 'system':
                ids = ADAPT + ids
                pixes.append(p)
            
            for msg in msgs[start+1:]:
                k,v = next(iter(msg.items()))
                v,p = replace_image_token(v)
                if k != pre_k:
                    if k != "assistant":
                        pixes.append(p)
                        ids.append(tokenizer.im_end_id)
                        if pre_k != "system":
                            divide.append(len(ids))
                    ids.extend(ROLE[k])
                    if k == "assistant":
                        divide.append(len(ids))
                    ids.extend(tokenizer.encode(v))         
                else:
                    ids.extend(tokenizer.encode(v))
                pre_k = k

            if k == "assistant":
                ids.append(tokenizer.im_end_id)

            if len(divide)%2==1:
                ids=ids[:divide[-1]]
            else:
                divide.append(len(ids))
                
            if len(divide)>2 :
                for qa_inputs,sub_divide in qa_inputs_generator(ids,
                                                                divide,
                                                                MAX_SEQ_LENGTH,
                                                                MAX_HISTORY_LENGTH = MAX_SEQ_LENGTH//2,
                                                                pad_token_id = tokenizer.pad_token_id,
                                                                IGNORE_TOKEN_ID = -100):
                    s = sub_divide[0]
                    e = sub_divide[-2] if len(sub_divide)%2 else sub_divide[-1]
                    matched = None
                    if s == divide[0] and e == divide[-2]:
                        matched = True
                        si,ei=0,0
                    else:
                        si,ei = np.searchsorted(divide,[s,e],side='right')
                        si = (si-1)//2
                        ei = (ei-1)//2+1
                        
                    if not sep:
                        qa_vl_inputs = img_token_alignment((tokenizer.img_bos_token_id,tokenizer.img_eos_token_id),qa_inputs,pixes,matched,(si,ei))
                        if qa_vl_inputs is not None:
                            qa_vl_inputs['images'] = np.hstack(qa_vl_inputs['images'])
                            yield qa_vl_inputs
                    else:
                        qa_inputs.update({'image_ids':pixes,'matched':matched,'siei':(si,ei)})
                        yield qa_inputs
    if sep:
        partition_file = os.path.join(sep , "image.info")
        data = {'pic':[],'rat':[]}
        for k,v in images_database.items():
            data['pic'].append(k)
            data['rat'].append(v[1])
        data['pic'].append(os.path.abspath(image_path))
        data['rat'].append(np.array([tokenizer.img_bos_token_id,tokenizer.img_eos_token_id,img_reader.image_size]))
        pyarrow.parquet.write_table(pyarrow.table(data),partition_file)
        print(f"\nAvailable pictures: {len(data['pic'])-1}")   

def main(args):
    print(args)
    source=os.path.abspath(args.i)
    source_dir=os.path.dirname(source)
    file =os.path.basename(source)
    file_name = os.path.splitext(file)[0]
    if os.path.isfile(source):
        tmp = os.path.join(source_dir,args.tmp)
        if not os.path.exists(tmp) or args.C:
            os.system(f" rm -rf {tmp}") 
            os.makedirs(tmp)
            os.system(f"cd {tmp};split -d -a 5 -{args.n} ../{file} {file_name}-;cd -;")
    else:
        tmp = source
    output_dir = args.o if args.o !="" else os.path.join(source_dir,file_name+f"_{os.path.basename(args.tokenizer)}")
    if os.path.exists(output_dir):
        if args.C:
            os.system(f" rm -rf {output_dir}/*")
            os.system(f" rm -rf {output_dir}/.*.crc")
    else:
        os.makedirs(output_dir)
    
    Pool = ThreadPoolExecutor if args.T else ProcessPoolExecutor
    cpus=int(os.cpu_count()*0.8) if args.cores <0 else  args.cores
    print(f"########## begine converting {args.t} data with {cpus} executors.###########" )
    with Pool(max_workers=cpus) as exe:
        func = partial(write_parquet,
                       output_dir=output_dir,
                       tokenizer=args.tokenizer,
                       MAX_SEQ_LENGTH= args.seq_len,
                       dtype=args.t,
                       batch_size=args.batch_size,
                       compression=args.c.lower(),
                       stack=args.stack,
                       max_num=args.max_num,
                       image_path=args.image_path,
                       sep=args.sep)
        files =[os.path.join(tmp, i) for i in os.listdir(tmp)]
        files.sort()
        np.random.shuffle(files)
        list(exe.map(func,files))
    if tmp != source: os.system(f" rm -rf {tmp}") 
    print(f"{source} has been converted into {output_dir} successfully!")
    '''
    Example
    # head -n2 news-commentary-v13-zh-en.txt
    1929年还是1989年?   1929 or 1989?
    巴黎-随着经济危机不断加深和蔓延，整个世界一直在寻找历史上的类似事件希望有助于我们了解目前正在发生的情况。   PARIS – As the economic crisis deepens and widens, the world has been searching for historical analogies to help us understand what has been happening.
    # python raw2ids.py -i news-commentary-v13-zh-en.txt -n 30000
    Namespace(t='ft', i='news-commentary-v13-zh-en.txt', o='', n=30000, c='gzip', batch_size=32768, cores=-1, tokenizer='openlm-research/open_llama_13b', tmp='tmp', T=False, C=False)
    /mnt/e/NLP
    ########## begine converting ft data with 12 executors.###########
    12777it [00:06, 2018.59it/s]
    15178it [00:07, 1980.01it/s]/mnt/e/NLP/tmp/news-commentary-v13-zh-en-part-08 saved with 12777 samples
    30000it [00:14, 2070.87it/s]
    30000it [00:14, 2068.94it/s]
    30000it [00:14, 2064.20it/s]
    29999it [00:14, 2062.98it/s]
    30000it [00:14, 2063.26it/s]
    30000it [00:14, 2062.92it/s]
    30000it [00:14, 2056.94it/s]
    30000it [00:14, 2054.50it/s]
    /mnt/e/NLP/tmp/news-commentary-v13-zh-en-part-06 saved with 30000 samples
    /mnt/e/NLP/tmp/news-commentary-v13-zh-en-part-00 saved with 30000 samples
    /mnt/e/NLP/tmp/news-commentary-v13-zh-en-part-05 saved with 30000 samples
    /mnt/e/NLP/tmp/news-commentary-v13-zh-en-part-01 saved with 30000 samples
    /mnt/e/NLP/tmp/news-commentary-v13-zh-en-part-04 saved with 30000 samples
    /mnt/e/NLP/tmp/news-commentary-v13-zh-en-part-02 saved with 29999 samples
    /mnt/e/NLP/tmp/news-commentary-v13-zh-en-part-07 saved with 30000 samples
    /mnt/e/NLP/tmp/news-commentary-v13-zh-en-part-03 saved with 30000 samples
    /mnt/e/NLP/news-commentary-v13-zh-en.txt has been converted into /mnt/e/NLP/news-commentary-v13-zh-en_open_llama_13b successfully!
    '''

    
def llama_template(tokenizer):
    
    PREFIX,ADAPT=[],[]
    tokenizer.pad_token_id = 0
    tokenizer.im_start_id = tokenizer.encode('<|im_start|>')[0]
    tokenizer.im_end_id = tokenizer.encode('<|im_end|>')[0]
    nl_token_id = tokenizer.encode("\n")
    system_id = tokenizer.encode("system")
    user_id = tokenizer.encode("user")
    assistant_id = tokenizer.encode("assistant")
    
    ROLE = {
        'system': [tokenizer.im_start_id] + system_id + nl_token_id,
        'user': nl_token_id + [tokenizer.im_start_id]+ user_id + nl_token_id,
        'assistant': [tokenizer.im_end_id] + nl_token_id + [tokenizer.im_start_id] + assistant_id + nl_token_id
    }
    
    return tokenizer,ROLE,PREFIX,ADAPT
    
def llama3_template(tokenizer):
    
    PREFIX,ADAPT=[],[]
    tokenizer.bos_token_id = tokenizer.encode('<|begin_of_text|>')[0]
    tokenizer.im_end_id = tokenizer.encode('<|eot_id|>')[0]
    tokenizer.pad_token_id = tokenizer.encode('<|end_of_text|>')[0]
    tokenizer.eos_token_id = tokenizer.im_end_id

    start_header_id = tokenizer.encode('<|start_header_id|>')
    end_header_id = tokenizer.encode('<|end_header_id|>')
    nl_token_id = tokenizer.encode("\n\n")
    system_id = tokenizer.encode("system")
    user_id = tokenizer.encode("user")
    assistant_id = tokenizer.encode("assistant")

    ROLE = {
        'system': [tokenizer.bos_token_id]+start_header_id+system_id+end_header_id+ nl_token_id,
        'user': start_header_id+user_id+end_header_id+nl_token_id,
        'assistant': [tokenizer.im_end_id ]+start_header_id+assistant_id+end_header_id+nl_token_id
    }
    
    ADAPT = [tokenizer.bos_token_id,start_header_id[0]]
    return tokenizer,ROLE,PREFIX,ADAPT
    
def qwen_template(tokenizer): 
    
    PREFIX,ADAPT=[],[]
    tokenizer.bos_token_id = tokenizer.im_start_id
    tokenizer.eos_token_id = tokenizer.im_end_id
    tokenizer.pad_token_id = tokenizer.encode("<|endoftext|>")[0]
    nl_token_id = tokenizer.encode("\n")
    system_id = tokenizer.encode("system", allowed_special=set())
    user_id = tokenizer.encode("user", allowed_special=set())
    assistant_id = tokenizer.encode("assistant", allowed_special=set())
    
    ROLE = {
        'system': [tokenizer.im_start_id] + system_id + nl_token_id,
        'user': nl_token_id + [tokenizer.im_start_id] + user_id + nl_token_id,
        'assistant': [tokenizer.im_end_id] + nl_token_id + [tokenizer.im_start_id] + assistant_id + nl_token_id
    }
    PREFIX = [{"system":""}]
    
    return tokenizer,ROLE,PREFIX,ADAPT 
    
def qwen2_template(tokenizer): 
    
    PREFIX,ADAPT=[],[]
    tokenizer.bos_token_id = tokenizer.im_start_id = tokenizer.encode("<|im_start|>")[0]
    tokenizer.eos_token_id = tokenizer.im_end_id = tokenizer.encode("<|im_end|>")[0]
    tokenizer.pad_token_id = tokenizer.encode("<|endoftext|>")[0]
    nl_token_id = tokenizer.encode("\n")
    system_id = tokenizer.encode("system")
    user_id = tokenizer.encode("user")
    assistant_id = tokenizer.encode("assistant")
    
    ROLE = {
        'system': [tokenizer.im_start_id] + system_id + nl_token_id,
        'user': nl_token_id + [tokenizer.im_start_id] + user_id + nl_token_id,
        'assistant': [tokenizer.im_end_id] + nl_token_id + [tokenizer.im_start_id] + assistant_id + nl_token_id
    }
    PREFIX = [{"system":"You are a helpful assistant."}]
    
    return tokenizer,ROLE,PREFIX,ADAPT 

def baichcuan_template(tokenizer): 
    
    PREFIX,ADAPT=[],[]
    tokenizer.im_end_id = tokenizer.eos_token_id
    tokenizer.pad_token_id = 0
    ROLE = {
        'user':[195],
        'assistant':[196]
    }
    
    return tokenizer,ROLE,PREFIX,ADAPT 

import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = np.array(IMAGENET_MEAN)[:,None,None], np.array(IMAGENET_STD)[:,None,None]
    transform = T.Compose([
        #T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        #T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.Lambda(lambda img:np.array(img).transpose(2,0,1)),
        #T.Lambda(lambda x:(x/255-MEAN)/STD)
    ])
    return transform
    
class ImageReader:
    
    def __init__(self,min_num=1, max_num=6,input_size = 448):
        
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = np.array(sorted(target_ratios, key=lambda x: x[0] * x[1]))
        rdict={}
        for i,(w,h) in enumerate(target_ratios):
            r = w/h
            if r in rdict:
                rdict[r].append(i)
            else:
                rdict[r]=[i]
        target_ratios_rat = np.array(list(rdict.keys()))

        self.rdict = {k:np.array(v) for k,v in rdict.items()}
        self.target_ratios = target_ratios
        self.target_ratios_rat = target_ratios_rat
        self.target_sizes = target_ratios*input_size
        self.target_half_areas = self.target_sizes[:,0]*self.target_sizes[:,1]*0.5
        self.image_size = input_size
        self.transform = build_transform(input_size)

    def find_closest_aspect_ratio(self,image):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        min_index = np.argmin(np.abs(self.target_ratios_rat-aspect_ratio))
        min = self.target_ratios_rat[min_index]
        mins = self.rdict[min]
        if len(mins)==1:
            return self.target_ratios[mins[0]],self.target_sizes[mins[0]]
        idx = mins[self.target_half_areas[mins]<orig_width*orig_height]
        if len(idx)>0:
            return self.target_ratios[idx[-1]],self.target_sizes[idx[-1]]
        return self.target_ratios[mins[0]],self.target_sizes[mins[0]]
    
    def dynamic_preprocess(self,image, use_thumbnail=False):
        # find the closest aspect ratio to the target
        target_aspect_ratio,(target_width,target_height) = self.find_closest_aspect_ratio(image)
        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = [resized_img.crop((i*self.image_size,j*self.image_size,(i+1)*self.image_size,(j+1)*self.image_size)) 
                            for i in range(target_aspect_ratio[0]) for j in range(target_aspect_ratio[1])]
        #assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((self.image_size, self.image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def __call__(self,image_file):
        image = Image.open(image_file).convert('RGB')
        images = self.dynamic_preprocess(image, use_thumbnail=True)
        pixel_values = [self.transform(image) for image in images]
        pixel_values = np.stack(pixel_values)
        return pixel_values

import cv2

class ImageReaderCV2:
    
    def __init__(self,min_num=1, max_num=6,input_size = 448):
        
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = np.array(sorted(target_ratios, key=lambda x: x[0] * x[1]))
        rdict={}
        for i,(w,h) in enumerate(target_ratios):
            r = w/h
            if r in rdict:
                rdict[r].append(i)
            else:
                rdict[r]=[i]
        target_ratios_rat = np.array(list(rdict.keys()))

        self.rdict = {k:np.array(v) for k,v in rdict.items()}
        self.target_ratios = target_ratios
        self.target_ratios_rat = target_ratios_rat
        self.target_sizes = target_ratios*input_size
        self.target_half_areas = self.target_sizes[:,0]*self.target_sizes[:,1]*0.5
        self.image_size = input_size

    def find_closest_aspect_ratio(self,image):
        orig_height, orig_width = image.shape[:2]
        aspect_ratio = orig_width / orig_height
        min_index = np.argmin(np.abs(self.target_ratios_rat-aspect_ratio))
        min = self.target_ratios_rat[min_index]
        mins = self.rdict[min]
        if len(mins)==1:
            return self.target_ratios[mins[0]],self.target_sizes[mins[0]]
        idx = mins[self.target_half_areas[mins]<orig_width*orig_height]
        if len(idx)>0:
            return self.target_ratios[idx[-1]],self.target_sizes[idx[-1]]
        return self.target_ratios[mins[0]],self.target_sizes[mins[0]]
    
    def dynamic_preprocess(self,image, use_thumbnail=False):
        # find the closest aspect ratio to the target
        target_aspect_ratio,(target_width,target_height) = self.find_closest_aspect_ratio(image)
        # resize the image
        resized_img = cv2.resize(image, (target_width,target_height), interpolation=cv2.INTER_AREA)
        processed_images = [resized_img[j*self.image_size:(j+1)*self.image_size,i*self.image_size:(i+1)*self.image_size,:]
                            for i in range(target_aspect_ratio[0]) for j in range(target_aspect_ratio[1])]
        #assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = cv2.resize(image,(self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            processed_images.append(thumbnail_img)
        return processed_images

    def __call__(self,image_file):
        image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
        images = self.dynamic_preprocess(image, use_thumbnail=True)
        pixel_values = np.stack(images).transpose(0,3,1,2)
        return pixel_values

def internvl_template(tokenizer): 
    
    PREFIX,ADAPT=[],[]
    tokenizer.im_start_id =  tokenizer.encode('<|im_start|>')[0]
    tokenizer.im_end_id = tokenizer.encode('<|im_end|>')[0]
    
    nl_token_id = tokenizer.encode("\n")
    system_id = tokenizer.encode("system")
    user_id = tokenizer.encode("user")
    assistant_id = tokenizer.encode("assistant")
    
    ROLE = {
        'system': [tokenizer.bos_token_id]+[tokenizer.im_start_id] + system_id + nl_token_id,
        'user': [tokenizer.im_start_id] + user_id + nl_token_id,
        'assistant': [tokenizer.im_end_id] + [tokenizer.im_start_id] + assistant_id + nl_token_id
    }
    PREFIX = [{"system":"你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。"}]
    ADAPT = [tokenizer.bos_token_id,tokenizer.im_start_id]
    
    try:
        config = AutoConfig.from_pretrained(tokenizer.name_or_path,trust_remote_code=True)
        num_image_token = int((config.vision_config.image_size // config.vision_config.patch_size) ** 2 * (config.downsample_ratio ** 2))
        tokenizer.get_image_tokens = lambda n:'<img>' + '<IMG_CONTEXT>' * n * num_image_token + '</img>'
        #tokenizer.img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        tokenizer.img_bos_token_id = tokenizer.convert_tokens_to_ids('<img>')
        tokenizer.img_eos_token_id = tokenizer.convert_tokens_to_ids('</img>')
    except:
        print('only llm!')
    
    return tokenizer,ROLE,PREFIX,ADAPT

TOKENIZER = {
    'LlamaTokenizer':llama_template,
    'LlamaTokenizerFast':llama_template,
    'PreTrainedTokenizerFast':llama3_template,
    'QWenTokenizer':qwen_template,
    'Qwen2TokenizerFast':qwen2_template,
    'BaichuanTokenizer':baichcuan_template,
    'InternLM2Tokenizer':internvl_template,
    'InternLM2TokenizerFast':internvl_template
}

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str, default="ft")
    parser.add_argument('-i', type=str, default="data.txt")
    parser.add_argument('-o', type=str, default="")
    parser.add_argument('-n', type=int, default=2**23)
    parser.add_argument('-c', type=str, default="gzip",choices=('gzip','brotli','snappy','lz4','zstd'))
    parser.add_argument('--batch_size', type=int, default=2**16)
    parser.add_argument('--seq_len', type=int, default=2**11)
    parser.add_argument('--cores', type=int, default=-1)
    parser.add_argument('--max_num', type=int, default=1)
    parser.add_argument('--tokenizer', type=str, default="openlm-research/open_llama_13b")
    parser.add_argument('--image_path', type=str, default="")
    parser.add_argument('--tmp', type=str, default="tmp")
    parser.add_argument('--stack', action='store_true', help="stack tokens")
    parser.add_argument('-T', action='store_true', help="thread")
    parser.add_argument('-C', action='store_true', help="clean")
    parser.add_argument('--sep', action='store_true', help="separate text and images.")
    args = parser.parse_args()
    main(args)