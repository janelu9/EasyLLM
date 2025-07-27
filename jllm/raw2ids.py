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

os.environ['TOKENIZERS_PARALLELISM']='true'
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
    if isinstance(file,list):
        try:
            for line in file:
                doc = json.loads(line.strip())['text']
                for block in split_doc(doc):
                    yield block
        except:
            doc=''
            bos=True
            for line in file:
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
            if doc:
                for block in split_doc(doc,bos):
                    yield block
    else:
        with open(file,"r") as f:
            line = f.readline()
            try:
                while line:
                    doc = json.loads(line.strip())['text']
                    for block in split_doc(doc):
                        yield block
                    line = f.readline()
            except:
                doc=""
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
            
def pretrain_stack(iteritems,tokenizer,MAX_SEQ_LENGTH,**kwargs):
    ids = []
    cu_seqlens = [0]
    l = 0
    for bos,doc,eos in iteritems:
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
                yield {'input_ids':np.array(input_ids,dtype=np.int32),'cu_seqlens':np.array(cu_seqlens,dtype=np.int32)}
                cu_seqlens = [0]
            p += MAX_SEQ_LENGTH-OVERLAPPING_LENGTH
            
        if l==MAX_SEQ_LENGTH: # all ids yielded, clear ids
            ids = []   
        # elif l>=MAX_SEQ_LENGTH-3: # pad few then yield  
            # input_ids.extend([tokenizer.pad_token_id]*(MAX_SEQ_LENGTH-l))
            # cu_seqlens.append(l)
            # if l!=MAX_SEQ_LENGTH-1:cu_seqlens.append(MAX_SEQ_LENGTH-1)
            # yield {'input_ids':np.array(input_ids,dtype=np.int32),'cu_seqlens':np.array(cu_seqlens,dtype=np.int32)}
            # cu_seqlens = [0]
            # ids = []
        else: # join to next doc
            ids=input_ids

    if 2<l<MAX_SEQ_LENGTH:
        cu_seqlens.append(l)
        if l!=MAX_SEQ_LENGTH-1:cu_seqlens.append(MAX_SEQ_LENGTH-1)
        input_ids.extend([tokenizer.pad_token_id]*(MAX_SEQ_LENGTH-l))
        yield {'input_ids':np.array(input_ids,dtype=np.int32),'cu_seqlens':np.array(cu_seqlens,dtype=np.int32)}

def pretrain_dynamic(iteritems,tokenizer,MAX_SEQ_LENGTH,**kwargs):
    for bos,doc,eos in iteritems:
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
                yield {'input_ids':np.array(input_ids,dtype=np.int32)}
            p += MAX_SEQ_LENGTH-OVERLAPPING_LENGTH
            
        if 2<l:
            # input_ids.extend([tokenizer.pad_token_id]*(MAX_SEQ_LENGTH-l))
            yield {'input_ids':np.array(input_ids,dtype=np.int32)}
    
def finetune_generator(file):
    if isinstance(file,list):
        for line in file:
            yield json.loads(line)
    else:
        with open(file,"r") as f:
            line = f.readline()
            while line:
                yield json.loads(line)
                line = f.readline()

def rlhf(rows,tokenizer,MAX_SEQ_LENGTH,
         ROLE = {},PREFIX = [],ADAPT = [],**kwargs):
    # rows = json.loads(rows)
    labels  = np.array([rows.pop(0)],dtype=np.int32)
    msgs = (PREFIX + rows) if 'system' not in rows[0] else rows
    if 'assistant' not in rows[-1]:
        msgs.append({'assistant':''})
    ids = []
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
            ids.extend(ROLE[k])
            ids.extend(tokenizer.encode(v))         
        else:
            ids.extend(tokenizer.encode(v))
        pre_k = k
    yield {'input_ids':np.array(ids[-MAX_SEQ_LENGTH:],dtype=np.int32),'labels':labels}

def finetune(rows,tokenizer,MAX_SEQ_LENGTH,
             ROLE = {},PREFIX = [],ADAPT = [],
             qa_inputs_generator=None,padding=False,filter_null=False,**kwargs):
    # rows = json.loads(rows)
    msgs = (PREFIX + rows) if 'system' not in rows[0] else rows
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
        disable_thinking = True if k=='assistant' and v.startswith('<think>\n\n</think>') else False
        if k != pre_k:
            if k != "assistant":
                ids.append(tokenizer.im_end_id)
                if pre_k != "system":
                    divide.append(len(ids))
            ids.extend(ROLE[k])
            if k == "assistant":
                divide.append((len(ids)+tokenizer.no_thinking_len) if disable_thinking else len(ids))
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
                                               IGNORE_TOKEN_ID = -100,
                                               padding = padding):
            if filter_null:
                if np.sum(qa_inputs['labels']!=-100)<=1:
                    continue
            yield qa_inputs
            
def finetune_vl(iteritems,tokenizer,MAX_SEQ_LENGTH,
                ROLE = {},PREFIX = [],ADAPT = [],
                output_dir='',image_path='',padding = False,filter_null=False,**kwargs):

    from jllm.data.utils import qa_inputs_generator,img_token_alignment
    
    max_num_patches = 0
    max_num_images = 0
    images_database = {}
    def replace_image_token(v):
        if not isinstance(v,str):
            v,*imgs = v
            pixes = []
            num_patches = 0
            num_images = 0
            for img in imgs:
                try:
                    if img not in images_database:
                        image = cv2.imread(os.path.join(image_path,img))
                        image_tokens,resize_info,num_patch = tokenizer.get_image_tokens(image)
                        images_database[img] = (image_tokens,resize_info,num_patch)
                    else:
                        image_tokens,_,num_patch = images_database[img]
                    v=v.replace('<image>', image_tokens,1)
                    pixes.append(img)
                    num_patches+=num_patch
                    num_images+=1
                except:
                    v=v.replace('<image>', '<图片>',1)
                    pixes.append('')
            return v,pixes if len(pixes) else [''],num_patches,num_images
        return v,[''],0,0

    for rows in iteritems:
        # rows = json.loads(rows)
        msgs = (PREFIX + rows) if 'system' not in rows[0] else rows
        ids = []
        divide = [0]
        pixes = []
        num_patches = []
        num_images = []
        for start,msg in enumerate(msgs):
            k,v = next(iter(msg.items()))
            v,p,nps,nis = replace_image_token(v)
            if k != "assistant":
                ids.extend(ROLE[k] if k == 'system' or len(ROLE[k])==1 else ROLE[k][1:])
                ids.extend(tokenizer.encode(v))
                pre_k = k
                break
                
        if k != 'system':
            ids = ADAPT + ids
            pixes.append(p)
            num_patches.append(nps)
            num_images.append(nis)
            
        for msg in msgs[start+1:]:
            k,v = next(iter(msg.items()))
            v,p,nps,nis = replace_image_token(v)
            if k != pre_k:
                if k != "assistant":
                    pixes.append(p)
                    num_patches.append(nps)
                    num_images.append(nis)
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
                                                            IGNORE_TOKEN_ID = -100,
                                                            padding=padding):
                if filter_null:
                    if np.sum(qa_inputs['labels']!=-100)<=1:
                        continue
                        
                s = sub_divide[0]
                e = sub_divide[-2] if len(sub_divide)%2 else sub_divide[-1]
                matched = None
                if s == divide[0] and e == divide[-2]:
                    matched = True
                    si,ei=0,len(divide)//2
                else:
                    si,ei = np.searchsorted(divide,[s,e],side='right')
                    si = (si-1)//2
                    ei = (ei-1)//2+1
                    
                qa_inputs.update({'image_ids':pixes,'matched':matched,'siei':(si,ei)})
                max_num_patches = max(max_num_patches,sum(num_patches[si:ei]))
                max_num_images = max(max_num_images,sum(num_images[si:ei]))
                yield qa_inputs
                
    while output_dir.endswith('/'):
        output_dir=output_dir[:-1]
    partition_file = f"{output_dir}-image.info"
    data = {'pic':[],'rat':[]}
    for k,v in images_database.items():
        data['pic'].append(k)
        data['rat'].append(v[1])
    data['pic'].append(os.path.abspath(image_path))
    data['rat'].append(np.array([tokenizer.img_bos_token_id,tokenizer.img_eos_token_id,tokenizer.image_size,max_num_patches,max_num_images]))
    pyarrow.parquet.write_table(pyarrow.table(data),partition_file)
    print(f"\nAvailable pictures: {len(data['pic'])-1}")  
   
def tokenize(file,tokenizer,MAX_SEQ_LENGTH,
             ROLE = {},PREFIX = [],ADAPT = [],
             padding=False,filter_null=False,
             image_path=None,output_dir=None,
             **kwargs):
    iteritems = finetune_generator(file)
    if not hasattr(tokenizer,'get_image_tokens'):
        row0= next(iteritems)
        if isinstance(row0[0],int):
            func = partial(rlhf,tokenizer=tokenizer,MAX_SEQ_LENGTH=MAX_SEQ_LENGTH, ROLE=ROLE, PREFIX=PREFIX, ADAPT=ADAPT)
        else:
            from jllm.data.utils import qa_inputs_generator
            func = partial(finetune,
                           tokenizer=tokenizer,
                           MAX_SEQ_LENGTH=MAX_SEQ_LENGTH,
                           ROLE=ROLE, PREFIX=PREFIX, ADAPT=ADAPT,
                           qa_inputs_generator=qa_inputs_generator,
                           padding=padding,filter_null=filter_null)
            
        for item in func(row0):
            yield item
        for row in iteritems:
            for item in func(row):
                yield item
    else:
        func = partial(finetune_vl,
                       tokenizer=tokenizer,
                       MAX_SEQ_LENGTH=MAX_SEQ_LENGTH,
                       ROLE=ROLE, PREFIX=PREFIX, ADAPT=ADAPT, 
                       image_path=image_path,output_dir = output_dir,
                       padding=padding,filter_null=filter_null)
        for item in func(iteritems):
            yield item
   
def write_parquet(filename,
                  output_dir,
                  tokenizer,
                  MAX_SEQ_LENGTH=2048,
                  dtype='ft',
                  batch_size=2**15,
                  compression='gzip',
                  stack=False,
                  image_path='',
                  max_pixels = 1024**2,
                  padding=False,
                  filter_null=False):
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer,use_fast=True,trust_remote_code=True,add_bos_token = False)
    tokenizer.encode = partial(tokenizer.encode,add_special_tokens=False)
    tokenizer_class = tokenizer.__class__.__name__ 
    tokenizer,ROLE,PREFIX,ADAPT = TOKENIZER[tokenizer_class](tokenizer,max_pixels=max_pixels)
    auto_batch_size = False
    if isinstance(filename,tuple):
        filename,chunk_id,contents = filename
        chunk_id = f'-chunk-{chunk_id:02d}'
    else:
        contents = filename
        chunk_id = '-chunk-00'
        
    file = os.path.splitext(os.path.basename(filename))[0]
    partition_dir = output_dir #os.path.join(output_dir , file)
    partition_file = os.path.join(partition_dir , f"{file}-%05d{chunk_id}.{compression}.parquet")
    check_file = os.path.join(output_dir , f".{file}{chunk_id}.crc")
    file_id = f"{os.path.dirname(filename)}/{file}{chunk_id}"
    if os.path.exists(check_file):
        print(f"{file_id} converted, continue!")
        return

    if dtype == 'ft':
        item_iter =tokenize(contents,
                            tokenizer,
                            MAX_SEQ_LENGTH, 
                            ROLE=ROLE, 
                            PREFIX=PREFIX, 
                            ADAPT=ADAPT,
                            padding=padding,
                            filter_null=filter_null,
                            image_path=image_path,
                            output_dir = os.path.join(output_dir , file),
                            )
    else:
        generator = pretrain_generator(contents)
        if stack:
            item_iter = pretrain_stack(generator,tokenizer,MAX_SEQ_LENGTH)
        else:
            item_iter = pretrain_dynamic(generator,tokenizer,MAX_SEQ_LENGTH)
        
    # os.makedirs(partition_dir,exist_ok=True)
    max_seq_len = 0
    max_num_blocks = 0
    if not auto_batch_size:
        pbar = tqdm.tqdm(float("inf"))
        data = next(item_iter)
        data_batch={k:[data[k]] for k in data}
        max_seq_len = max(max_seq_len,len(data['input_ids']))
        max_num_blocks = max(max_num_blocks,len(data.get('cu_seqlens',[])))
        i=0
        pbar.update(1)
        try:
            while True:
                for _ in range(batch_size-1):
                    data = next(item_iter)
                    for k in data:data_batch[k].append(data[k])
                    max_seq_len = max(max_seq_len,len(data['input_ids']))
                    max_num_blocks = max(max_num_blocks,len(data.get('cu_seqlens',[])))
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
                max_seq_len = max(max_seq_len,len(data['input_ids']))
                max_num_blocks = max(max_num_blocks,len(data.get('cu_seqlens',[])))
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
        max_seq_len = max(max_seq_len,len(data['input_ids']))
        max_num_blocks = max(max_num_blocks,len(data.get('cu_seqlens',[])))
        element_num = sum(data[k].size for k in data)
        i=0
        p=0
        pbar.update(1)
        for data in item_iter:
            i+=1
            pbar.update(1)
            for k in data: 
                data_batch[k].append(data[k])
                max_seq_len = max(max_seq_len,len(data['input_ids']))
                max_num_blocks = max(max_num_blocks,len(data.get('cu_seqlens',[])))
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
        if data_batch[k]:
            pyarrow.parquet.write_table(pyarrow.table(data_batch),
                                        partition_file % (p), 
                                        compression=compression)
    del data_batch                                
    os.system(f"echo '{i+1} {max_seq_len} {batch_size} {max_num_blocks} {len(data.keys())}' > {check_file}")
    print(f"{file_id} stored in parquet with {i+1} samples")
    gc.collect()
    return {file_id:{'num_samples':i+1,'max_len':max_seq_len,'num_samples_per_file':batch_size,'max_num_blocks':max_num_blocks,'fields':list(data.keys())}}

def main(args):
    print(args)
    source=os.path.abspath(args.input)
    while source.endswith('/'):
        source=source[:-1]
    source_dir=os.path.dirname(source)
    file =os.path.basename(source)
    file_name = os.path.splitext(file)[0]
    
    cpus=int(os.cpu_count()*0.8) if args.cores <0 else args.cores
    
    if os.path.isfile(source):
        if args.num_lines_per_partition is not None:
            tmp = os.path.join(source_dir,args.tmp)
            if not os.path.exists(tmp) or args.clean:
                os.system(f" rm -rf {tmp}") 
                os.makedirs(tmp)
                os.system(f"cd {tmp};split -d -a 5 -{args.num_lines_per_partition} ../{file} {file_name}-;cd -;")
            files =[os.path.join(tmp, i) for i in os.listdir(tmp)]
            cpus=min(len(files),cpus)
        else:
            tmp = None
            with open(source,'r') as f:
                files = f.readlines()
            num_samples = len(files)
            cpus = min((num_samples+args.chunk_size-1)//args.chunk_size,cpus)
            chunk_size = (num_samples+cpus-1)//cpus
            files = [(source,i,files[i*chunk_size:(i+1)*chunk_size]) for i in range(cpus)]
    else:
        tmp = source
        files =[os.path.join(tmp, i) for i in os.listdir(tmp)]
        cpus=min(len(files),cpus)
        
    output_dir = args.output if args.output !="" else os.path.join(source_dir,file_name+f"_{os.path.basename(args.tokenizer)}")
    if os.path.exists(output_dir):
        if args.clean:
            os.system(f" rm -rf {output_dir}/*")
            os.system(f" rm -rf {output_dir}/.*.crc")
    else:
        os.makedirs(output_dir)

    Pool = ThreadPoolExecutor if args.thread else ProcessPoolExecutor
    print(f"########## begine converting {args.type} data with {cpus} executors.###########" )
    with Pool(max_workers=cpus) as exe:
        func = partial(write_parquet,
                       output_dir=output_dir,
                       tokenizer=args.tokenizer,
                       MAX_SEQ_LENGTH= args.max_len,
                       dtype=args.type,
                       batch_size=args.batch_size,
                       compression=args.compress.lower(),
                       stack=args.stack,
                       image_path=args.image_path,
                       max_pixels=args.max_pixels,
                       padding=args.pad,
                       filter_null=args.filter)
        files.sort()
        np.random.shuffle(files)
        data_infos = list(exe.map(func,files))
    num_samples = 0
    max_len = 0
    max_num_blocks = 0
    fields = set()
    data_json = {}
    for data_info in data_infos:
        k,v =  next(iter(data_info.items()))
        num_samples = num_samples+v['num_samples']
        max_len = max(v['max_len'],max_len)
        max_num_blocks = max(v['max_num_blocks'],max_num_blocks)
        fields.update(v['fields'])
        data_json.update(data_info)
    data_json.update({'num_samples':num_samples,
                      'max_len':max_len,
                      'num_samples_per_file':args.batch_size,
                      'max_num_blocks':max_num_blocks,
                      'fields':list(fields)})
    with open(os.path.join(output_dir,f'{file_name}_info.json'),'w')as f:json.dump(data_json,f,indent=2)
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

    
def llama_template(tokenizer,**kwargs):
    
    PREFIX,ADAPT=[],[]
    if len(tokenizer.encode('<think>'))==1:
        tokenizer.im_end_id = tokenizer.eos_token_id
        ROLE = {
            'user': tokenizer.encode('<｜User｜>'),
            'assistant': tokenizer.encode('<｜Assistant｜>')
        }
        ADAPT = [tokenizer.bos_token_id]
    else:
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
    
def llama3_template(tokenizer,**kwargs):
    
    PREFIX,ADAPT=[],[]
    tokenizer.bos_token_id = tokenizer.encode('<|begin_of_text|>')[0]
    tokenizer.im_end_id = tokenizer.encode('<|eot_id|>')[0]
    tokenizer.pad_token_id = tokenizer.encode('<|finetune_right_pad_id|>')[0]
    tokenizer.eos_token_id = tokenizer.encode('<|end_of_text|>')[0]

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
    
    if 'Cutting Knowledge Date: December 2023' in tokenizer.chat_template and '26 Jul 2024' in tokenizer.chat_template:
        PREFIX = [{'system':'Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n'}]
        
    return tokenizer,ROLE,PREFIX,ADAPT
    
def qwen_template(tokenizer,**kwargs): 
    
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
    
def qwen2_template(tokenizer,**kwargs): 
    
    PREFIX,ADAPT=[],[]
    tokenizer.bos_token_id = tokenizer.im_start_id = tokenizer.encode("<|im_start|>")[0]
    tokenizer.im_end_id = tokenizer.encode("<|im_end|>")[0]
    tokenizer.pad_token_id = tokenizer.eos_token_id = tokenizer.encode("<|endoftext|>")[0]
    nl_token_id = tokenizer.encode("\n")
    system_id = tokenizer.encode("system")
    user_id = tokenizer.encode("user")
    assistant_id = tokenizer.encode("assistant")
    
    tokenizer.no_thinking_len = len(tokenizer.encode('<think>\n\n</think>\n\n'))
    
    ROLE = {
        'system': [tokenizer.im_start_id] + system_id + nl_token_id,
        'user': nl_token_id + [tokenizer.im_start_id] + user_id + nl_token_id,
        'assistant': [tokenizer.im_end_id] + nl_token_id + [tokenizer.im_start_id] + assistant_id + nl_token_id
    }

    if 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' in tokenizer.chat_template:
        PREFIX = [{'system':'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'}]
    elif 'You are a helpful assistant.' in tokenizer.chat_template:
        PREFIX = [{'system':'You are a helpful assistant.'}]
    
    config = AutoConfig.from_pretrained(tokenizer.name_or_path)
    if hasattr(config,'vision_config'):
        from transformers import AutoProcessor
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
        
        processor = AutoProcessor.from_pretrained(tokenizer.name_or_path)
        
        def get_image_tokens(images,pad_token='<|image_pad|>'):
            if not isinstance(images,(list,tuple)):
                images=[images]
            height,width,_=images[0].shape
            resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
                    min_pixels=processor.image_processor.min_pixels,
                    max_pixels=kwargs['max_pixels'],
                )
            patch_num = len(images)
            grid_h = resized_height // processor.image_processor.patch_size
            grid_w = resized_width // processor.image_processor.patch_size
            merge_length = processor.image_processor.merge_size**2
            if patch_num == 1:
                grid_t = 1
            else:
                grid_t = patch_num//processor.image_processor.temporal_patch_size
            num_patches = grid_t*grid_h*grid_w
            img_tokens =  num_patches//merge_length
            return '<|vision_start|>' + pad_token*img_tokens + '<|vision_end|>',(grid_t,grid_h,grid_w),num_patches
            
        tokenizer.get_image_tokens = get_image_tokens
        tokenizer.get_video_tokens = partial(get_image_tokens,pad_token='<|video_pad|>')
        tokenizer.img_bos_token_id = tokenizer.convert_tokens_to_ids('<|vision_start|>')
        tokenizer.img_eos_token_id = tokenizer.convert_tokens_to_ids('<|vision_end|>')
        tokenizer.image_size = kwargs['max_pixels']

    return tokenizer,ROLE,PREFIX,ADAPT 

def baichcuan_template(tokenizer,**kwargs): 
    
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

def internvl_template(tokenizer,**kwargs): 
    
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
        img_reader=ImageReaderCV2(max_num=args.max_num)
        num_image_token = int((config.vision_config.image_size // config.vision_config.patch_size) ** 2 * (config.downsample_ratio ** 2))
        
        def get_image_tokens(image):
            if isinstance(image,int):return image
            target_aspect_ratio,_ = img_reader.find_closest_aspect_ratio(image)
            num_patches = target_aspect_ratio[0]*target_aspect_ratio[1]
            num_patches = num_patches + int(num_patches>1)
            img_tokens = num_patches * num_image_token
            tokens = '<img>' + '<IMG_CONTEXT>' *img_tokens  + '</img>'
            return tokens,target_aspect_ratio,num_patches
        
        tokenizer.get_image_tokens = get_image_tokens
        #tokenizer.img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        tokenizer.img_bos_token_id = tokenizer.convert_tokens_to_ids('<img>')
        tokenizer.img_eos_token_id = tokenizer.convert_tokens_to_ids('</img>')
        tokenizer.image_size = config.vision_config.image_size**2
        
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
    parser.add_argument('-t','--type', type=str, default="ft")
    parser.add_argument('-i','--input', type=str, default="data.txt")
    parser.add_argument('-o','--output', type=str, default="")
    parser.add_argument('-n','--num_lines_per_partition', type=int, default=None)
    parser.add_argument('--chunk_size', type=int, default=16384)
    parser.add_argument('-c','--compress', type=str, default="gzip",choices=('gzip','brotli','snappy','lz4','zstd'))
    parser.add_argument('--batch_size', type=int, default=2**16)
    parser.add_argument('--max_len', type=int, default=2**11)
    parser.add_argument('--cores', type=int, default=-1)
    parser.add_argument('--max_num', type=int, default=1)
    parser.add_argument('--max_pixels', type=int, default=2560*1440)
    parser.add_argument('--tokenizer', type=str, default="openlm-research/open_llama_13b")
    parser.add_argument('--image_path', type=str, default="")
    parser.add_argument('--tmp', type=str, default="tmp")
    parser.add_argument('--stack', action='store_true', help="stack tokens")
    parser.add_argument('-T','--thread', action='store_true', help="thread")
    parser.add_argument('-C','--clean', action='store_true', help="clean")
    parser.add_argument('--pad', action='store_true', help="pad the token.")
    parser.add_argument('--filter', action='store_true', help="filter the samples of all null tokens.")
    args = parser.parse_args()
    main(args)