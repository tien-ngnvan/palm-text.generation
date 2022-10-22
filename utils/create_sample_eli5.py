from utils.lfqa_utils import query_qa_dense_index,load_model_qs,get_embds_qs
from datasets import load_dataset

import os
from tqdm import tqdm
def reprocess_text(text):
    text=text.replace("\n"," ").replace("\t"," ").replace("  "," ").replace(" --t--", "").lower().strip()
    return text

if (__name__=='__main__'):
    class args: # args needed to connect database
        dbname="wiki_pgvector"
        host="db"
        port=5432
        user="postgresql"
        pwd="postgresql"
        topk_doc_wiki=3
    li_task=['train','val','test']
    
    qs_model, qs_tokenizer=load_model_qs() # initial model and tokenizer
    for task in li_task:
        path_base="../{}_eli5".format(task) # path to directory which stores all samples
        if (not os.path.exists(path_base)): os.makedirs(path_base)    
        dataset = load_dataset("eli5") # load dataset eli5

        for num,i in tqdm(enumerate(dataset['{}_eli5'.format(task)])):

            if (num%100==0):
                print("created {} samples".format(num))
            query=i['title'] # get query
            query=reprocess_text(query)

            documents=query_qa_dense_index(args,query, qs_model=qs_model, qs_tokenizer=qs_tokenizer,device="cuda:0") #get docs from database

            pos=i['answers']['score'].index(max(i['answers']['score'])) # position which has the highest score
            answer=i['answers']['text'][pos].replace("\n"," ").replace("  "," ") # get answers for this position
            answer=reprocess_text(answer)

            conditioned_doc = "\<P> " + " \<P> ".join([reprocess_text(d.content) for d in documents]) #join all docs with tag \<P>

            query_and_docs = "question: {} context: {}".format(query, conditioned_doc) #join query and docs
            text="{}\n\n@answer\n{}".format(query_and_docs,answer) # join answers too

            path=os.path.join(path_base,"{}_eli5_sample_".format(task))
            with open("{}{:05d}".format(path,num),"w") as w: # write answer
                w.write(text)