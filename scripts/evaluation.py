import sys
sys.path.append('/home/weiguang/Mixture-of-Domain-Adapters/')
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from tqdm import tqdm
import json
import re

question = []
answer = []
index = []
with open("result_bos.json",'r') as f:
    for line in f:
        line = json.loads(line.strip())
        value = re.split(r'(?<=回答: )',list(line.values())[0])
        index.append(int(list(line.keys())[0]))
        question.append(value[0][4:-6]) ## strip "问题: " and "\n\n回答: "
        answer.append(value[1])

pred_data = pd.DataFrame({"index":index,"question":question,"description":[""]*len(answer),"answer":answer})
truth_data = pd.read_csv("/home/weiguang/SaBART/matinf_1.0_encrypted/test.csv")

bleu = []
rouge1 = []
rouge2 = []
rougeL = []
for i in tqdm(range(len(pred_data))):
    truth = truth_data['answer'][i]
    pred = pred_data['answer'][i]
    if type(pred)!= str:
        breakpoint()
    #candidate is inference data, reference is original data
    bleu_score = sentence_bleu([truth.split()] ,pred.split())
    bleu.append(bleu_score)
    
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    rouge_score = scorer.score(truth, pred)
    rouge1.append(rouge_score['rouge1'])
    rouge2.append(rouge_score['rouge2'])
    rougeL.append(rouge_score['rougeL'])

score = pd.DataFrame({"index": list(range(len(pred_data))),"bleu":bleu, "rouge1":rouge1, "rouge2":rouge2, "rougeL":rougeL})
score.to_csv("score.csv")

print('bleu:', sum(bleu)/len(bleu))

rouge1 = [x[2] for x in rouge1]
rouge2 = [x[2] for x in rouge2]
rougeL = [x[2] for x in rougeL]
print('rouge1:', sum(rouge1)/len(rouge1))
print('rouge2:', sum(rouge2)/len(rouge2))
print('rougeL:', sum(rougeL)/len(rougeL))