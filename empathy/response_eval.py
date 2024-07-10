from metrics import dialogue_evaluation

def load_preds_SEEK(load_path):
    pred = []
    ref = []
    with open(load_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i in range(0, len(lines)):
        if(lines[i].find("Greedy:") != -1):
            pred.append(lines[i][7:].strip())
        if(lines[i].find("Ref:") != -1):
            ref.append(lines[i][4:].strip())
    return pred, ref

preds, golds = load_preds_SEEK('./response.txt')
results = dialogue_evaluation(preds, golds)
print('** results: ', results)