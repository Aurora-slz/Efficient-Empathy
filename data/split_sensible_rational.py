import json
import random

def load_file(load_path):
    with open(load_path, 'r', encoding='utf-8') as f1:
        data = json.load(f1)
        print(data[0])
    return data

def save_file(data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f1:
        f1.write(json.dumps(data))

def convert_qwen(idx, data):
    res = {}
    res_conv = []
    conv = data['context'].split("</s>")
    for i in range(0, len(conv)):
        if(i % 2 == 0):
            # res_conv.append({"role":"user", "content":conv[i]})
            res_conv.append({"from":"user", "value":conv[i]})
        else:
            # res_conv.append({"role":"assistant", "content":conv[i]})
            res_conv.append({"from":"assistant", "value":conv[i]})
    # res_conv.append({"role":"assistant", "content":data["response"]})
    res_conv.append({"from":"assistant", "value":data["response"]})
    res["conversations"] = res_conv
    res["id"] = str(idx)
    # print(res)
    return res

def get_origin_data(data):
    origin_data = []
    for i in range(0, len(data)):
        tmp_dic = convert_qwen(i, data[i])
        origin_data.append(tmp_dic)
    return origin_data

def get_conversation(context):
    new_con = ""
    for i in range(0, len(context)):
        if(i % 2 == 0):
            new_con += "Speaker: {}\n".format(context[i]["content"])
        else:
            new_con += "Listener: {}\n".format(context[i]["content"])
    return new_con

def get_conversation_expand(context):
    new_con = ""
    for i in range(0, len(context)):
        new_con += "Speaker: {}\n".format(context[i]["Context"])
        new_con += "Listener: {}\n".format(context[i]["Response"])
    return new_con

def get_llama3_data(load_path):
    
    data = load_file(load_path)
    llama3_data = []
    for i in range(0, len(data)):
        context = get_conversation(data[i]["conversations"][:-1])
        context += "Listener: "
        response = data[i]["conversations"][-1]["value"]

        tmp = {}
        tmp["instruction"] = context
        tmp["input"] = ''
        tmp["output"] = response
        llama3_data.append(tmp)
    return llama3_data



def get_high_score(data):
    YUZHI = 5
    print('** len-overall: ', len(data))
    ration_data = []
    sen_data = []
    neuro_data = []
    for i in range(0, len(data)):
        eva = data[i]["evaluation"]
        tmp = eva[eva.find("Rationality: ") + 13: eva.find("Rationality: ") + 15]
        if(tmp[-1] == "\""):
            tmp = tmp[0]
        v_ration = int(tmp)

        tmp = eva[eva.find("Sensibility: ") + 13: eva.find("Sensibility: ") + 15]
        if(tmp[-1] == "\""):
            tmp = tmp[0]
        v_sen = int(tmp)

        tmp_dic = convert_qwen(i, data[i])

        if(v_ration>YUZHI and v_sen<YUZHI):
            ration_data.append(tmp_dic)
        elif(v_ration<YUZHI and v_sen>YUZHI):
           sen_data.append(tmp_dic)
        else:
            neuro_data.append(tmp_dic)
    
    print('** len-ration: ', len(ration_data))
    print('** len-sen: ', len(sen_data))
    print('** len-neuro: ', len(neuro_data))

    return ration_data, sen_data, neuro_data





if __name__ == '__main__':
    
    load_path = "/data/train_results.json"
    data = load_file(load_path)


    ration_data, sen_data, neuro_data = get_high_score(data)
    save_file(ration_data, "/data/qwen_train_rational.json")
    save_file(sen_data, "/data/qwen_train_sensibile.json")
    save_file(neuro_data, "/data/qwen_train_neuro.json")
