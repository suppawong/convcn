import json

def get_number_list(str):
    str_list = str.split('\t')
    # str_list.remove('\n')
    num_list = [float(s) for s in str_list]
    return num_list

eval_result = []

def get_result(filename,method_name):
    # file = open('result/PaperRank_result.txt','r')
    # file = open('result/CF_result.txt','r')
    file = open(filename,'r')
    # result_dict['method_name'] = method_name
    # result_dict['dataset'] = d
    # result_dict['precision'] = list(pr_allpaper.values())
    # result_dict['recall'] = list(rec_allpaper.values())
    # result_dict['f1'] = list(f1_allpaper.values())
    # result_dict['mrr'] = mrr_allpaper
    # result_dict['bpref'] = bpref_allpaper
    # eval_result.append(result_dict)

    for i in range(0,4): # for 4 datasets
        result_dict = {}
        # result_dict['method_name'] = 'PaperRank'
        result_dict['method_name'] = method_name
        result_dict['dataset'] = file.readline().strip().replace('\n','')
        # prec = file.readline()
        # print(prec.replace('Precision\t',''))
        result_dict['precision'] = get_number_list(file.readline().replace('Precision\t',''))
        result_dict['recall'] = get_number_list(file.readline().replace('Recall\t',''))
        result_dict['f1'] = get_number_list(file.readline().replace('F1\t',''))
        result_dict['mrr'] = float(file.readline().replace('MRR\t','').replace('\n','').strip())
        result_dict['bpref'] = float(file.readline().replace('Bpref\t','').replace('\n','').strip())
        eval_result.append(result_dict)
        # print(result_dict)

'''
    file format:
        dataset name
        precision
        recall
        f1
        mrr
        bpref
'''

get_result('result/CF_result.txt','CF')
get_result('result/PaperRank_result.txt','PaperRank')
print(eval_result)
with open('baselines_result.json', 'w') as outfile:
    json.dump(eval_result, outfile)

# with open('PaperRank_result.json', 'w') as outfile:
#     json.dump(eval_result, outfile)

# with open('CF_result.json', 'w') as outfile:
#     json.dump(eval_result, outfile)