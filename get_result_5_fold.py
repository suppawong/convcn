# import os
import numpy as np
import json

k = 20
def get_number_list(str):
    str_list = str.split('\t')
    str_list.remove('\n')
    num_list = [float(s) for s in str_list]
    return num_list

dataset = ['AlgoCitation_5_fold','TopicCitation_5_fold','CiteSeer_umd_5_fold', 'CiteULike_5_fold' ]

'''
method list
    ConvCN_PaperRank_harmonic
    ConvCN_PaperRank_weighted_avg
    ConvCN_PaperRank_dot_product
    ConvCN_CF_weighted_avg
    ConvCN_CF_harmonic
    ConvCN_CF_dot_product
'''
# method_name = 'ConvCN_PaperRank_weighted_avg'

# print(method_names)
# exit(0)
model_name = 'ver1_'
# epochs = ['500','1000','2000','3000']
fold = ['1','2','3','4','5']

eval_result = []

def add_to_result_dict(method_name, root_dir,d):
    pr_allpaper = {}
    rec_allpaper = {}
    f1_allpaper = {}
    mrr_allpaper = 0
    bpref_allpaper = 0

    for i in range(1,k+1):
        pr_allpaper[i] = 0
        rec_allpaper[i] = 0
        f1_allpaper[i] = 0

    for epoch in epochs:
        # print('Dataset:',d, ' at ',epoch,'epochs')
        # print('Dataset:',d, ' Epochs:', epoch,' Alpha:',alpha)
        for f in fold:
            dir = root_dir + f + '-' + epoch + 'epochs.txt'
            # dir = root_dir + f + '.txt'
            print(dir)
            f = open(dir, 'r')

            l = f.readline()  # read precision
            prec = get_number_list(l)
            for i in range(1, k + 1):
                pr_allpaper[i] = pr_allpaper[i] + prec[i - 1]

            l = f.readline()  # read recall
            rec = get_number_list(l)
            for i in range(1, k + 1):
                rec_allpaper[i] = rec_allpaper[i] + rec[i - 1]

            l = f.readline()  # read f1
            f1 = get_number_list(l)
            for i in range(1, k + 1):
                f1_allpaper[i] = f1_allpaper[i] + f1[i - 1]

            l = f.readline()  # read MRR,Bpref
            s = l.split('\t')
            mrr_allpaper = mrr_allpaper + float(s[0])
            bpref_allpaper = bpref_allpaper + float(s[1])

        for i in range(1, k + 1):
            pr_allpaper[i] = pr_allpaper[i] / 5.0
            rec_allpaper[i] = rec_allpaper[i] / 5.0
            f1_allpaper[i] = f1_allpaper[i] / 5.0

        mrr_allpaper = mrr_allpaper / 5.0
        bpref_allpaper = bpref_allpaper / 5.0

        result_dict = {}
        result_dict['method_name'] = method_name
        result_dict['dataset'] = d
        result_dict['precision'] = list(pr_allpaper.values())
        result_dict['recall'] = list(rec_allpaper.values())
        result_dict['f1'] = list(f1_allpaper.values())
        result_dict['mrr'] = mrr_allpaper
        result_dict['bpref'] = bpref_allpaper
        eval_result.append(result_dict)

def get_eval_result(alphas,method_names):
    for method_name in method_names:
        for d in dataset:
            pr_str = ''
            rec_str = ''
            f1_str = ''
            bpref_str = ''
            mrr_str = ''
            # print(d)
            if alphas == []:
                root_dir = '.\\result\\' + method_name + '\\' + model_name + '\\ensemble result_' + d + '_' + model_name
                add_to_result_dict(method_name, root_dir,d)
            else:
                for alpha in alphas:
                    root_dir = '.\\result\\' + method_name + '\\' + model_name + '\\alpha=' + str(alpha) + '\\ensemble result_' + d + '_' + model_name
                    add_to_result_dict(method_name+'_alpha='+str(alpha), root_dir,d)

            # for alpha in alphas:
            # # for d in dataset:
            #     root_dir = '.\\result\\'+method_name+'\\'+model_name +'\\alpha='+str(alpha)+ '\\ensemble result_'+d + '_' + model_name
            #     for epoch in epochs:
            #         # print('Dataset:',d, ' at ',epoch,'epochs')
            #         # print('Dataset:',d, ' Epochs:', epoch,' Alpha:',alpha)
            #         for f in fold:
            #             dir = root_dir+f+'-'+epoch+'epochs.txt'
            #             # dir = root_dir + f + '.txt'
            #             # print(dir)
            #             f = open(dir,'r')
            #
            #             l = f.readline() # read precision
            #             prec = get_number_list(l)
            #             for i in range(1,k+1):
            #                 pr_allpaper[i] = pr_allpaper[i]+prec[i-1]
            #
            #             l = f.readline() # read recall
            #             rec = get_number_list(l)
            #             for i in range(1,k+1):
            #                 rec_allpaper[i] = rec_allpaper[i]+rec[i-1]
            #
            #             l = f.readline() # read f1
            #             f1 = get_number_list(l)
            #             for i in range(1,k+1):
            #                 f1_allpaper[i] = f1_allpaper[i]+f1[i-1]
            #
            #             l = f.readline() # read MRR,Bpref
            #             s = l.split('\t')
            #             mrr_allpaper = mrr_allpaper + float(s[0])
            #             bpref_allpaper = bpref_allpaper + float(s[1])
            #
            #         for i in range(1,k+1):
            #             pr_allpaper[i] = pr_allpaper[i]/5.0
            #             rec_allpaper[i] = rec_allpaper[i]/5.0
            #             f1_allpaper[i] = f1_allpaper[i]/5.0
            #
            #         mrr_allpaper = mrr_allpaper/5.0
            #         bpref_allpaper = bpref_allpaper/5.0
            #
            #         # print(pr_allpaper)
            #         # print(rec_allpaper)
            #         # print(f1_allpaper)
            #         # print(mrr_allpapaer, bpref_allpaper)
            #
            #         # print('k', [k for k in range(1,11)])
            #         # print('Prec:',[pr for pr in pr_allpaper.values()])
            #         # print('Rec:',[rec for rec in rec_allpaper.values()])
            #         # print('F1:',[f1 for f1 in f1_allpaper.values()])
            #         # print('MRR',mrr_allpapaer)
            #         # print('Bpref',bpref_allpaper,'\n')
            #         # print(*[pr for pr in pr_allpaper.values()], sep=' ')
            #         # print(*[rec for rec in rec_allpaper.values()],sep=' ')
            #         # print(*[f1 for f1 in f1_allpaper.values()], sep=' ')
            #         # print(mrr_allpapaer)
            #         # print(bpref_allpaper, '\n')
            #
            #         # print(list(pr_allpaper.values()))
            #         # pr_str = pr_str+'alpha='+str(alpha)+'\t'+str(list(pr_allpaper.values()))+'\n'
            #         # pr_str = pr_str.replace(']','').replace('[','').replace(',','\t')
            #         # # print(pr_str)
            #         #
            #         # rec_str = rec_str+'alpha='+str(alpha)+'\t'+str(list(rec_allpaper.values()))+'\n'
            #         # rec_str = rec_str.replace(']', '').replace('[', '').replace(',', '\t')
            #         #
            #         # f1_str = f1_str+'alpha='+str(alpha)+'\t'+str(list(f1_allpaper.values()))+'\n'
            #         # f1_str = f1_str.replace(']', '').replace('[', '').replace(',', '\t')
            #         #
            #         # bpref_str = bpref_str + 'alpha='+str(alpha)+'\t'+str(bpref_allpaper) +'\n'
            #         # mrr_str = mrr_str + 'alpha='+str(alpha)+'\t'+str(mrr_allpaper) +'\n'
            #
            #         result_dict = {}
            #         result_dict['method_name'] = method_name
            #         result_dict['dataset'] = d
            #         result_dict['precision'] = list(pr_allpaper.values())
            #         result_dict['recall'] = list(rec_allpaper.values())
            #         result_dict['f1'] = list(f1_allpaper.values())
            #         result_dict['mrr'] = mrr_allpaper
            #         result_dict['bpref'] = bpref_allpaper
            #         eval_result.append(result_dict)
                    # print(method_name, d, 'Precision', list(pr_allpaper.values()),sep='\t')
                    # print(method_name, d, 'Recall', list(rec_allpaper.values()),sep='\t')
                    # print(method_name, d, 'F1', list(f1_allpaper.values()),sep='\t')
                    # print(method_name, d, 'MRR', mrr_allpaper,sep='\t')
                    # print(method_name, d, 'Bpref', bpref_allpaper,sep='\t')

            # print(d)
            # print(method_name,d,'Precision',pr_str)
            # # print(pr_str)
            # print(method_name, d, 'Recall', rec_str)
            # # print('recall')
            # # print(rec_str)
            # print(method_name, d, 'F1', f1_str)
            # print(method_name, d, 'MRR', mrr_str)
            # print(method_name, d, 'Bpref', bpref_str)
            # print('f1')
            # print(f1_str)
            # print(bpref_str)
            # print(mrr_str)

alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 ]
method_names1 = '''ConvCN_PaperRank_harmonic
ConvCN_PaperRank_dot_product
ConvCN_CF_harmonic
ConvCN_CF_dot_product'''.split('\n')

method_names2 = '''ConvCN_PaperRank_weighted_avg
ConvCN_CF_weighted_avg'''.split('\n')

get_eval_result([],method_names1)
get_eval_result(alphas,method_names2)

print(eval_result)
with open('eval_result_k=20.json', 'w') as outfile:
    json.dump(eval_result, outfile)