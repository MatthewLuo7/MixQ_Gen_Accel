import json
import os


new_dict = {}
idx = 0
with open(os.path.normpath('./opt_dataset.json'), 'r') as file:
    opt_dicts = json.load(file)
for (opt_name, opt_val) in opt_dicts.items():
    if opt_val['Config']['Opt_Type'] == 'FP_Opt':
    	new_name = f'FP_Opt_{str(idx)}'
    	new_dict[new_name] = {'Config': opt_val['Config'], 'Syn_Res': opt_val['Syn_Res']}
    	idx += 1

with open(os.path.normpath('./opt_dataset_FP1000.json'), 'r') as file:
    opt_dicts = json.load(file)
for (opt_name, opt_val) in opt_dicts.items():
    if opt_val['Config']['Opt_Type'] == 'FP_Opt':
    	new_name = f'FP_Opt_{str(idx)}'
    	new_dict[new_name] = {'Config': opt_val['Config'], 'Syn_Res': opt_val['Syn_Res']}
    	idx += 1


print(idx)
with open('./dataset.json', 'w', encoding='utf-8') as f:
	json.dump(new_dict, f, indent=4)
