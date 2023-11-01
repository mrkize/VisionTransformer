import os
atk_list = ["roll_nn", "last_attn_nn", "out", "base_d"]
dataset_list = ["cifar10", "cifar100", "ImageNet100"]
metric = ["cos-sim", "pearson", "Euclid"]
for atk in atk_list:
    for dataset in dataset_list:
        cmd = "python run_atk.py --atk_method {} --dataset {}".format(atk, dataset)
        os.system(cmd)
for atk in atk_list:
    for dataset in dataset_list:
        cmd = "python run_atk.py --atk_method {} --dataset {} --adaptive".format(atk, dataset)
        os.system(cmd)
# atk = "metric"
# for dataset in dataset_list:
#     cmd = "python run_atk.py --atk_method {} --dataset {}".format(atk, dataset)
#     os.system(cmd)
# for atk in atk_list:
#     for dataset in dataset_list:
#         cmd = "python mia_attn.py --atk_method {} --dataset {} --metric {}".format(atk, dataset, mt)
#         os.system(cmd)