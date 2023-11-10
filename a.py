import pandas as pd
def change_csv(atk, defence, i):
    source_file = 'confound_cifar100/{}/pre/{}.csv'.format(atk, defence)
    target_file = 'tmp.csv'

    source_data = pd.read_csv(source_file)
    target_data = pd.read_csv(target_file)

    target_data.at[i, 'Precision'] = source_data['Basic_shadow.pth'].iloc[1]
    target_data.at[i+2, 'Precision'] = source_data['Basic_shadow.pth'].iloc[2]
    target_data.at[i+4, 'Precision'] = source_data['Basic_shadow.pth'].iloc[3]
    target_data.at[i+6, 'Precision'] = source_data['Basic_shadow.pth'].iloc[4]

    target_data.to_csv(target_file, index=False)

def change_csv2(atk, defence, i):
    source_file = 'confound_cifar100/{}/rec/{}.csv'.format(atk, defence)
    target_file = 'tmp.csv'

    source_data = pd.read_csv(source_file)
    target_data = pd.read_csv(target_file)

    target_data.at[i, 'Recall'] = source_data['Basic_shadow.pth'].iloc[1]
    target_data.at[i+2, 'Recall'] = source_data['Basic_shadow.pth'].iloc[2]
    target_data.at[i+4, 'Recall'] = source_data['Basic_shadow.pth'].iloc[3]
    target_data.at[i+6, 'Recall'] = source_data['Basic_shadow.pth'].iloc[4]

    target_data.to_csv(target_file, index=False)


defence_list = ["PEdrop", "label_smoothing", "DPSGD", "RelaxLoss", "adv_reg"]

for i, defence in enumerate(defence_list):
    change_csv("roll_nn", defence, i*8)
    change_csv2("roll_nn", defence, i*8)
    # change_csv("roll_nn", defence, 0)
    # change_csv2("roll_nn", defence, 0)