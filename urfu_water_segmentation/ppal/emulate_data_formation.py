import os
import pandas as pd
import shutil
from pathlib import Path



def update_labeled():

    diverse_csv = "./data/glh_water/cache/diverse_candidates.csv"  
    all_data_csv = "./data/glh_water/all_data.csv"            
    labeled_csv = "./data/glh_water/cache/labeled.csv"             
    unlabaled_csv = "./data/glh_water/cache/unlabeled.csv"

    df_candidates = pd.read_csv(diverse_csv)


    df_all = pd.read_csv(all_data_csv)
    print(f"Всего записей в all_data.csv: {len(df_all)}")

    selected_rows = []


    df_unlabaled = pd.read_csv(unlabaled_csv)
    l = len(df_unlabaled)
    df_selected = df_candidates['index'].tolist()

    for cid in df_selected:
        match = df_all[df_all['index'] == cid]
        if match.empty:
            match = df_all[df_all['img_path'].astype(str).str.contains(str(cid))]
        if not match.empty:
            selected_rows.append(match)
    if selected_rows:
        df_selected = pd.concat(selected_rows, ignore_index=True)
 
    df_selected.to_csv(labeled_csv, index=False)
    ids = df_selected['index'].to_list()
    df_unlabel = pd.read_csv(unlabaled_csv)
    df_unlabel = df_unlabel[~df_unlabel['index'].isin(ids)]
    df_unlabel.to_csv(unlabaled_csv,index=False)


def update_checkpoint():
    with open('./config_landcover.py', 'r', encoding='utf-8') as file:
        lines = file.readlines()
    experiment_path = Path("./logs/GLH_AL_max__Poolformer_ALDataset_512_FocalLoss_AdamW_bsize_128")

    weights_path = (experiment_path / "last_checkpoint").read_text().strip()
    lines[82] =  f"checkpoint_file = '{weights_path}'\n"

    with open('./config_landcover.py', 'w', encoding='utf-8') as file:
        file.writelines(lines)

    print("checkpoint обновлен")

def clean_mask_folder():
    folder_path = "./data/cache/mask"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  
        except Exception as e:
            print(f'Не удалось удалить {file_path}. Причина: {e}')

    


    

