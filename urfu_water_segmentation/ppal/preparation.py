import os
import shutil
from pathlib import Path
import pandas as pd 

def preparation():
    folder_path = "./data/glh_water/cache"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  
        except Exception as e:
            print(f'Не удалось удалить {file_path}. Причина: {e}')

    

    with open('./config_landcover.py', 'r', encoding='utf-8') as file:
        lines = file.readlines()

    lines[82] =  "checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/poolformer/poolformer-m48_3rdparty_32xb128_in1k_20220414-9378f3eb.pth' \n"

    with open('./config_landcover.py', 'w', encoding='utf-8') as file:
        file.writelines(lines)


    

    labeled_file = './data/glh_water/labeled.csv'
    unlabeled_file = './data/glh_water/unlabeled.csv'

    shutil.copy(labeled_file, folder_path)
    shutil.copy(unlabeled_file, folder_path)



    




    









    


    