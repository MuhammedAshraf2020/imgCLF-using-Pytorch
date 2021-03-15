import os
import pandas as pd
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--name")
parser.add_argument("--save")

args = parser.parse_args()
name = args.name
save = args.save

data = pd.DataFrame(columns = ["image" , "class"] , index = None)
path = os.path.join(os.getcwd() , name)
for index , img_name in tqdm(enumerate(os.listdir(path))):
	img_path = os.path.join(path , img_name)
	if img_name[:3] == "cat":
		data.loc[index] = [img_path , 0]
	else:
		data.loc[index] = [img_path , 1]

data.to_csv("{save_name}.csv".format(save_name = save))
