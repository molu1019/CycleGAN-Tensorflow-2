import json                                                         # this module is used to read json files
import os                                                           # this modules is used to get all the names of your json files and rename them
import glob

folder_path = "/mnt/robolab/data/Bilddaten/GAN_train_data_sydavis-ai/SydavisAI/Blattfeder/train"                                         # you specifie the folder conatining your json files

#files = os.listdir(folder_path)                                   # you all the files names in the specified folder
files = glob.glob(os.path.join(folder_path, '*.json'))


for file in files:                                                  # for each file in the folder
    with open(f"{file}", "r") as f:                   # you open the file
        json_data = json.load(f)   
    # fuer 2. Stufe                                 # you get the infos stored in the json file as a dictionary
#   for item in json_data['imagePath']:
#       item['label'] = item['label'].replace('..\\train\\', '')
    #fuer erste Stufe
    json_data['imagePath'] = json_data['imagePath'].replace("..\\blattfeder\\", '')

    with open(f"{file}", "w") as fil:
        json.dump(json_data, fil)
print(f"Rename was successful")