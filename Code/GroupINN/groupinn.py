import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
import scipy.io
import random
import pandas as pd
import subprocess

def parse_args():
    parser = argparse.ArgumentParser()
	
    parser.add_argument('d', help='dataset', type=str)
    parser.add_argument('c1', help='class1', type=str)
    parser.add_argument('c2', help='class2', type=str)
    parser.add_argument('-t', help = 'threshold', type = int, default = None)
    parser.add_argument('--b', help = 'binary', action='store_true')
    parser.add_argument('--a', help = 'abs value', action='store_true')
    parser.add_argument('--dropout_rate', default = 0)
    parser.add_argument('--c_train', default = 0.5)
    parser.add_argument('--feature_reduction', default = 5)
    return parser.parse_args()

def label(file_name, dir1, dir2):
    if "{}.csv".format(file_name) in dir1:
        return 1
    elif "{}.csv".format(file_name) in dir2:
        return 2


# run run.py -f ER -opc 0  

args  =  parse_args()

dir1 = os.listdir("../Dataset/{}/{}/".format(args.d,args.c1))
dir2 = os.listdir("../Dataset/{}/{}/".format(args.d,args.c2))

c1_nets = [utils.open_file("../Dataset/{}/{}/{}".format(args.d,args.c1, network_file), args.t, args.b, args.a) for network_file in dir1]
c2_nets = [utils.open_file("../Dataset/{}/{}/{}".format(args.d,args.c2, network_file), args.t, args.b, args.a) for network_file in dir2]




if not os.path.exists("Dataset/{}".format(args.d)):
        os.mkdir("m_dataset/{}".format(args.d))

accs = []

for _ in range(5):
    
    obs = [["Subject", "Quartile", "Train", "Test"]]
    obs1 = []
    obs2 = []

    for file, net in zip(dir1,c1_nets):
        scipy.io.savemat("m_dataset/{}/{}.mat".format(args.d,file[:-4]), {"mat":net})
        obs1.append([file[:-4], 1, 0, 0])
        
    for file, net in zip(dir2,c2_nets):
        scipy.io.savemat("m_dataset/{}/{}.mat".format(args.d,file[:-4]), {"mat":net})
        obs2.append([file[:-4], 0, 0, 0])
    
    obs1_train = set(random.sample(range(len(obs1)), int(round(len(obs1)*0.8,0))))
    obs1_test = set(range(len(obs1))).difference(obs1_train)
    
    for elem in obs1_train:
        obs1[elem][2]=1
    
    for elem in obs1_test:
        obs1[elem][2]=1
        obs1[elem][3]=1
    
    
    obs2_train = set(random.sample(range(len(obs2)), int(round(len(obs2)*0.8,0))))
    obs2_test = set(range(len(obs2))).difference(obs2_train)
    
    for elem in obs2_train:
        obs2[elem][2]=1
    
    for elem in obs2_test:
        obs2[elem][2]=1
        obs2[elem][3]=1
        
    obs = obs+obs1+obs2
        
    metadata = pd.DataFrame(obs[1:],columns=obs[0])    
    metadata.to_csv("m_dataset/{}/metadata.csv".format(args.d))



    subprocess.call("python train_model.py --dataset_dir m_dataset/{} --meta_filename m_dataset/{}/metadata.csv  --dropout_rate {} --feature_reduction {} --c_train 1.2".format(args.d,
                    args.d, args.dropout_rate,  args.feature_reduction, args.c_train), shell = True)

    with open("final_acc.txt", "r") as f:
        acc = float(f.readline())
    
    accs.append(acc)
    
    
print(accs)

with open("{}_{}_{}.txt", "w") as f:
    f.write(str(accs))