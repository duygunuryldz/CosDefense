import os
import sys
import csv
import random
import numpy as np

import torch
from torchvision import transforms
import torchvision.datasets as datasets

from utils import *
from models import ResNet18, MNISTClassifier

torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = str(1)


if __name__ == "__main__":
    
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    setup = dict(device=DEVICE, dtype=torch.float)

    mode = "cosDefense" # in ["krum", "cosDefense", "clipping_median", "no_defense"]
    attack_mode = "ipm" # in ["no_attack", "ipm"]

    dataset = "cifar10" #in ["cifar10", "mnist", "fmnist"]
    batch_size = 128
    num_class = 10 #same for all datasets
    q = 0.5  # i.i.d level, 0.1 means i.i.d, higher means non-i.i.d

    num_clients= 100
    num_attacker= 30
    subsample_rate= 0.1

    fl_rounds=1000
    lr=0.01
    norm="1"

    # Global initialization
    torch.cuda.init()

    #Load data
    if "mnist" in dataset:
      apply_transform = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize((0.1307,), (0.3081,))])

      trainset = datasets.MNIST(root='./data', train=True, download=True, transform=apply_transform)
      testset = datasets.MNIST(root='./data', train=False, download=True, transform=apply_transform)

    elif "fmnist" in dataset:
      apply_transform = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize((0.2859,), (0.3530,))])
      
      trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=apply_transform)
      testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=apply_transform)

    elif "cifar10" in dataset:
      transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
      trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
      testset = datasets.CIFAR10(root='./data',train=False, download=True, transform=transform)

    else:
      print("Invalid dataset! Choose from: \"cifar10\", \"mnist\", \"fmnist\"")
      sys.exit()


    #Create Dataloaders
    groups=build_groups_by_q(trainset, q)
    trainloaders=[]
    num_group_clients=int(num_clients/num_class)
    for gid in range(num_class):
        num_data=int(len(groups[gid])/num_group_clients)
        for cid in range(num_group_clients):
            ids = list(range(cid*num_data, (cid+1)*num_data))
            client_trainset = torch.utils.data.Subset(groups[gid], ids)
            trainloaders.append(torch.utils.data.DataLoader(client_trainset, batch_size=batch_size, shuffle=True, drop_last=True))
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, drop_last=False)

    #randomly select attacker ids
    for seed in [1001]:
      random.seed(seed)
      att_ids=random.sample(range(num_clients),num_attacker)
      att_ids=list(np.sort(att_ids, axis = None))
      print('attacker ids: ', att_ids)

      #Create Network
      if "mnist" in dataset:
        net = torch.load("mnist_init").to(**setup)
      else:
        net = ResNet18().to(**setup)
        

      #Do Simulaiton
      if "no_attack" in attack_mode:
          print("----------No Attack Train--------------")
          acc_list = []
          old_weights = get_parameters(net)
          for rnd in range(fl_rounds):
              print('---------------------------------------------------')
              print('rnd: ',rnd+1)
              random.seed(rnd)
              cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
              while len(common(cids, att_ids)) == int(num_clients*subsample_rate):
                  cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
              print('chosen clients: ', cids)
              weights_lis=[]

              for cid in cids:  #if there is an attack
                  set_parameters(net, old_weights)
                  train_real(net, trainloaders[cid], epochs=1, lr=lr)
                  new_weight=get_parameters(net)
                  weights_lis.append(new_weight)

              #aggregation
              if "krum" in mode:
                old_weights = Krum(old_weights, weights_lis, 0)
              elif "median" in mode:
                old_weights=Clipping_Median(old_weights, weights_lis)
              elif "no_defense" in mode:
                old_weights = average(weights_lis)
              elif "cos" in mode:
                old_weights, _ =CosDefense(old_weights, weights_lis)
              else:
                  print("Defense wrong!")
                  sys.exit()

              set_parameters(net, old_weights)
              loss, acc = test(net, testloader)
              print('global_acc: ', acc, 'loss: ', loss)
              acc_list.append(acc)

          f = open(str(num_attacker)+mode+str(q)+"NA.csv", 'w')
          writer = csv.writer(f)
          writer.writerow(acc_list)
          f.close()


      elif "ipm" in attack_mode: 
          print("----------IPM Train--------------")
          ipm_acc = []
          old_weights = get_parameters(net)
          for rnd in range(fl_rounds):
              print('---------------------------------------------------')
              print('rnd: ',rnd+1)
              random.seed(rnd)
              weights_lis=[]
              cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
              while len(common(cids, att_ids)) == int(num_clients*subsample_rate):
                  cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
              print('chosen clients: ', cids)
              print('selected attackers: ',common(cids, att_ids))
              
              client_list = exclude(cids,att_ids) if rnd >= 200 else cids
              for cid in client_list:  #if there is an attack
                  set_parameters(net, old_weights)
                  train_real(net, trainloaders[cid], epochs=1, lr=lr)
                  new_weight=get_parameters(net)
                  weights_lis.append(new_weight)

              #IPM
              if rnd >= 200 and check_attack(cids, att_ids):
                  for i in range(len(common(cids, att_ids))):
                      set_parameters(net, old_weights)
                  if len(weights_lis)!=0:
                      crafted_weights = [craft(old_weights, average(weights_lis), 5, -1)]*len(common(cids, att_ids))
                  else:
                      crafted_weights = [craft(old_weights, get_parameters(net), 5, -1)]*len(common(cids, att_ids))
                  for new_weight in crafted_weights:
                      weights_lis.append(new_weight)

              #aggregation
              if rnd < 200:
                old_weights = average(weights_lis)
              else:
                if "krum" in mode:
                  old_weights = Krum(old_weights, weights_lis, len(common(cids,att_ids)))
                elif "median" in mode:
                  old_weights=Clipping_Median(old_weights, weights_lis)
                elif "no_defense" in mode:
                  old_weights = average(weights_lis)
                elif "cos" in mode:
                  old_weights, cs =CosDefense(old_weights, weights_lis)
                else:
                    print("Defense wrong!")
                    sys.exit()

              set_parameters(net, old_weights)
              loss, acc = test(net, testloader)
              print('global_acc: ', acc, 'loss: ', loss)
              ipm_acc.append(acc)

          f = open(str(num_attacker)+mode+str(q)+"ipm.csv", 'w')
          writer = csv.writer(f)
          writer.writerow(ipm_acc)
          f.close()
      else:
          print("Invalid attack mode! Select from \'no_attack\' or \'IPM\'")
          sys.exit()
        
