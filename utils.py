import random
import numpy as np
from functools import reduce
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity

import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def common(a,b):
    c = [value for value in a if value in b]
    return c

def exclude(a,b):
    c = [value for value in a if value not in b]
    return c

def build_groups_by_q(trainset, q, num_class = 10):
    groups=[]
    for _ in range(num_class):
      groups.append([])
    for img,lable in trainset:
      if random.random() < (q-0.1)*num_class /(num_class-1):
        groups[lable].append((img,lable))
      else:
        groups[random.randint(0, num_class-1)].append((img,lable))
    return groups

def check_attack(cids,att_ids):
    return  np.array([(id in att_ids) for id in cids]).any()

def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    for i in range(len(parameters)):
        if len(parameters[i].shape) == 0:
            parameters[i] = np.asarray([parameters[i]])
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def weights_to_vector(weights):
    """Convert NumPy weights to 1-D Numpy array."""
    Lis=[np.ndarray.flatten(ndarray) for ndarray in weights]
    return np.concatenate(Lis, axis=0)

def vector_to_weights(vector,weights):
    """Convert 1-D Numpy array tp NumPy weights."""
    indies = np.cumsum([0]+[layer.size for layer in weights]) #indies for each layer of a weight
    Lis=[np.asarray(vector[indies[i]:indies[i+1]]).reshape(weights[i].shape) for i in range(len(weights))]
    return Lis

def craft(old_weights, new_weights, action, b):
    weight_diff = [w1-w2 for w1,w2 in zip(old_weights, new_weights)] #weight_diff = grad*lr here
    crafted_weight_diff = [b*diff_layer* action for diff_layer in weight_diff]
    vec_weight_diff = weights_to_vector(crafted_weight_diff)
    crafted_weight_diff = vector_to_weights(vec_weight_diff, old_weights)
    crafted_weight = [w1-w2 for w1,w2 in zip(old_weights, crafted_weight_diff)] #old_weight - lr*gradient
    return crafted_weight


def train_real(net, trainloader, epochs, lr):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    for _ in range(epochs):
        images, labels = next(iter(trainloader))
        labels = labels.type(torch.LongTensor)
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(net(images), labels)
        loss.backward()
        optimizer.step()


def test(net, valloader):
    """Validate the network on the 10% training set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in valloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
    return loss, accuracy

'''
          ----- Server Aggregation Methods -----
'''

def average(new_weights):
        fractions=[1/len(new_weights) for _ in range(len(new_weights))]
        fraction_total=np.sum(fractions)
        # Create a list of weights, each multiplied by the related fraction
        weighted_weights = [
            [layer * fraction for layer in weights] for weights, fraction in zip(new_weights, fractions)
        ]
        # Compute average weights of each layer
        aggregate_weights = [
            reduce(np.add, layer_updates) / fraction_total
            for layer_updates in zip(*weighted_weights)
        ]
        return aggregate_weights


def Krum(old_weight, new_weights, num_round_attacker):
    """Compute Krum average."""
    grads=[]
    for new_weight in new_weights:
        grad = [layer_old_weight-layer_new_weight for layer_old_weight,layer_new_weight in zip(old_weight, new_weight)]
        grads.append(grad)

    scrs=[]
    for i in grads:
        scr=[]
        for j in grads:
            dif=weights_to_vector(i)-weights_to_vector(j)
            sco=np.linalg.norm(dif)
            scr.append(sco)
        top_k = sorted(scr)[1:len(grads)-2-num_round_attacker]
        scrs.append(sum(top_k))
    chosen_grads= grads[scrs.index(min(scrs))]
    krum_weights = [w1-w2 for w1,w2 in zip(old_weight, chosen_grads)]
    return krum_weights

def Clipping_Median(old_weights, new_weights):
    max_norm=2
    grads=[]
    for new_weight in new_weights:
        norm_diff=np.linalg.norm(weights_to_vector(old_weights)-weights_to_vector(new_weight))
        clipped_grad = [(layer_old_weight-layer_new_weight)*min(1,max_norm/norm_diff) for layer_old_weight,layer_new_weight in zip(old_weights, new_weight)]
        grads.append(clipped_grad)

    med_grad=[]
    for layer in range(len(grads[0])):
        lis=[]
        for weight in grads:
            lis.append(weight[layer])
        arr=np.array(lis)
        med_grad.append(np.median(arr,axis=0))

    Centered_weights=[w1-w2 for w1,w2 in zip(old_weights, med_grad)]
    return Centered_weights

def CosDefense(old_weight, new_weights):

    global_last_layer = old_weight[-2].reshape(-1)
    last_layer_grads=[]
    for new_weight in new_weights:
        last_layer_grads.append(new_weight[-2].reshape(-1) - global_last_layer)

    scores = np.abs(cosine_similarity(np.array(last_layer_grads), [global_last_layer]).reshape(-1))
    min_score = np.min(scores)
    scores = (scores - min_score) / (np.max(scores) - min_score)
    threshold = np.mean(scores) #+ 0.5 * np.std(scores)
    benign_indices = scores < threshold

    weight =  1 / (sum(benign_indices))
    fractions = benign_indices * weight
    # Create a list of weights, each multiplied by the related fraction
    weighted_weights = [
        [layer * fraction for layer in weights] for weights, fraction in zip(new_weights, fractions)
    ]

    # Compute average weights of each layer
    aggregate_weights = [
        reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)
    ]

    return aggregate_weights, scores