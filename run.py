import numpy as np
from dataset import get_dataset, get_handler
from model import get_net
from torchvision import transforms
from tqdm import tqdm
import torch
from options import args_parser
from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
                                LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
                                KMeansSampling, KCenterGreedy, BALDDropout, CoreSet, \
                                AdversarialBIM, AdversarialDeepFool, ActiveLearningByLearning, Balance

# parse args
arg = args_parser()
# parameters
SEED = 1
total_samples = 12000

NUM_INIT_LB = 200
NUM_QUERY = 200
NUM_ROUND = 50

DATA_NAME = 'MNIST'
# DATA_NAME = 'FashionMNIST'
# DATA_NAME = 'SVHN'
# DATA_NAME = 'CIFAR10'

args_pool = {'MNIST':
                {'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
            'FashionMNIST':
                {'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
            'SVHN':
                {'n_epoch': 20, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
            'CIFAR10':
                {'n_epoch': 20, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.05, 'momentum': 0.3}}
            }
args = args_pool[DATA_NAME]

# set seed
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.enabled = False



# load dataset
X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME)
X_tr = X_tr[:total_samples]
Y_tr = Y_tr[:total_samples]

# start experiment
n_pool = len(Y_tr)
n_test = len(Y_te)
print('number of labeled pool: {}'.format(NUM_INIT_LB))
print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB))
print('number of testing pool: {}'.format(n_test))

# generate initial labeled pool
#n_pool = 40000
idxs_lb = np.zeros(n_pool, dtype=bool)

#print(idxs_lb)
idxs_tmp = np.arange(n_pool)
#print(idxs_tmp)
np.random.shuffle(idxs_tmp)
#print(idxs_tmp)
#random select 600 change into labeled, select the first 600 of shuffled dataset
#idxs_lb is which sample is labeled
idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True



# load network
net = get_net(DATA_NAME)
handler = get_handler(DATA_NAME)


if(arg.qs=='random'):
    strategy = RandomSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
if(arg.qs=='balance'):
    strategy = Balance(X_tr, Y_tr, idxs_lb, net, handler, args)
if(arg.qs=='leastconfidence'):
    strategy = LeastConfidence(X_tr, Y_tr, idxs_lb, net, handler, args)
if(arg.qs=='marginsampling'):
    strategy = MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
if(arg.qs=='entropysampling'):
    strategy = EntropySampling(X_tr, Y_tr, idxs_lb, net, handler, args)
if(arg.qs=='leastconfidencedropout'):
    strategy = LeastConfidenceDropout(X_tr, Y_tr, idxs_lb, net, handler, args, n_drop=10)
if (arg.qs == 'marginsamplingdropout'):
    strategy = MarginSamplingDropout(X_tr, Y_tr, idxs_lb, net, handler, args, n_drop=10)
if (arg.qs == 'entropysamplingdropout'):
    strategy = EntropySamplingDropout(X_tr, Y_tr, idxs_lb, net, handler, args, n_drop=10)
if (arg.qs == 'kmeanssampling'):
    strategy = KMeansSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
if (arg.qs == 'kcentergreedy'):
    strategy = KCenterGreedy(X_tr, Y_tr, idxs_lb, net, handler, args)
if (arg.qs == 'balddropout'):
    strategy = BALDDropout(X_tr, Y_tr, idxs_lb, net, handler, args, n_drop=10)
if (arg.qs == 'coreset'):
    strategy = CoreSet(X_tr, Y_tr, idxs_lb, net, handler, args)
if (arg.qs == 'adversarialbim'):
    strategy = AdversarialBIM(X_tr, Y_tr, idxs_lb, net, handler, args, eps=0.05)
if (arg.qs == 'adversarialdeepfool'):
    strategy = AdversarialDeepFool(X_tr, Y_tr, idxs_lb, net, handler, args, max_iter=50)
if (arg.qs == 'activelearningbylearning'):
    albl_list = [MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args),
                 KMeansSampling(X_tr, Y_tr, idxs_lb, net, handler, args)]
    strategy = ActiveLearningByLearning(X_tr, Y_tr, idxs_lb, net, handler, args, strategy_list=albl_list, delta=0.1)
print(strategy)
# print info

print(DATA_NAME)
print('SEED {}'.format(SEED))
print(type(strategy).__name__)

# round 0 accuracy
strategy.train()
P = strategy.predict(X_te, Y_te)
acc = np.zeros(NUM_ROUND+1)
acc[0] = 1.0 * (Y_te==P).sum().item() / len(Y_te)
print('Round 0\ntesting accuracy {}'.format(acc[0]))
#print('Round 0\ntesting accuracy {}'.format(acc[0]),file=f)

for rd in tqdm(range(1, NUM_ROUND+1)):
    print('Round {}'.format(rd))

    # query
    if(arg.qs=='balance'):
        #求Balance
        balance = [1,1,1,1,1,1,1,1,1,1]
        balance_tmp=[0,0,0,0,0,0,0,0,0,0]
        for i in range(total_samples):
            if(idxs_lb[i]==True):
                balance_tmp[Y_tr[i]] = balance_tmp[Y_tr[i]]+1
        print(balance_tmp)
        total = sum(balance_tmp)
        for i in range(10):
            balance[i] = 2-(10*balance_tmp[i]/total)
        print(balance)
        q_idxs = strategy.query(NUM_QUERY, balance)
        idxs_lb[q_idxs] = True
    else:
        balance = [1,1,1,1,1,1,1,1,1,1]
        balance_tmp=[0,0,0,0,0,0,0,0,0,0]
        for i in range(total_samples):
            if(idxs_lb[i]==True):
                balance_tmp[Y_tr[i]] = balance_tmp[Y_tr[i]]+1
        print(balance_tmp)
        total = sum(balance_tmp)
        for i in range(10):
            balance[i] = 2-(10*balance_tmp[i]/total)
        print(balance)
        q_idxs = strategy.query(NUM_QUERY)
        idxs_lb[q_idxs] = True

    # update
    strategy.update(idxs_lb)
    strategy.train()

    # round accuracy
    P = strategy.predict(X_te, Y_te)
    acc[rd] = 1.0 * (Y_te==P).sum().item() / len(Y_te)
    print('testing accuracy {}'.format(acc[rd]))


# print results
print('SEED {}'.format(SEED))
print(type(strategy).__name__)
print(acc)
f = open('./test.txt', 'a')
print(args, file=f)
print(acc,file=f)
f.close()
