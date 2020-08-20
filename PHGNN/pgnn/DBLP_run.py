from utils import *
from model_dblp import *
from warnings import filterwarnings
filterwarnings("ignore")
import argparse
from data_dblp import *
import time
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')

parser.add_argument('--seed', type=int, default=64, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--data_path', type=str, default='../data/preprocessed/DBLP_processed/',help='path to data')
parser.add_argument('--conv_name', type=str, default='phgnn')
parser.add_argument('--patience', type=int, default=100, help='Patience')

parser.add_argument('--n_hid', type=int, default=64,help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int, default=8,help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=4,help='Number of GNN layers')
parser.add_argument('--dropout', type=int, default=0.2, help='Dropout ratio')

parser.add_argument('--n_batch', type=int, default=1,help='Number of batch (sampled graphs) for each epoch')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if  args.cuda:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)


features_0,features_1,features_2,features_3,adjM,type_mask,labels=load_DBLP_data()
# print('loading finish')
features_0,features_1,features_2,features_3, node_type, edge_index,h_mat=to_torch(features_0,features_1,features_2,features_3,adjM, type_mask)
# idx=np.random.permutation(len(labels))
# idx_train = idx[:int(len(labels)*0.2)]
# idx_val=  idx[int(len(labels)*0.8):int(len(labels)*0.9)]
# idx_test = idx[int(len(labels)*0.9):]
# np.savez(args.data_path+ 'train_val_test_idx.npz',
#          val_idx=idx_val,
#          train_idx=idx_train,
#          test_idx=idx_test)
train_val_test_idx = np.load(args.data_path + '/train_val_test_idx.npz')
idx_train = train_val_test_idx['train_idx']
idx_val = train_val_test_idx['val_idx']
idx_test= train_val_test_idx['test_idx']

labels = torch.LongTensor(labels)
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)
if args.cuda:
    features_0 = features_0.cuda()
    features_1 = features_1.cuda()
    features_2 = features_2.cuda()
    features_3 = features_3.cuda()
    node_type = node_type.cuda()
    labels = labels.cuda()
    edge_index=edge_index.cuda()
    h_mat=h_mat.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

gnn = GNN(conv_name = args.conv_name, in_dim = [334,4231,50,20],n_hid = args.n_hid,\
          n_heads = args.n_heads, n_layers = args.n_layers, dropout = args.dropout,\
          num_types = 4).to(device)
classifier = Classifier(args.n_hid, 4).to(device)

model = nn.Sequential(gnn, classifier)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay = 0.001)

def train(epoch):

    model.train()
    t = time.time()
    losses_train=[]

    for batch_id in np.arange(args.n_batch):

        node_rep = gnn.forward(features_0,features_1,features_2,features_3, node_type,edge_index,h_mat)
        output = classifier.forward(node_rep[idx_train])
        loss_train = F.nll_loss(output,labels[idx_train])
        acc_train, confusion_matrix, micro_f1, macro_f1 = accuracy(output, labels[idx_train])
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        loss_train.backward()
        losses_train += [loss_train.cpu().detach().tolist()]
        optimizer.step()


    model.eval()
    with torch.no_grad():
        node_rep = gnn.forward(features_0,features_1,features_2,features_3,node_type, edge_index, h_mat)
        output = classifier.forward(node_rep[idx_val])
        loss_val = F.nll_loss(output, labels[idx_val])
        acc_val, confusion_matrix, micro_f1, macro_f1= accuracy(output, labels[idx_val])
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(np.mean(losses_train)),
              'acc_train: {:.4f}'.format(acc_train.data.item()),
              'loss_val: {:.4f}'.format(loss_val.data.item()),
              'acc_val: {:.4f}'.format(acc_val.data.item()),
              'time: {:.4f}s'.format(time.time() - t))
        return loss_val.data.item()


def compute_test():
    model.eval()
    node_rep = gnn.forward(features_0,features_1,features_2,features_3,node_type, edge_index, h_mat)
    output = classifier.forward(node_rep[idx_test])
    loss_test = F.nll_loss(output, labels[idx_test])
    acc_test, confusion_matrix, micro_f1, macro_f1 = accuracy(output, labels[idx_test])
    print("====================Test set results====================")
    print("Loss= {:.4f}".format(loss_test.item()),
          "Accuracy= {:.4f}".format(acc_test.item()))
    print('Confusion Matrix (tn, fp, fn, tp)=', confusion_matrix)
    print('micro_f1=', micro_f1)
    print('macro_f1=', macro_f1)


# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
        torch.save(model, args.conv_name+'.pth')
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break


print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model=torch.load(args.conv_name+'.pth')


# Testing
compute_test()
