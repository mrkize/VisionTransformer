import tim
import torch
import matplotlib.pyplot as plt # for showing handwritten digits
from torchvision import datasets, transforms

from mymodel import ViT
from utils import MyConfig
from dataloader import get_loader
from mymodel import ViT_mask
from umap import UMAP

config = MyConfig.MyConfig(path="config/cifar10/")
config.set_subkey('patch', 'image_size', 32)
config.set_subkey('patch', 'patch_size', 4)
config.set_subkey('patch', 'num_patches', 64)
config.set_subkey('patch', 'embed_dim', 192)
model = ViT.creat_VIT(config)
Transform = transforms.Compose([transforms.Resize([32, 32]),transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
train_set = datasets.ImageFolder('../data/cifar-10/train', Transform)
val_set = datasets.ImageFolder('../data/cifar-10/val', Transform)
# data_loader, data_size = get_loader('cifar100', config, is_target=True)
seq_ord = list(torch.arange(64))
PE = model.pos_embedding.data[1:,:]
patch = model.to_patch_embedding
ba_id = 1000
memlist = []
nomemlist = []
for i in range(1000):
    data, _ = train_set[i]
    memlist.append(data)
    data, _ = val_set[i]
    nomemlist.append(data)

# for idx,(data,label) in enumerate(data_loader['train']):
    # a = data
    # a = patch(data)
    # a = patch(data) + PE
    # a = model(data)
    # memlist.append(a.reshape(a.shape[0],-1))
    # if idx == ba_id:
    #     break

mem = torch.stack(memlist,dim=0)

# for idx,(data,label) in enumerate(data_loader['val']):
    # b = data
    # b = patch(data)
    # b = patch(data) +PE
    # b = model(data)
    # name = 'res'
    # nomemlist.append(b.reshape(b.shape[0],-1))
    # if idx == ba_id:
    #     break
nomem = torch.stack(nomemlist,dim=0)


name = 'res'
x = torch.cat([mem, nomem], dim=0)
if name == 'patch':
    x = patch(x)
elif name == 'pe':
    x = patch(x)+PE
else:
    x = model(x)
x = x.reshape(x.shape[0],-1).detach().numpy()

color = ['b', 'coral', 'peachpuff', 'sandybrown', 'linen', 'tan', 'orange', 'gold', 'darkkhaki', 'yellow', 'chartreuse', 'green', 'turquoise', 'skyblue']
# Configure UMAP hyperparameters
# model = model.load_VIT('./Network/VIT_Model_cifar10/VIT_PE.pth')

# PE = model.pos_embed[0].detach().numpy()
# PE = model.pos_embed[0,:,:].detach().numpy()
# print(PE.shape)
reducer = UMAP(n_neighbors=15, # default 15, The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
               n_components=2, # default 2, The dimension of the space to embed into.
               n_epochs=1000, # default None, The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result in more accurate embeddings.
              )

# Fit and transform the data
X_trans = reducer.fit_transform(x)

# Check the shape of the new data
# print('Shape of X_trans: ', X_trans.shape)
fig = plt.figure( figsize=(40,24), dpi=160 )
# plt.scatter(mem_trans[0,0],mem_trans[0,1], c='r')
for i in range(2):
    plt.scatter(X_trans[1000*i:1000*(i+1),0],X_trans[1000*i:1000*(i+1),1], c=color[i])

# for i in range(PE.shape[0]):
#     plt.annotate(str(i), xy = (X_trans[i,0], X_trans[i,1]), xytext = (X_trans[i,0]+0.05, X_trans[i,1]+0.05))


# z = range(40)
# x_label = ['11:{}'.format(i) for i in x]
# plt.xticks( x[::5], x_label[::5])
# plt.yticks(z[::5])  #5是步长


plt.grid(True, linestyle='--', alpha=0.5)


plt.xlabel('X')
plt.ylabel('Y')
plt.title('mem and nomem')


plt.savefig('{}_{}.png'.format(ba_id,name))
plt.close(fig)
