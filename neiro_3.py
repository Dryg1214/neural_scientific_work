import fastbook
fastbook.setup_book()
from fastai.vision.all import *
from fastbook import *
from pandas.core.base import PandasObject

matplotlib.rc('image', cmap='Greys')

path = untar_data(URLs.MNIST_SAMPLE)
Path.BASE_PATH = path

print(path.ls())

# (path/'train').ls()
threes = (path/'train'/'3').ls().sorted()
sevens = (path/'train'/'7').ls().sorted()


#print(threes)
im3_path = threes[1]
im3 = Image.open(im3_path)
#im3.show()

im3_t = tensor(im3)
df = pd.DataFrame(im3_t[4:15,4:22])
#df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greys')
df.style.background_gradient('Greys')
df

seven_tensors = [tensor(Image.open(o)) for o in sevens]
three_tensors = [tensor(Image.open(o)) for o in threes]
print(len(three_tensors))
print(len(seven_tensors))

show_image(three_tensors[2])
show_image(three_tensors[3])
#plt.show()
#print("kek")
#three_tensors[1].show()
stacked_sevens = torch.stack(seven_tensors).float()/255
stacked_threes = torch.stack(three_tensors).float()/255

print(stacked_threes[0])


print(str(stacked_threes.shape) + "stacked_threes")

#tensor's rank
print(stacked_threes.ndim)

mean3 = stacked_threes.mean(0)
mean7 = stacked_sevens.mean(0)
show_image(mean7)
show_image(mean3)
#plt.show()

a_3 = stacked_threes[1]
dist_3_abs = (a_3 - mean3).abs().mean()
dist_3_sqr = ((a_3 - mean3)**2).mean().sqrt()

dist_7_abs = (a_3 - mean7).abs().mean()
dist_7_sqr = ((a_3 - mean7)**2).mean().sqrt()

#print(dist_3_abs)
#print(dist_3_sqr)

# loss functions.
print(F.l1_loss(a_3.float(),mean7))# аналог abs
print(F.mse_loss(a_3,mean7).sqrt())# ско

valid_3_tens = torch.stack([tensor(Image.open(o)) 
                            for o in (path/'valid'/'3').ls()])
valid_3_tens = valid_3_tens.float()/255
valid_7_tens = torch.stack([tensor(Image.open(o)) 
                            for o in (path/'valid'/'7').ls()])
valid_7_tens = valid_7_tens.float()/255

#print(valid_3_tens.shape, valid_7_tens.shape)

def mnist_distance(a,b): return (a-b).abs().mean((-1,-2))

print("tensor mean3 - " + str(mnist_distance(a_3, mean3)))
print("tensor mean7 - " + str(mnist_distance(a_3, mean7)))

valid_3_dist = mnist_distance(valid_3_tens, mean3)
print(valid_3_dist, valid_3_dist.shape)

def is_3(x): return mnist_distance(x,mean3) < mnist_distance(x,mean7)

print(is_3(a_3), is_3(a_3).float())

# Точность
accuracy_3s = is_3(valid_3_tens).float().mean()
accuracy_7s = (1 - is_3(valid_7_tens).float()).mean()

print(accuracy_3s,accuracy_7s,(accuracy_3s+accuracy_7s)/2)

def pr_eight(x,w): return (x*w).sum()


train_x = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28*28)
train_y = tensor([1]*len(threes) + [0]*len(sevens)).unsqueeze(1)
print(train_x.shape, train_y.shape)

valid_x = torch.cat([valid_3_tens, valid_7_tens]).view(-1, 28*28)
valid_y = tensor([1]*len(valid_3_tens) + [0]*len(valid_7_tens)).unsqueeze(1)
valid_dset = list(zip(valid_x,valid_y))

#Initially weight for every pixel
def init_params(size, std=1.0): return (torch.randn(size)*std).requires_grad_()# random num

def linear1(xb): return xb@weights + bias

def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1, 1-predictions, predictions).mean()

weights = init_params((28*28,1))
bias = init_params(1)

dset = list(zip(train_x,train_y))

dl = DataLoader(dset, batch_size=256)
xb,yb = first(dl)
print(xb.shape,yb.shape)

valid_dl = DataLoader(valid_dset, batch_size=256)
batch = train_x[:4] #мини - пакеты

preds = linear1(batch)

def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward()

calc_grad(batch, train_y[:4], linear1)
print(weights.grad.mean(), bias.grad)


def train_epoch(model, lr, params):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        for p in params:
            p.data -= p.grad*lr
            p.grad.zero_()

            
#print((preds>0.0).float() == train_y[:4])

#validation accuracy - точность валидации данных
def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean()

#print(batch_accuracy(linear1(batch), train_y[:4]))

def validate_epoch(model):
    accs = [batch_accuracy(model(xb), yb) for xb,yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)

#print(validate_epoch(linear1))


# Первая эпоха
"""
lr = 1.
params = weights,bias
train_epoch(linear1, lr, params)
print(validate_epoch(linear1))

for i in range(20):
    train_epoch(linear1, lr, params)
    print(validate_epoch(linear1), end=' ')
"""
lr = 1. #learning rate
linear_model = nn.Linear(28*28,1)

class BasicOptim:
    def __init__(self,params,lr): self.params,self.lr = list(params),lr

    def step(self, *args, **kwargs):
        for p in self.params: p.data -= p.grad.data * self.lr

    def zero_grad(self, *args, **kwargs):
        for p in self.params: p.grad = None

opt = BasicOptim(linear_model.parameters(), lr)

def train_epoch(model):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        opt.step()
        opt.zero_grad()

validate_epoch(linear_model)

def train_model(model, epochs):
    for i in range(epochs):
        train_epoch(model)
        print(validate_epoch(model), end=' ')

train_model(linear_model, 20)

if __name__ == '__main__':
    # Library Basic Optimizator (our class)
    linear_model = nn.Linear(28*28,1)
    opt = SGD(linear_model.parameters(), lr)
    train_model(linear_model, 20)


    #Learner
    dls = DataLoaders(dl, valid_dl)
    learn = Learner(dls, nn.Linear(28*28,1), opt_func=SGD,
                    loss_func=mnist_loss, metrics=batch_accuracy)

    learn.fit(10, lr=lr)

    # w - тензоры весов 
    # b - тензоры смещений
    w1 = init_params((28*28,30))
    b1 = init_params(30)
    w2 = init_params((30,1))
    b2 = init_params(1)


    def simple_net(xb): 
        res = xb@w1 + b1
        res = res.max(tensor(0.0)) #  каждое отрицательное число заменяется нулем
        res = res@w2 + b2
        return res




    simple_net = nn.Sequential(
        nn.Linear(28*28,30),
        nn.ReLU(),
        nn.Linear(30,1)
    )

    learn = Learner(dls, simple_net, opt_func=SGD,
                    loss_func=mnist_loss, metrics=batch_accuracy)

    learn.fit(40, 0.1)


    plt.plot(L(learn.recorder.values).itemgot(2));

    # Окончательная точность
    print(learn.recorder.values[-1][2])

    print("18 layer")
    # 18-layer model
    dls = ImageDataLoaders.from_folder(path)
    learn = vision_learner(dls, resnet18, pretrained=False,
                        loss_func=F.cross_entropy, metrics=accuracy)
    learn.fit_one_cycle(1, 0.1)

# 0         0.176130    0.087036    0.987733  05:37


"""
#Прогноз для одного изображения
(train_x[0]*weights.T).sum() + bias
preds = linear1(train_x)
corrects = (preds>0.0).float() == train_y
print(corrects.float().mean().item()) # 0.5379
# Проверим при изменении одного из весов
with torch.no_grad(): weights[0] *= 1.0001
preds = linear1(train_x)
print(((preds>0.0).float() == train_y).float().mean().item())
#def sigmoid(x): return 1/(1+torch.exp(-x))
"""