from caffe import layers as L, params as P
import caffe
from pylab import imshow, axis, subplots, figure, xlabel, ylabel
import numpy as np


def lenet(lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1. / 255), ntop=2)

    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.fc1 = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.fc1, in_place=True)
    n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()


with open('lenet_auto_train.prototxt', 'w') as f:
    f.write(str(lenet('mnist_train_lmdb', 64)))

with open('lenet_auto_test.prototxt', 'w') as f:
    f.write(str(lenet('mnist_test_lmdb', 100)))

[(k, v.data.shape) for k, v in solver.net.blobs.items()]

# just print the weight sizes (we'll omit the biases)
[(k, v[0].data.shape) for k, v in solver.net.params.items()]

solver.net.forward()  # train net
solver.test_nets[0].forward()  # test net (there can be more than one)

% % time
niter = 200
test_interval = 25

# 记录每次反向传播时的损失
train_loss = np.zeros(niter)

# 验证的次数为：训练次数 / 测试间隔 （结果进行上取整）
test_acc = np.zeros(int(np.ceil(niter / test_interval)))

output = np.zeros((niter, 8, 10))

# the main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe

    # 记录训练损失
    train_loss[it] = solver.net.blobs['loss'].data

    # store the output on the first test batch
    # (start the forward pass at conv1 to avoid loading new data)
    solver.test_nets[0].forward(start='conv1')

    # 记录模型输出数字图像的前八个
    output[it] = solver.test_nets[0].blobs['score'].data[:8]

    if it % test_interval == 0:
        print
        'Iteration', it, 'testing...'
        correct = 0
        for test_it in range(100):
            solver.test_nets[0].forward()
            correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)
                           == solver.test_nets[0].blobs['label'].data)
        test_acc[it // test_interval] = correct / 1e4

_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(np.arange(niter), train_loss)
ax2.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))