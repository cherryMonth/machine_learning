! cp t* /opt/caffe/data/mnist/  # 复制数据文件到caffe的数据目录下
! cp /opt/caffe/examples/mnist/lenet_auto_solver.prototxt ~/lenet  # 复制caffe样例的求解文件到用户目录下
# 运行caffe自带的转换脚本生成lmdb数据文件
! cd /opt/caffe && examples/mnist/create_mnist.sh && cp -r examples/mnist/mnist_test_lmdb ~/lenet/ && cp -r examples/mnist/mnist_train_lmdb ~/lenet/