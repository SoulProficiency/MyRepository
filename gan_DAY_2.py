import tensorflow as tf
import numpy as np
from pylab import mpl
import matplotlib.pyplot as plt
import math

plt.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.sans-serif'] = ['SimHei']
# 同时我们要在中文前面加上u

# 随机种子
tf.set_random_seed(1)
np.random.seed(1)

# 设置超参数
BATCH_SIZE = 64  # 批量大小
LR_G = 0.0001  # 生成器的学习率
LR_D = 0.0001  # 判别器的学习率
N_IDEAS = 5  # 假设生成了5种不同的信号（5种初始化曲线）
num_sample = 15  # 用15个点绘制拟合包络（波形）的形状
# 列表解析式代替了for循环，PAINT_POINTS.shape=(64,15),
# np.vstack()默认逐行叠加（axis=0）
input_data = np.vstack([np.linspace(-2 * math.pi, 2 * math.pi, num_sample) for _ in range(BATCH_SIZE)])  # (64, 15)


def generate_signal():
    # 基本的正弦序列，不包括相位偏移或者幅度畸变
    signal_shape = np.sin(input_data)
    return signal_shape


with tf.variable_scope('Generator'):
    """网络结构batch_size --->128--->num_sample"""
    G_in = tf.placeholder(tf.float32, [None, N_IDEAS])  # 随机的ideals（来源于正态分布）
    G_l1 = tf.layers.dense(G_in, 128, tf.nn.sigmoid)
    G_out = tf.layers.dense(G_l1, num_sample)  # 生成一副生成信号的包络（15个数据点）
with tf.variable_scope('Discriminator'):
    """判别器与生成器不同，生成器只需要输入生成的信号数据，它无法接触到真实信号，
    如果能接触到真实信号数据，那就不用学习了，直接导入到判别器就是0.5的概率，换句话说，
    生成器只能通过生成器的误差反馈来调节权重(优化方法可以是梯度下降，随机梯度下降，批量梯度下降，动量下降，或者Adam)，
    使得逐渐生成逼真实的信号波形出来。"""

    # 接受真实的信号
    true_signal = tf.placeholder(tf.float32, [None, num_sample], name='real_in')
    """
    回到神经网络的基本知识，我们输入矩阵是一个 （ 特征*样本数 ）的矩阵
    在使用优化算法的时候，如果选择梯度下降，动量下降或者Adam（虽然这两者基于梯度下降）
    他们都会遍历所有的训练样本，然后在进行网络权值的优化，所耗费的时间长
    为了让网络学习的更好，可以多按照真实的信号波形产生一些抽样得到的样本来优化网络
    """
    # 将真实信号输入到判别器，判别器判断这些信号数据来自于真实信号的概率
    D_l0 = tf.layers.dense(true_signal, 128, tf.nn.relu, name='Discriminate')
    prob_from_true_signal = tf.layers.dense(D_l0, 1, tf.nn.sigmoid, name='out')

    # 输入生成的信号数据，G_out代入到判别器中。
    D_l1 = tf.layers.dense(G_out, 128, tf.nn.relu, name='Discriminate', reuse=True)

    # 代入生成的数据，判别器判断这数据来自生成器的概率
    prob_from_generate_signal = tf.layers.dense(D_l1, 1, tf.nn.sigmoid, name='out', reuse=True)
    """注意到，判别器中当输入生成的信号时，这层是可以重复利用的，通过动态调整这次的权重来完成判别器的loss最小，关键一步。"""
# 判别器loss，此时需同时优化两部分的概率

# 使用交叉熵损失
D_loss = -tf.reduce_mean(tf.log(prob_from_true_signal) + tf.log(1 - prob_from_generate_signal))

# 对于生成器的loss，此时prob_from_true_signal是固定的，可以看到生成器并没有输入真实的信号数据，
# 所以tf.log(prob_artist0)是一个常数，故在这里不用考虑。
G_loss = tf.reduce_mean(tf.log(1 - prob_from_generate_signal))
train_D = tf.train.AdamOptimizer(LR_D).minimize(
    D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))
train_G = tf.train.AdamOptimizer(LR_G).minimize(
    G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))

# 以下两步是TensorFlow必须要做的
sess = tf.Session()
sess.run(tf.global_variables_initializer())

plt.ion()  # 连续画图
for step in range(5000):
    every_step_generate_signal = generate_signal()  # 真实信号数据，每一轮真实信号的数据都是随机生成的！
    G_ideas = np.random.randn(BATCH_SIZE, N_IDEAS)  # 生成信号的5个可行方法
    '''
    再次回归到最初的神经网络，我们的输入矩阵是一个（特征*样本数的矩阵）维矩阵
    损失函数不论使用梯度下降，还是动量下降或Adam优化（虽说这两者是基于梯度下降算法的）
    都是遍历了所有的样本，学习了所有的样本对神经网络的训练是有好处的
    '''
    G_signal, pa0, Dl = sess.run([G_out, prob_from_true_signal, D_loss, train_D, train_G],
                                 {G_in: G_ideas, true_signal: every_step_generate_signal})[:3]  # 训练和获取结果

    if step % 100 == 0:  # 每100次训练画一次图
        plt.cla()
        plt.plot(input_data[0], G_signal[0], c='black', lw=3, label='真实信号包络')
        plt.plot(input_data[0], every_step_generate_signal[0], c='red', lw=3, label='生成信号的包络')
        plt.text(-.5, -2.5, '第{}次训练'.format(step), fontdict={'size': 15})
        plt.text(-.5, -1.3, 'D mean accuracy=%.2f ' % pa0.mean(), fontdict={'size': 15})
        # -1.38 for G to converge
        plt.text(-.5, -1.5, 'D score= %.2f ' % -Dl, fontdict={'size': 15})
        plt.ylim((-2, 3))
        plt.xlim((-2 * math.pi - 1, 2 * math.pi + 1))
        plt.legend(loc='upper right', fontsize=12)
        plt.draw()
        plt.pause(0.01)

plt.ioff()
plt.show()
