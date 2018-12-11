import tensorflow as tf
import gym
import numpy as np


class QNet:  # 估计值
    def __init__(self):
        self.w1 = tf.Variable(tf.truncated_normal([4, 30], stddev=0.1))  # 状态是四个数字
        self.b1 = tf.Variable(tf.zeros([30]))

        self.w2 = tf.Variable(tf.truncated_normal([30, 30], stddev=0.1))
        self.b2 = tf.Variable(tf.zeros([30]))

        self.w3 = tf.Variable(tf.truncated_normal([30, 2], stddev=0.1))  # 输出有两个动作 两个Q值
        self.b3 = tf.Variable(tf.zeros([2]))

    def forward(self, observation):
        y = tf.nn.relu(tf.matmul(observation, self.w1) + self.b1)
        y = tf.nn.relu(tf.matmul(y, self.w2) + self.b2)
        y = tf.nn.relu(tf.matmul(y, self.w3) + self.b3)  # 不用softmax 不是估计概率
        return y


class TargetQNet:  # 真实值
    def __init__(self):
        self.w1 = tf.Variable(tf.truncated_normal([4, 30], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([30]))

        self.w2 = tf.Variable(tf.truncated_normal([30, 30], stddev=0.1))
        self.b2 = tf.Variable(tf.zeros([30]))

        self.w3 = tf.Variable(tf.truncated_normal([30, 2], stddev=0.1))
        self.b3 = tf.Variable(tf.zeros([2]))

    def forward(self, observation):
        y = tf.nn.relu(tf.matmul(observation, self.w1) + self.b1)
        y = tf.nn.relu(tf.matmul(y, self.w2) + self.b2)
        y = tf.nn.relu(tf.matmul(y, self.w3) + self.b3)
        return y


class Net:
    def __init__(self):
        self.observation = tf.placeholder(dtype=tf.float32, shape=[None, 4])  # St当前状态
        self.action = tf.placeholder(dtype=tf.int32, shape=[None, 1])  # at当前动作
        self.reward = tf.placeholder(dtype=tf.float32, shape=[None, 1])  # rt 回报
        self.next_observation = tf.placeholder(dtype=tf.float32, shape=[None, 4])  # St+1 下一次状态
        self.done = tf.placeholder(dtype=tf.bool, shape=[None])  # 终止

        self.qNet = QNet()
        self.targetQNet = TargetQNet()

    def forward(self, discount):  # discount折扣率
        # 根据当前状态得到的Q值（估计的）
        self.pre_qs = self.qNet.forward(self.observation)
        # 选择当前动作对应的Q值, tf.squeeze降维，tf.expand_dims扩维，axis=1，tf.expand_dims求和，维度为1，tf.multiply相乘，tf.one_hot转换成one_hot形式，深度为2 只要前面两个数值
        self.pre_q = tf.expand_dims(
            tf.reduce_sum(tf.multiply(tf.squeeze(tf.one_hot(self.action, 2)), self.pre_qs), axis=1), axis=1)

        # 根据下一个状态得到Q（t+1）
        self.next_qs = self.targetQNet.forward(self.next_observation)
        # 选择最大的Q值
        self.next_q = tf.expand_dims(tf.reduce_max(self.next_qs, axis=1), axis=1)

        # 得到目标Q值，如果是最后一步，只用奖励，否则Q（t）= r（t）+ dis*maxQ（t+1），dis为打折率
        self.target_q = tf.where(self.done, self.reward, self.reward + discount * self.next_q)

    def play(self):  # 运行游戏
        self.qs = self.qNet.forward(self.observation)  # 传入当前状态
        # 最大的那个Q值的索引就是最大Q值对应的动作
        return tf.argmax(self.qs, axis=1)

    def backward(self):
        self.loss = tf.reduce_mean(tf.square(self.target_q - self.pre_q))
        self.optimizer = tf.train.RMSPropOptimizer(0.01).minimize(self.loss)

    # 把QNet网络训练好的权重W转移到TargetQNet网络
    def copy_params(self):
        return [tf.assign(self.targetQNet.w1, self.qNet.w1), tf.assign(self.targetQNet.w2, self.qNet.w2),
                tf.assign(self.targetQNet.w3, self.qNet.w3), tf.assign(self.targetQNet.b1, self.qNet.b1),
                tf.assign(self.targetQNet.b2, self.qNet.b2), tf.assign(self.targetQNet.b3, self.qNet.b3)]


class Game:
    def __init__(self):
        self.env = gym.make("CartPole-v0")  # 创建游戏

        # 用于训练的经验池
        self.experience_pool = []

        self.observation = self.env.reset()  # 重置游戏

        # 创建经验
        for i in range(10000):
            action = self.env.action_space.sample()  # 随机采样
            next_observation, reward, done, info = self.env.step(action)  # done终止态，next_observation下一步状态St+1
            # St，rt，at，St+1，done放入经验池
            self.experience_pool.append([self.observation, reward, action, next_observation, done])
            if done:
                self.observation = self.env.reset()  # 游戏结束，重置游戏
            else:
                self.observation = next_observation  # 游戏没有结束，把St+1做为下一个St

    def get_experiences(self, batch_size):  # 取经验 一批一批的
        experiences = []
        idxs = []
        for _ in range(batch_size):
            idx = np.random.randint(0, len(self.experience_pool))  # 在经验池里随机取经验，打破相关性
            idxs.append(idx)  # 经验序号
            experiences.append(self.experience_pool[idx])  # 经验
        # idxs是取出经验的序号列表，为了用新的经验替换调老的已经训练过得经验
        return idxs, experiences

    def reset(self):  # 重置游戏
        return self.env.reset()

    def render(self):  # 显示游戏
        return self.env.render()


if __name__ == '__main__':
    game = Game()

    net = Net()
    net.forward(0.9)  # 打折率0.9
    net.backward()
    copy_op = net.copy_params()  # 建立参数图
    run_action_op = net.play()  # 运行游戏图

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        batch_size = 200

        explore = 0.1  # 探索值
        for k in range(10000000):
            idxs, experiences = game.get_experiences(batch_size)

            # 整理数据，一个矩阵一个矩阵
            observations = []
            rewards = []
            actions = []
            next_observations = []
            dones = []

            for experience in experiences:
                observations.append(experience[0])
                rewards.append([experience[1]])
                actions.append([experience[2]])
                next_observations.append(experience[3])
                dones.append(experience[4])

            if k % 10 == 0:
                print("------------ copy param -----------")
                sess.run(copy_op)
                # time.sleep(2)

            _loss, _ = sess.run([net.loss, net.optimizer],
                                feed_dict={net.observation: observations, net.action: actions, net.reward: rewards,
                                           net.next_observation: next_observations, net.done: dones})#优化器传入 St, at ，rt ，St+1 ，done

            explore -= 0.0001
            if explore < 0.0001:
                explore = 0.0001
            print("============", _loss, "===========", explore)

            count = 0  #训练次数计数
            run_observation = game.reset()
            for idx in idxs:  #每次采集新的经验，采了多少经验出来就还多少回去
                if k > 500:
                    game.render()  #查看图像
                # 如果随机值小于探索值，就随机选一个动作作为探索
                if np.random.rand() < explore:
                    run_action = np.random.randint(0, 2)
                # 否则就选Q值最大的那个动作
                else:
                    run_action = sess.run(run_action_op, feed_dict={net.observation: [run_observation]})[0]

                # 采集新的经验
                run_next_observation, run_reward, run_done, run_info = game.env.step(run_action)

                game.experience_pool[idx] = [run_observation, run_reward, run_action, run_next_observation, run_done]
                if run_done:
                    run_observation = game.reset()
                    count += 1  #终止计数
                else:
                    run_observation = run_next_observation
            print("done ..............", count)
