import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


# TODO(hang): improve DQN with double, prioritised replay, dueling network
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            flag,
            learning_rate=0.001,
            reward_decay=0.9,
            e_greedy=0.3,
            e_greedy_min=0.01,
            replace_target_iter=500,
            memory_size=10000,
            e_greedy_decrement=1e-5,
            output_graph=False
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.flag = flag
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.epsilon_min = e_greedy_min
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.epsilon_decrement = e_greedy_decrement

        # total learning step
        self.learn_step_counter = 0

        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net' + str(self.flag))
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net' + str(self.flag))

        with tf.variable_scope('hard_replacement' + str(self.flag)):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s' + str(self.flag))  # current state
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features],
                                 name='s_' + str(self.flag))  # next state by do action in current state
        self.r = tf.placeholder(tf.float32, [None, ],
                                name='r' + str(self.flag))  # reward by do action in current  state
        self.a = tf.placeholder(tf.int32, [None, ], name='a' + str(self.flag))  # action in current state

        # initial random weight
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        # for q_eval
        with tf.variable_scope('eval_net' + str(self.flag)):
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        # for q_next
        with tf.variable_scope('target_net' + str(self.flag)):
            t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            # 这时的q_next是个操作
            self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t2')

        # q_target = reward + gamma * max(q_next)
        with tf.variable_scope('q_target' + str(self.flag)):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')  # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)

        # q_eval is a q-table for all actions,we only need the q-value of action belong to current state
        with tf.variable_scope('q_eval' + str(self.flag)):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)  # shape=(None, )

        with tf.variable_scope('loss' + str(self.flag)):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))

        with tf.variable_scope('train' + str(self.flag)):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = np.array(observation)
        observation = observation[np.newaxis, :]

        # forward feed the observation and get q value for every actions
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        # action = np.argmax(actions_value)
        # print(actions_value,np.argmax(actions_value))

        return np.argmax(actions_value[0])

    def learn(self, states, actions, rewards, states_next, done):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            # print('\ntarget_params_replaced\n')
        # print(len(rewards))
        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: states,
                self.a: actions,
                self.r: rewards,
                self.s_: states_next,
            })

        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon - self.epsilon_decrement if self.epsilon > self.epsilon_min else self.epsilon_min
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('loss')
        plt.xlabel('training steps')
        plt.show()
