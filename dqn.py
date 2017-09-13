import numpy as np
import sys
import pickle
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from collections import deque
import random
from simulate import PendulumDynamics


class KMostRecent:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque()
    
    def add(self, thing):
        self.buffer.appendleft(thing)
        if len(self.buffer) > self.max_size:
            self.buffer.pop()

    def random_sample(self, size):
        if size < len(self.buffer):
            return random.sample(self.buffer, size)
        else:
            return list(self.buffer)

    def is_full(self):
        return len(self.buffer) == self.max_size


class StatusProcessor:

    def __init__(self, sim_field_length, sim_max_speed, sim_theta_max, sim_thetadot_max):
        self.x_min = -sim_field_length / 2
        self.x_max = sim_field_length / 2
        self.xdot_min = -sim_max_speed
        self.xdot_max = sim_max_speed
        self.theta_min = -sim_theta_max
        self.theta_max = sim_theta_max
        self.thetadot_min = -sim_thetadot_max
        self.thetadot_max = sim_thetadot_max

    @staticmethod
    def _normalize(y, ymin, ymax):
        return 2 * (y - ymin) / (ymax - ymin) - 1

    @staticmethod
    def _denormalize(y, ymin, ymax):
        return ymin + (ymax - ymin) * (y + 1) / 2

    def _unpack_and_process(self, status, mapper):
        x, xdot, theta, thetadot = status
        return (
            mapper(x, self.x_min, self.x_max),
            mapper(xdot, self.xdot_min, self.xdot_max),
            mapper(theta, self.theta_min, self.theta_max),
            mapper(thetadot, self.thetadot_min, self.thetadot_max),
        )

    def normalize(self, status):
        return self._unpack_and_process(status, self._normalize)        
    
    def denormalize(self, status):
        return self._unpack_and_process(status, self._denormalize)        

    def is_valid(self, status):
        x, xdot, theta, thetadot = status
        return (
            self.x_min < x < self.x_max
        ) and (
            self.xdot_min < xdot < self.xdot_max
        ) and (
            self.theta_min < theta < self.theta_max
        ) and (
            self.thetadot_min < thetadot < self.thetadot_max
        )
    
    def is_valid_norm(self, status):
        return all(-1 <= x <= 1 for x in status)


def remove_nan(x, o):
    return o if np.isnan(x) else x


class QNetwork:

    nnet_action_size = 3

    def __init__(self, state_size, gamma, tau, regularization_coeff,
                 learning_rate, hidden_state_size, hidden_size):
        self.state_size = state_size
        self.gamma = gamma
        self.tau = tau
        self.regularization_coeff = regularization_coeff
        self.learning_rate = learning_rate
        self.hidden_state_size = hidden_state_size
        self.hidden_size = hidden_size

    def get_action(self, session, state):
        q_vals = session.run(self.output, feed_dict={
            self.nnet_input_state: np.array([state] * 3),
            self.nnet_input_action: np.array([
                [1, 0, 0], [0, 1, 0], [0, 0, 1]
            ], dtype=np.float32)
        })

        if q_vals[0][0] >= q_vals[1][0] and q_vals[0][0] >= q_vals[2][0]:
            action = -1
            q = q_vals[0][0]
        elif q_vals[1][0] >= q_vals[0][0] and q_vals[1][0] >= q_vals[2][0]:
            action = 0
            q = q_vals[1][0]
        else:  # if q_vals[2][0] >= q_vals[1][0] and q_vals[2][0] >= q_vals[0][0]:
            action = 1
            q = q_vals[2][0]

        return action, q

    def learn_from_replay(self, session, batch_replay):
        # get q value for each action in the batch
        next_moves_q_state, next_moves_q_action = [], []
        for state, action, reward, next_state, end in batch_replay:
            next_moves_q_state.extend([next_state] * 3)
            next_moves_q_action.extend([
                [1, 0, 0], [0, 1, 0], [0, 0, 1]
            ])

        q_vals = session.run(self.target_output, feed_dict={
            self.nnet_input_state: np.array(next_moves_q_state),
            self.nnet_input_action: np.array(next_moves_q_action, dtype=np.float32)
        })

        # compute expected reward for the states in the batch
        batch_inputs_state, batch_inputs_action, batch_outputs = [], [], []
        for i, (state, action, reward, next_state, end) in enumerate(batch_replay):
            batch_inputs_state.append(state)
            batch_inputs_action.append([(0, 1, 0), (0, 0, 1), (1, 0, 0)][action])
            consider_future = 0 if end else 1
            batch_outputs.append([reward + consider_future * self.gamma * max(
                q_vals[3 * i][0], q_vals[3 * i + 1][0], q_vals[3 * i + 2][0]
            )])

        # backpropagate
        _, loss_value = session.run([self.optimizer, self.loss], feed_dict={
            self.nnet_input_state: batch_inputs_state,
            self.nnet_input_action: batch_inputs_action,
            self.nnet_label: batch_outputs,
        })

        session.run(self.update_target_network_op)

        assert not np.isnan(loss_value)
        return loss_value


    def build(self):
        self._build_network()
        self._build_target_network()

    def _build_network(self):
        self.nnet_input_state = tf.placeholder(
            shape=[None, self.state_size], dtype=tf.float32, name='nnet_input_state'
        )
        self.nnet_input_action = tf.placeholder(
            shape=[None, self.nnet_action_size], dtype=tf.float32, name='nnet_input_action'
        )

        self.nnet_label = tf.placeholder(
            shape=[None, 1], dtype=tf.float32, name='nnet_label'
        )

        self.weights_1, self.bias_1 = self._make_weights(self.state_size, self.hidden_state_size)
        self.hidden_state_1 = self._compute_next_layer(self.nnet_input_state, self.weights_1, self.bias_1)

        self.weights_2, self.bias_2 = self._make_weights(self.hidden_state_size, self.hidden_size)
        self.hidden_state_2 = self._compute_next_layer(self.hidden_state_1, self.weights_2, self.bias_2)

        self.weights_3, self.bias_3 = self._make_weights(self.nnet_action_size, self.hidden_size)
        self.hidden_action = self._compute_next_layer(self.nnet_input_action, self.weights_3, self.bias_3)

        self.bias_5 = tf.Variable(tf.constant(0.1, shape=[self.hidden_size]))
        self.hidden_combined = tf.nn.relu(self.hidden_state_2 + self.hidden_action + self.bias_5)

        self.weights_4, self.bias_4 = self._make_weights(self.hidden_size, 1)
        self.output = self._compute_next_layer(self.hidden_combined, self.weights_4, self.bias_4, activation=None)

        self.squared_error = (self.nnet_label - self.output)**2
        self.loss = tf.reduce_mean(self.squared_error) + self.regularization_coeff * (
            tf.reduce_sum(self.weights_1 ** 2) + tf.reduce_sum(self.bias_1 ** 2) +
            tf.reduce_sum(self.weights_2 ** 2) + tf.reduce_sum(self.bias_2 ** 2) + 
            tf.reduce_sum(self.weights_3 ** 2) + tf.reduce_sum(self.bias_3 ** 2) + 
            tf.reduce_sum(self.weights_4 ** 2) + tf.reduce_sum(self.bias_4 ** 2) + 
            tf.reduce_sum(self.bias_5 ** 2)
        )

        self.network_params = [self.weights_1, self.bias_1, self.weights_2,
                               self.bias_2, self.weights_3, self.bias_3,
                               self.weights_4, self.bias_4, self.bias_5]

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def _build_target_network(self):
        self.target_network_params = [tf.Variable(var.initialized_value())
                                      for var in self.network_params]
        self.update_target_network_op = [
            target_var.assign(self.tau * var + (1 - self.tau) * target_var)
            for var, target_var in zip(self.network_params, self.target_network_params)
        ]

        (
            self.target_weights_1, self.target_bias_1, self.target_weights_2,
            self.target_bias_2, self.target_weights_3, self.target_bias_3,
            self.target_weights_4, self.target_bias_4, self.target_bias_5
        ) = self.target_network_params

        self.target_hidden_state_1 = self._compute_next_layer(self.nnet_input_state, self.target_weights_1, self.target_bias_1)
        self.target_hidden_state_2 = self._compute_next_layer(self.target_hidden_state_1, self.target_weights_2, self.target_bias_2)
        self.target_hidden_action = self._compute_next_layer(self.nnet_input_action, self.target_weights_3, self.target_bias_3)
        self.target_hidden_combined = tf.nn.relu(self.target_hidden_state_2 + self.target_hidden_action + self.target_bias_5)
        self.target_output = self._compute_next_layer(self.target_hidden_combined, self.target_weights_4, self.target_bias_4, activation=None)

    @staticmethod
    def _make_weights(rows, cols):
        weights = tf.Variable(tf.truncated_normal(shape=[rows, cols], stddev=0.1))
        bias = tf.Variable(tf.constant(0.1, shape=[cols]))
        return weights, bias

    @staticmethod
    def _compute_next_layer(input_layer, weights, bias, activation=tf.nn.relu):
        h = tf.matmul(input_layer, weights) + bias
        return activation(h) if activation else h


class DQNPendulum:

    replay_batch_size = 32          # size of the minibatch for experience replay
    simulation_length = 500         # how many steps each episode is
    num_episodes = 100000           # stop training after this many episodes
    replay_buffer_size = 1000000    # how many experiences to keep in the replay buffer
    force_factor = 75               # force intensity for bang-bang control
    epsilon_decay = 0.00025         # exploration rate coefficient
    min_eps = 0.1                   # minimum random exploration rate
    save_network_every = 25         # checkpoint interval (episodes)
    past_states_count = 4           # input this many last states to the Q network

    gamma = 0.99                    # Q-learning discount factor
    tau = 0.001                     # soft update strength for target network
    regularization_coeff = 0.001    # regularization in the loss
    learning_rate = 0.001           # learning rate for the Q network
    nnet_hidden_state_size = 128    # state is processed alone in this hidden layer
    nnet_hidden_size = 128          # this layer combines state and action

    sim_field_length = 50           # episode fails if pendulum is outside [-l/2, l/2]
    sim_max_speed = 100             # episode fails if the cart moves faster than this
    sim_theta_max = 10              # episode fails if pendulum angle is more than this
    sim_thetadot_max = 100          # episode fails if pendulum rotates faster than this

    def learn(self):
        graph = tf.Graph()
        self.qnet = QNetwork(4 * self.past_states_count, self.gamma, self.tau, self.regularization_coeff,
                             self.learning_rate, self.nnet_hidden_state_size, self.nnet_hidden_size)
        with graph.as_default():
            self.qnet.build()

        for f in os.listdir('logs'):
            os.remove('logs/' + f)

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()

            writer = tf.summary.FileWriter('logs', session.graph)
            saver = tf.train.Saver(self.qnet.target_network_params)
            replay_buffer = KMostRecent(self.replay_buffer_size)
            state_processor = StatusProcessor(self.sim_field_length, self.sim_max_speed,
                                              self.sim_theta_max, self.sim_thetadot_max)

            for episode in range(self.num_episodes):
                stats = self.do_episode(session, episode, replay_buffer, state_processor)
                self.write_summary(session, episode, writer, saver, replay_buffer, *stats)

            writer.close()

    def do_episode(self, session, episode_no, replay_buffer, state_processor):
        pendulum = PendulumDynamics(0, 0, np.pi, 0)
        last_states = KMostRecent(self.past_states_count)
        for _ in range(self.past_states_count):
            last_states.add(state_processor.normalize(pendulum.state))

        end_early, sim_trace, all_losses, all_rewards, all_qs = False, [], [], [], []
        for step in range(self.simulation_length):
            state = [x for s in last_states.buffer for x in s]

            # choose next action
            if self.epsilon_decay > 0 and np.random.random() < max(self.min_eps, np.exp(-episode_no * self.epsilon_decay)):
                action = random.choice([-1, 0, 1])
            else:
                action, qval = self.qnet.get_action(session, state)
                all_qs.append(qval)

            # perform action
            sim_trace.append((step * pendulum.dt, state_processor.denormalize(state[-4:]), action))
            old_state = state
            pendulum.step_simulate(self.force_factor * action)
            last_states.add(state_processor.normalize(pendulum.state))
            state = [x for s in last_states.buffer for x in s]

            # compute reward    
            if state_processor.is_valid(pendulum.state):
                reward = -0.05 * (abs(pendulum.theta) - 3) - 0.001 * (abs(pendulum.x) - 3)
                #reward = 0.1 if abs(pendulum.theta) < 0.25 and abs(pendulum.x < 0.25) else -0.001
            else:
                reward = -10
                end_early = True

            replay_buffer.add((old_state, action, reward, state, end_early))
            all_rewards.append(reward)

            # learn
            if not end_early:
                batch_replay = replay_buffer.random_sample(self.replay_batch_size)
                loss_value = self.qnet.learn_from_replay(session, batch_replay)
                all_losses.append(loss_value)

        return sim_trace, all_losses, all_rewards, all_qs

    def write_summary(self, session, episode_no, writer, saver, replay_buffer,
                      sim_trace, all_losses, all_rewards, all_qs):
        summary = tf.Summary()
        summary.value.add(tag='avg_loss', simple_value=np.mean(all_losses))
        summary.value.add(tag='avg_reward', simple_value=np.mean(all_rewards))
        summary.value.add(tag='sum_reward', simple_value=sum(all_rewards))
        summary.value.add(tag='avg_q', simple_value=remove_nan(np.mean(all_qs), 0))
        summary.value.add(tag='sum_q', simple_value=np.sum(all_qs))
        writer.add_summary(summary, global_step=episode_no)
        print('Episode %d - L: %.3f\tAR: %.3f\tSR: %.3f\tAQ: %.3f\tSQ: %.3f' % (
            episode_no, np.mean(all_losses), np.mean(all_rewards),
            sum(all_rewards), np.mean(all_qs), np.sum(all_qs)
        ))

        if episode_no and not episode_no % self.save_network_every:
            saver.save(session, './logs/updates', global_step=episode_no)
            with open('./logs/last-episode.csv', 'w') as f:
                f.write('t,force,x,xdot,theta,thetadot\n')
                for time, state, action in sim_trace:
                    f.write('%f,%f,%f,%f,%f,%d\n' % (
                        (time, action * self.force_factor) + state
                    ))
            with open('./logs/last-replay-buffer.pckl', 'wb') as f:
                pickle.dump(replay_buffer, f)
            print('saved')


def parse_args(gp):
    for arg in sys.argv[1:]:
        name, val = arg.split('=', 1)
        if name.startswith('_'):
            continue
        orig = getattr(gp, name)
        if type(orig) == list:  # lists must have homogenuous type
            value = [type(orig[0])(e) for e in val.split(',')]
        elif type(orig) == dict:
            value = {}
            for kvp in val.split(';'):
                k, v = kvp.split(':', 1)
                orig[k] = type(orig[k])(v)
            value = orig
        else:
            value = type(orig)(val)
        setattr(gp, name, value)
    return gp


if __name__ == '__main__':
    qp = parse_args(DQNPendulum())
    qp.learn()
