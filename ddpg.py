import pickle
import numpy as np
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from collections import deque
import random
from simulate import PendulumDynamics


class ReplayBuffer:
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


class OrnsteinUhlenbeckProcess:
    # https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    # implemented as detailed in https://math.stackexchange.com/a/1288406/99169
    def __init__(self, theta, mu, sigma, x0, dt):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.x0 = x0
        self.dt = dt
        self.n = 0
        self.last = self.x0

    def get_noise(self):
        self.last += (
            self.theta * (self.mu - self.last) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal()
        )
        return self.last


def make_weights(rows, cols):
    weights = tf.Variable(tf.truncated_normal(shape=[rows, cols], stddev=0.1))
    bias = tf.Variable(tf.constant(0.1, shape=[cols]))
    variable_summary(weights)
    variable_summary(bias)
    return weights, bias


def compute_next_layer(input_layer, weights, bias, activation=tf.nn.relu):
    h = tf.matmul(input_layer, weights) + bias
    output = activation(h) if activation else h
    variable_summary(output)
    return output


def variable_summary(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.histogram('histogram', var)


class CriticNetwork:
    nnet_state_size = 4
    nnet_action_size = 1
    nnet_hidden_state_size = 64
    nnet_hidden_size = 64
    tau = 0.001
    learning_rate = 0.001

    def get_q_values(self, session, states, actions):
        return session.run(self.output, feed_dict={
            self.nnet_input_state: np.array(states, dtype=np.float32),
            self.nnet_input_action: np.array(actions, dtype=np.float32)
        })

    def get_q_values_from_target(self, session, states, actions):
        return session.run(self.target_output, feed_dict={
            self.nnet_input_state: np.array(states, dtype=np.float32),
            self.nnet_input_action: np.array(actions, dtype=np.float32)
        })
    
    def update_weights(self, session, states, actions, outputs):
        _, loss = session.run([self.optimizer, self.loss], feed_dict={
            self.nnet_input_state: np.array(states, dtype=np.float32),
            self.nnet_input_action: np.array(actions, dtype=np.float32),
            self.nnet_label: np.array(outputs, dtype=np.float32),
        })
        return loss

    def update_target_network(self, session):
        session.run(self.update_target_network_node)

    def get_action_gradients(self, session, states, actions):
        aa = [
            session.run(self.action_gradients, feed_dict={
                self.nnet_input_state: np.array([state], dtype=np.float32),
                self.nnet_input_action: np.array([action], dtype=np.float32),
            }) for state, action in zip(states, actions)
        ]
        if sum(abs(x[0][0][0]) for x in aa) < 0.001:
            #import pdb; pdb.set_trace()
            pass
        return aa

    def build(self):
        self._build_network()
        self._build_target_network()

    def _build_network(self):
        self.nnet_input_state = tf.placeholder(
            shape=[None, self.nnet_state_size], dtype=tf.float32
        )
        self.nnet_input_action = tf.placeholder(
            shape=[None, self.nnet_action_size], dtype=tf.float32
        )

        self.nnet_label = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        with tf.name_scope('critic-state-hidden-1'):
            self.weights_1, self.bias_1 = make_weights(self.nnet_state_size, self.nnet_hidden_state_size)
            self.hidden_state_1 = compute_next_layer(self.nnet_input_state, self.weights_1, self.bias_1)

        with tf.name_scope('critic-state-hidden-2'):
            self.weights_2, self.bias_2 = make_weights(self.nnet_hidden_state_size, self.nnet_hidden_size)
            self.hidden_state_2 = compute_next_layer(self.hidden_state_1, self.weights_2, self.bias_2)

        with tf.name_scope('critic-action-hidden'):
            self.weights_3, self.bias_3 = make_weights(self.nnet_action_size, self.nnet_hidden_size)
            self.hidden_action = compute_next_layer(self.nnet_input_action, self.weights_3, self.bias_3)

        with tf.name_scope('critic-combined-hidden'):
            self.bias_5 = tf.Variable(tf.constant(0.1, shape=[self.nnet_hidden_size]))
            self.hidden_combined = tf.nn.relu(self.hidden_state_2 + self.hidden_action + self.bias_5)

        with tf.name_scope('critic-last'):
            self.weights_4, self.bias_4 = make_weights(self.nnet_hidden_size, 1)
            self.output = compute_next_layer(self.hidden_combined, self.weights_4, self.bias_4, activation=None)

        self.squared_error = (self.nnet_label - self.output)**2
        self.loss = tf.reduce_mean(self.squared_error)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.network_params = [self.weights_1, self.bias_1, self.weights_2,
                               self.bias_2, self.weights_3, self.bias_3,
                               self.weights_4, self.bias_4, self.bias_5]
        self.action_gradients = tf.gradients(self.output, self.nnet_input_action)

    def _build_target_network(self):
        self.target_network_params = [tf.Variable(var.initialized_value())
                                      for var in self.network_params]

        self.update_target_network_node = [
            target_var.assign(self.tau * var + (1 - self.tau) * target_var)
            for var, target_var in zip(self.network_params, self.target_network_params)
        ]

        (
            self.target_weights_1, self.target_bias_1, self.target_weights_2,
            self.target_bias_2, self.target_weights_3, self.target_bias_3,
            self.target_weights_4, self.target_bias_4, self.target_bias_5
        ) = self.target_network_params

        self.target_hidden_state_1 = compute_next_layer(
            self.nnet_input_state, self.target_weights_1, self.target_bias_1
        )
        self.target_hidden_state_2 = compute_next_layer(
            self.target_hidden_state_1, self.target_weights_2, self.target_bias_2
        )
        self.target_hidden_action = compute_next_layer(
            self.nnet_input_action, self.target_weights_3, self.target_bias_3
        )
        self.target_hidden_combined = tf.nn.relu(
            self.target_hidden_state_2 + self.target_hidden_action + self.target_bias_5
        )
        self.target_output = compute_next_layer(
            self.target_hidden_combined, self.target_weights_4, self.target_bias_4, activation=None
        )


class ActorNetwork:
    nnet_state_size = 4
    nnet_action_size = 1
    nnet_hidden1_size = 64
    nnet_hidden2_size = 32
    tau = 0.001
    learning_rate = 0.001

    def get_actions(self, session, states):
        return session.run(self.output, feed_dict={
            self.nnet_input_state: np.array(states, dtype=np.float32),
        })

    def get_actions_from_target(self, session, states):
        return session.run(self.target_output, feed_dict={
            self.nnet_input_state: np.array(states, dtype=np.float32),
        })

    def get_param_gradients(self, session, states):
        return [
            session.run(self.param_gradients, feed_dict={
                self.nnet_input_state: np.array([s], dtype=np.float32),
            }) for s in states
        ]

    def update_weights(self, session, replay_states, critic_action_gradients):
        # shape of actor_gradients is len(replay_states) x 6
        # each row of actor_gradients is multiplied by the corresponding critic gradient
        # then take a column-wise average
        critic_gradients = np.array(critic_action_gradients).reshape((len(replay_states), 1))
        actor_gradients = np.array(self.get_param_gradients(session, replay_states))
        avg_gradients = (actor_gradients * critic_gradients).mean(axis=0)

        new_params = session.run(self.update_weights_op, feed_dict={
            self.weights_1_gradient: avg_gradients[0],
            self.bias_1_gradient: avg_gradients[1].reshape((self.nnet_hidden1_size,)),
            self.weights_2_gradient: avg_gradients[2],
            self.bias_2_gradient: avg_gradients[3].reshape((self.nnet_hidden2_size,)),
            self.weights_3_gradient: avg_gradients[4],
            self.bias_3_gradient: avg_gradients[5].reshape((self.nnet_action_size,))
        })
        
        return sum(np.sum(x) for g in avg_gradients for x in g)
        #return np.sum([np.sum(x) for x in new_params])

    def update_target_network(self, session):
        session.run(self.update_target_network_node)

    def build(self):
        self._build_network()
        self._build_target_network()

    def _build_network(self):
        self.nnet_input_state = tf.placeholder(shape=[None, self.nnet_state_size], dtype=tf.float32)
        self.nnet_label = tf.placeholder(shape=[None, self.nnet_action_size], dtype=tf.float32)

        with tf.name_scope('actor-hidden-1'):
            self.weights_1, self.bias_1 = make_weights(self.nnet_state_size, self.nnet_hidden1_size)
            self.hidden_1 = compute_next_layer(self.nnet_input_state, self.weights_1, self.bias_1)
        
        with tf.name_scope('actor-hidden-2'):
            self.weights_2, self.bias_2 = make_weights(self.nnet_hidden1_size, self.nnet_hidden2_size)
            self.hidden_2 = compute_next_layer(self.hidden_1, self.weights_2, self.bias_2)

        with tf.name_scope('actor-output'):
            self.weights_3, self.bias_3 = make_weights(self.nnet_hidden2_size, self.nnet_action_size)
            self.output = compute_next_layer(self.hidden_2, self.weights_3, self.bias_3,
                                             activation=tf.nn.tanh)

        # update operations for gradient descent
        self.network_params = [self.weights_1, self.bias_1, 
                               self.weights_2, self.bias_2,
                               self.weights_3, self.bias_3]
        self.param_gradients = tf.gradients(self.output, self.network_params)

        with tf.name_scope('actor-gradients'):
            names = ['weights-1', 'bias-1', 'weights-2', 'bias-2', 'weights-3', 'bias-3']
            for name, grad in zip(names, self.param_gradients):
                with tf.name_scope(name):
                    variable_summary(grad)
    
        self.weights_1_gradient = tf.placeholder(shape=self.weights_1.shape, dtype=tf.float32)
        self.update_weights_1 = self.weights_1.assign_add(self.learning_rate * self.weights_1_gradient)

        self.bias_1_gradient = tf.placeholder(shape=self.bias_1.shape, dtype=tf.float32)
        self.update_bias_1 = self.bias_1.assign_add(self.learning_rate * self.bias_1_gradient)

        self.weights_2_gradient = tf.placeholder(shape=self.weights_2.shape, dtype=tf.float32)
        self.update_weights_2 = self.weights_2.assign_add(self.learning_rate * self.weights_2_gradient)

        self.bias_2_gradient = tf.placeholder(shape=self.bias_2.shape, dtype=tf.float32)
        self.update_bias_2 = self.bias_2.assign_add(self.learning_rate * self.bias_2_gradient)

        self.weights_3_gradient = tf.placeholder(shape=self.weights_3.shape, dtype=tf.float32)
        self.update_weights_3 = self.weights_3.assign_add(self.learning_rate * self.weights_3_gradient)

        self.bias_3_gradient = tf.placeholder(shape=self.bias_3.shape, dtype=tf.float32)
        self.update_bias_3 = self.bias_3.assign_add(self.learning_rate * self.bias_3_gradient)

        self.update_weights_op = [
            self.update_weights_1, self.update_bias_1,
            self.update_weights_1, self.update_bias_2,
            self.update_weights_3, self.update_bias_3,
        ]

    def _build_target_network(self):
        self.target_network_params = [tf.Variable(var.initialized_value())
                                      for var in self.network_params]
        self.update_target_network_node = [
            target_var.assign(self.tau * var + (1 - self.tau) * target_var)
            for var, target_var in zip(self.network_params, self.target_network_params)
        ]

        (
            self.target_weights_1, self.target_bias_1,
            self.target_weights_2, self.target_bias_2,
            self.target_weights_3, self.target_bias_3,
        ) = self.target_network_params

        self.target_hidden_1 = compute_next_layer(
            self.nnet_input_state, self.target_weights_1, self.target_bias_1
        )

        self.target_hidden_2 = compute_next_layer(
            self.target_hidden_1, self.target_weights_2, self.target_bias_2
        )

        self.target_output = compute_next_layer(
            self.target_hidden_2, self.target_weights_3, self.target_bias_3,
            activation=tf.nn.tanh
        )


class StatusProcessor:
    x_min = -25
    x_max = 25
    xdot_min = -100
    xdot_max = 100
    theta_min = -10
    theta_max = 10
    thetadot_min = -100
    thetadot_max = 100

    @staticmethod
    def _normalize(y, ymin, ymax):
        return 2 * (y - ymin) / (ymax - ymin) - 1

    def normalize(self, status):
        x, xdot, theta, thetadot = status
        return (
            self._normalize(x, self.x_min, self.x_max),
            self._normalize(xdot, self.xdot_min, self.xdot_max),
            self._normalize(theta, self.theta_min, self.theta_max),
            self._normalize(thetadot, self.thetadot_min, self.thetadot_max),
        )

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


def main():

    # learning hyper parameters
    replay_batch_size = 32
    simulation_length = 500
    num_episodes = 100000
    replay_buffer_size = 10000000
    force_factor = 50
    gamma = 0.99
    save_network_every = 1

    noise_theta = 0.15
    noise_mu = 0
    noise_sigma_decay= 0.005
    noise_x0 = 0

    graph = tf.Graph()
    with graph.as_default():
        actor = ActorNetwork()
        actor.build()

        critic = CriticNetwork()
        critic.build()

    for f in os.listdir('logs'):
        os.remove('logs/' + f)

    with tf.Session(graph=graph) as session:
        all_summaries = tf.summary.merge_all()
        tf.global_variables_initializer().run()

        writer = tf.summary.FileWriter('logs', session.graph)
        saver = tf.train.Saver(actor.target_network_params + critic.target_network_params)
        replay_buffer = ReplayBuffer(replay_buffer_size)
        state_processor = StatusProcessor()

        for episode in range(num_episodes):
            initial_theta = np.pi * (2 * np.random.random() - 1)
            pendulum = PendulumDynamics(0, 0, np.pi, 0)
            noise_sigma = np.exp(-episode * noise_sigma_decay)
            noise_process = OrnsteinUhlenbeckProcess(noise_theta, noise_mu, noise_sigma,
                                                     noise_x0, pendulum.dt)
            last_episode, episode_critic_losses, episode_actor_gradients, episode_rewards = [], [], [], []
            end_early = False

            for step in range(simulation_length):
                # save current state
                state = state_processor.normalize(pendulum.state)

                # choose next action
                action = actor.get_actions(session, [state])[0][0] + noise_process.get_noise()
                force = force_factor * min(max(action, -1), 1)

                # perform action and compute reward
                last_episode.append((step, state, force))
                old_state = state
                pendulum.step_simulate(force)
                state = state_processor.normalize(pendulum.state)

                if state_processor.is_valid_norm(state):
                    #reward = -0.05 * (abs(pendulum.theta) - 2) - 0.001 * (abs(pendulum.x) - 2)
                    #reward = 0.1 if abs(pendulum.theta) < 0.25 and abs(pendulum.x < 0.25) else -0.001

                    if abs(pendulum.theta) < 0.25 and abs(pendulum.x) < 0.25:
                        reward = 0.05 * (0.25 - abs(pendulum.theta)) + 0.001 * (0.25 - abs(pendulum.x))
                    else:
                        #reward = -0.001
                        reward = 0
                else:
                    reward = -25
                    end_early = True

                replay_buffer.add((old_state, [action], reward, state))
                episode_rewards.append(reward)

                if end_early:
                    break
                
                # experience replay
                batch_replay = replay_buffer.random_sample(replay_batch_size)
                (
                    replay_states, replay_actions,
                    replay_rewards, replay_next_states
                ) = zip(*batch_replay)

                # update critic
                replay_next_actions = actor.get_actions_from_target(session, replay_next_states)
                replay_q_vals = critic.get_q_values_from_target(session, replay_next_states,
                                                                replay_next_actions)
                replay_outputs = [r + gamma * q if state_processor.is_valid_norm(n) else [r]
                                  for r, q, n in zip(replay_rewards, replay_q_vals, replay_next_states)]

                critic_loss = critic.update_weights(session, replay_states, replay_actions,
                                                    replay_outputs)
                assert not np.isnan(critic_loss)
                critic.update_target_network(session)

                # update actor
                predicted_actions = actor.get_actions(session, replay_states)
                critic_action_gradients = critic.get_action_gradients(
                    session, replay_states, predicted_actions
                )
                gradient_magnitude = actor.update_weights(session, replay_states, critic_action_gradients)
                actor.update_target_network(session)

                # summary
                print(episode, step, reward, critic_loss, gradient_magnitude)
                episode_actor_gradients.append(gradient_magnitude)
                episode_critic_losses.append(critic_loss)
                episode_rewards.append(reward)

            summary = tf.Summary()
            summary.value.add(tag='critic_loss', simple_value=np.mean(episode_critic_losses))
            summary.value.add(tag='actor_gradients_mag', simple_value=np.mean(episode_actor_gradients))
            summary.value.add(tag='reward', simple_value=sum(episode_rewards))
            writer.add_summary(summary, global_step=episode)

            print('Episode %d - CL: %.3f\tAG: %.3f, \tAR: %.3f\tSR: %.3f' % (
                episode, np.mean(episode_critic_losses), np.mean(episode_actor_gradients),
                np.mean(episode_rewards), sum(episode_rewards))
            )

            if not episode % save_network_every:
                saver.save(session, './logs/updates', global_step=episode)
                with open('./logs/last-episode.csv', 'w') as f:
                    f.write('t,force,x,xdot,theta,thetadot\n')
                    for step, state, force in last_episode:
                        f.write('%f,%f,%f,%f,%f,%f\n' % (
                            (step * pendulum.dt, force) + state
                        ))
                print('saved')

        writer.close()


if __name__ == '__main__':
    main()