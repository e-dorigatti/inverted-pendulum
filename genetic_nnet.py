from py_neuralnet import NeuralNetwork, genetic_learn, GeneticAlgorithm
from simulate import PendulumDynamics
from animate import Animate
import numpy as np
import matplotlib.pyplot as plt
import pickle
import itertools
import sys
import csv


class GeneticPendulum(GeneticAlgorithm):
    nnet_size = [5, 5, 1]                 # nnet arch (must start with 5 and end with 1)
    nnet_activation = 'tanh'              # activation function used by nnet
    pop_size = 20                         # n neural networks compete (learning is O(n**2))
    pop_dump = ''                         # initial population dump
    force_factor = 50.                    # multiply nnet output (-1 to 1) by this factor
    time_limit = 2.5                      # duration of the simulation in seconds (O(n**2))
    checkpoint_interval = 20              # checkpoint, dump population and show plots
    checkpoint_animation = True           # show animation at checkpoint (blocking!)
    checkpoint_summary = False            # simple summary plots at checkpoint (blocking!)
    weight_noise_stdev = 0.0002           # apply gaussian noise of given stdev to weights
    weight_noise_worst = 10               # apply weight noise to the worst n nnets
    regularization_coeff = 1.             # prefer nnets with low weights
    nnet_control_interval = 1             # allow the nnet to apply force every n steps
    target = dict(x=0., theta=0.)         # the nnet should keep the pendulum close to this
    target_weights = dict(x=.5, theta=.5) # weights for the multi-objective optimization
    pendulum_init = dict(x=0., xdot=0., theta=np.pi, thetadot=0.)
    pendulum_phys = dict(m=0.1, M=1.0, l=1.0, g=9.81, b=2.0, dt=0.02)

    def __init__(self):
        self.gen_history = []

    def learn(self):
        genetic_learn(self, self.nnet_size, self.pop_size)

    def initial_conditions(self):
        p = PendulumDynamics(**self.pendulum_init)
        for k, v in self.pendulum_phys.iteritems():
            setattr(p, k, v)
        return p

    def simulate(self, nnet, pd):
        t = 0
        data = []

        while t * pd.dt < self.time_limit:
            if t % self.nnet_control_interval == 0:
                force = nnet.value([pd.x,
                                    pd.xdot,
                                    pd.theta,
                                    pd.thetadot,
                                    t * pd.dt / self.time_limit,
                                    ])[0][0]

            data.append({
                't': pd.dt * t,
                'x': pd.x,
                'xdot': pd.xdot,
                'theta': pd.theta,
                'thetadot': pd.thetadot,
                'force': force,
            })
            pd.step_simulate(force * self.force_factor)
            t += 1

        return data

    def checkpoint(self, i, pop):
        init = self.initial_conditions()
        sim = self.simulate(pop[0], init)

        with open('last-run-%d.csv' % i, 'w') as f:
            writer = csv.DictWriter(f, sim[0].keys())
            writer.writeheader()
            writer.writerows(sim)

        with open('last-run-%d.pickle' % i, 'w') as f:
            pickle.dump(pop, f)

        if self.checkpoint_animation:
            Animate(sim, init.l, 2 * init.dt * 1000, tight_layout=True).show()

        if self.checkpoint_summary:
            self.plot(sim)

    def fitness(self, nnet, sim=None):
        sim = sim or self.simulate(nnet, self.initial_conditions())

        fitness = -sum((w**2).sum() for w in nnet.weights) * self.regularization_coeff
        for k in self.target:
            weight = self.target_weights.get(k, 1.)
            fitness -= weight * sum((self.target[k] - p[k])**2 for p in sim)

        return fitness

    def stop(self, i, pop):
        if i % self.checkpoint_interval == 0:
            self.checkpoint(i, pop)

        best_worst = [self.fitness(n) for n in pop[:2] + pop[-2:]]
        self.gen_history.append((i, best_worst[0], best_worst[-1]))
        print i, '    '.join(str(x) for x in best_worst)

        if self.weight_noise_stdev > 0:
            for nnet in pop[self.weight_noise_worst:]:
                for w in nnet.weights:
                    w += np.random.normal(0, self.weight_noise_stdev, w.shape)

        return False

    def get_new_neural_network(self, size):
        return NeuralNetwork(size, activation=self.nnet_activation)

    def initial_population(self, nnet_size, pop_size):
        if self.pop_dump:
            with open(self.pop_dump) as f:
                return pickle.load(f)
        else:
            return [self.get_new_neural_network(nnet_size) for _ in range(pop_size)]

    def plot(self, best):
        time = [x['t'] for x in best]
        epochs = [i for i, b, w in self.gen_history]

        plt.subplot(1, 2, 1); plt.title('Sample run from best network')
        plt.plot(time, [x['theta'] for x in best], 'b-', label='Theta (rad)')
        plt.plot(time, [x['force'] for x in best], 'g-', label='Force (norm.)')
        plt.plot(time, [x['x'] for x in best], 'r-', label='Position (m)')
        plt.grid(); plt.legend(); plt.xlabel('Time (s)')

        plt.subplot(1, 2, 2); plt.title('Score History')
        plt.plot(epochs, [b for i, b, w in self.gen_history], 'b-', label='Best')
        plt.plot(epochs, [w for i, b, w in self.gen_history], 'b--', label='Worst')
        plt.grid(); plt.legend(); plt.xlabel('Training Epoch'); plt.ylabel('Score')
        plt.show()


def parse_args(gp):
    for arg in sys.argv[1:]:
        name, val = arg.split('=', 1)
        orig = getattr(gp, name)
        if type(orig) == list:  # lists must have homogenuous type
            value = [type(orig[0])(e) for e in val.split(',')]
        elif type(orig) == dict:
            value = {}
            for kvp in val.split(';'):
                k, v = kvp.split(':', 1)
                orig[k] = type(orig[k])(v)
        else:
            value = type(orig)(val)
        setattr(gp, name, value)
    return gp


if __name__ == '__main__':
    gp = parse_args(GeneticPendulum())
    gp.learn()
