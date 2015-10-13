from py_neuralnet import NeuralNetwork, genetic_learn
from simulate import PendulumDynamics
from animate import Animate
import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys
import csv


class GeneticPendulum:
    nnet_size = [5, 5, 1]           # nnet arch (must start with 5 and end with 1)
    pop_size = 20                   # n neural networks compete (learning is O(n**2))
    force_factor = 50.              # multiply nnet output (-1 to 1) by this factor
    time_limit = 2.5                # duration of the simulation in seconds (O(n**2))
    plot_interval = 20              # plot every n time steps
    weight_noise_stdev = 0.0002     # apply gaussian noise of given stdev to weights
    weight_noise_worst = 10         # apply weight noise to the worst n nnets
    regularization_coeff = 1.       # prefer nnets with low weights
    nnet_control_interval = 1       # allow the nnet to apply force every n steps
    target = dict(x=0., theta=0.)   # the nnet should keep the pendulum close to this
    pendulum_init = dict(x=0., xdot=0., theta=np.pi, thetadot=0.)
    pendulum_phys = dict(m=0.1, M=1.0, l=1.0, g=9.81, b=2.0, dt=0.02)

    def __init__(self):
        self.gen_history = []

    def learn(self):
        genetic_learn(self.nnet_size, self.pop_size, self.evaluate, self.stop,
                      activation='tanh')

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
                                    t * pd.dt
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

    def evaluate(self, nnet, sim=None):
        sim = sim or self.simulate(nnet, self.initial_conditions())
        return (-sum(abs(self.target['theta'] - p['theta']) * p['t'] for p in sim)
                -sum(abs(self.target['x'] - p['x']) * p['t'] for p in sim)
                -sum((w**2).sum() for w in nnet.weights) * self.regularization_coeff)

    def stop(self, i, pop):
        if i % self.plot_interval == 0:
            init = self.initial_conditions()
            sim = self.simulate(pop[0], init)
            with open('/tmp/last-run-%d.csv' % i, 'w') as f:
                writer = csv.DictWriter(f, sim[0].keys())
                writer.writeheader()
                writer.writerows(sim)
            Animate(sim, init.l, 2 * init.dt * 1000, tight_layout=True).show()

        best_worst = [self.evaluate(n) for n in pop[:2] + pop[-2:]]
        self.gen_history.append((i, best_worst[0], best_worst[-1]))
        print i, '    '.join(str(x) for x in best_worst)

        if self.weight_noise_stdev > 0:
            for nnet in pop[self.weight_noise_worst:]:
                for w in nnet.weights:
                    w += np.random.normal(0, self.weight_noise_stdev, w.shape)

        return False

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


if __name__ == '__main__':
    gp = GeneticPendulum()
    for arg in sys.argv[1:]:
        name, value = arg.split('=', 1)
        orig = getattr(gp, name)
        if type(orig) == list:  # lists must have homogenuous type
            value = [type(orig[0])(e) for e in value.split(',')]
        elif type(orig) == dict:
            for kvp in value.split(';'):
                k, v = kvp.split(':', 1)
                orig[k] = type(orig[k])(v)
            value = orig
        else:
            value = type(orig)(value)
        setattr(gp, name, value)

    gp.learn()