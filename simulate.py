from __future__ import print_function
from math import sin, cos, pi
import matplotlib.pyplot as plt


class PendulumDynamics:
    m = 0.2     # mass of the pendulum (kg)
    M = 1.0     # mass of the cart (kg)
    l = 0.5     # length of the rod (m)
    g = 9.81    # gravitational acceleration (m/s^2)
    b = 5.0     # friction coefficient (?)
    dt = 0.02   # simulation delta time (s)

    def __init__(self, x, xdot, theta, thetadot):
        self.x = x
        self.xdot = xdot
        self.theta = theta
        self.thetadot = thetadot

    def step_simulate(self, force):
        xacc_numerator = (force -
                          self.m * self.l * self.thetadot**2 * sin(self.theta) +
                          self.m * self.g * sin(self.theta) * cos(self.theta) -
                          self.b * self.xdot)

        xacc_denominator = self.M + self.m - self.m * cos(self.theta)**2
        xacc = xacc_numerator / xacc_denominator

        theta_acc = (xacc*cos(self.theta) + self.g * sin(self.theta)) / self.l

        self.xdot = self.xdot + xacc * self.dt
        self.x = self.x + self.xdot * self.dt

        self.thetadot = self.thetadot + theta_acc * self.dt
        self.theta = self.theta + self.thetadot * self.dt

    @property
    def state(self):
        return self.x, self.xdot, self.theta, self.thetadot


def to_deg(rad):
    return rad*180/pi


def plot(time, positions, angles):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Cart position')
    plt.plot(time, [x for x, xdot in positions], label='Position (m)')
    plt.plot(time, [xdot for x, xdot in positions], label='Accelaration(m/s^2)')
    plt.xlabel('Time (s)')
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title('Cart angle')
    plt.plot(time, [tdot for t, tdot in angles], label='Angle variation (deg/s)')
    plt.plot(time, [t for t, tdot in angles], label='Angle (deg)')
    plt.plot([0, time[-1]], [180, 180], color='gray', linestyle='--')
    plt.xlabel('Time (s)')
    plt.grid()
    plt.legend()

    plt.show()


def drop_test():
    print('simple drop test, do the plots look realistic?')

    cart = PendulumDynamics(0, 0, 0.05, 0)

    time = [x * cart.dt for x in range(250)]
    positions, angles = [], []
    for i in time:
        positions.append((cart.x, cart.xdot))
        angles.append((to_deg(cart.theta), to_deg(cart.thetadot)))

        cart.step_simulate(0)

    plot(time, positions, angles)


def balance_test():
    print('trying to balance the thing...')

    cart = PendulumDynamics(0, 0, 0.05, 0)

    time = [x * cart.dt for x in range(250)]
    positions, angles = [], []
    for i in time:
        positions.append((cart.x, cart.xdot))
        angles.append((to_deg(cart.theta), to_deg(cart.thetadot)))

        force = cart.theta * -70
        cart.step_simulate(force)

    plot(time, positions, angles)


if __name__ == '__main__':
    drop_test()
    balance_test()
