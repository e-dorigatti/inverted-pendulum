from __future__ import print_function
import click
import csv
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from collections import defaultdict


class Animate:
    def __init__(self, data_points, rod_length, delta_t, blit=True, **fig_kwargs):
        self.animation_length = len(data_points)
        self.rod_length = rod_length
        self.delta_t = delta_t

        self.data = defaultdict(list)
        for point in data_points:
            for k, v in point.items():
                self.data[k].append(float(v))

        self.fig = plt.figure(**fig_kwargs)
        self._create_pendulum()
        self._create_graph()

        self.animation = animation.FuncAnimation(self.fig, self.animation_step,
                                                 frames=self.animation_length,
                                                 init_func=self.animation_init,
                                                 interval=self.delta_t,
                                                 blit=blit)

    def _create_pendulum(self):
        """ creates the plot for the animated pendulum (left) """
        xmin, xmax = (min(self.data['x']) - self.rod_length,
                     max(self.data['x']) + self.rod_length)
        ymin, ymax = (-self.rod_length * 2, self.rod_length * 2)
        self.pendulum = self.fig.add_subplot(1, 2, 1,
                                             aspect='equal', autoscale_on=False,
                                             xlim=(xmin, xmax), ylim=(ymin, ymax))
        self.pendulum.grid()
        self.pendulum.title.set_text('Animated Pendulum')
        self.pendulum.plot([0., 0.], [ymin, ymax], 'b--')
        self.pendulum.xaxis.label.set_text('Cart Position (m)')
        self.rod, = self.pendulum.plot([], [], 'o-', lw=2)

    def _create_graph(self):
        """ creates the plot for the actual data (right) """
        ymin = min(self.data['x'] + self.data['theta'] + self.data['force'])
        ymax = max(self.data['x'] + self.data['theta'] + self.data['force'])
        self.graph = self.fig.add_subplot(1, 2, 2,
                                          xlim=(0, max(self.data['t']) + 1),
                                          ylim=(ymin - 1, ymax + 1))
        self.theta_plot, = self.graph.plot([], [], 'b-', label='Theta (deg)')
        self.force_plot, = self.graph.plot([], [], 'g-', label='Force (norm)')
        self.pos_plot, = self.graph.plot([], [], 'r-', label='Position (m)')
        self.graph.grid(); self.graph.legend()
        self.graph.xaxis.label.set_text('Time (s)')

    def _plot_pendulum(self, t):
        """ plots the state of the pendulum """
        # theta=0 is upward in our data but righward everywhere else
        x0, y0 = self.data['x'][t], 0
        x1, y1 = (x0 + self.rod_length * np.cos(self.data['theta'][t] + np.pi / 2),
                  y0 + self.rod_length * np.sin(self.data['theta'][t] + np.pi / 2))

        self.rod.set_data([x0, x1], [y0, y1])

    def _plot_graph(self, t):
        """ plots the simulation data up until now """
        time = self.data['t'][:t]
        self.theta_plot.set_data(time, self.data['theta'][:t])
        self.force_plot.set_data(time, self.data['force'][:t])
        self.pos_plot.set_data(time, self.data['x'][:t])

    def animation_init(self):
        self.rod.set_data([], [])
        self.theta_plot.set_data([], [])
        self.force_plot.set_data([], [])
        self.pos_plot.set_data([], [])
        return self.theta_plot, self.force_plot, self.pos_plot, self.rod

    def animation_step(self, i):
        t = i % self.animation_length
        self._plot_pendulum(t)
        self._plot_graph(t)
        return self.theta_plot, self.force_plot, self.pos_plot, self.rod

    def show(self):
        plt.show()


@click.command()
@click.argument('input-run', type=click.File('r'))
@click.option('--rod-length', default=0.5, help='Length of the rod (m)')
@click.option('--delta-t', default=20, help='Simulation delta t (ms)')
@click.option('--gif-name', type=click.STRING,
              help='If specified, save a GIF with this filename')
@click.option('--gif-fps', type=click.INT, default=24)
@click.option('--width', type=click.FLOAT, default=8.,
              help='Width of the animation (inches)')
@click.option('--height', type=click.FLOAT, default=8.,
              help='Height of the animation (inches)')
@click.option('--dpi', type=click.FLOAT, default=96.)
@click.option('--blit/--no-blit', default=True)
def main(input_run, rod_length, delta_t, gif_name, gif_fps, width, height, dpi, blit):
    data = [p for p in csv.DictReader(input_run)]
    anim = Animate(data, rod_length, delta_t, blit, dpi=dpi, tight_layout=True,
                   figsize=(width, height))

    if gif_name and gif_fps:
        print('Saving as animated gif')
        anim.animation.save(gif_name, writer='imagemagick', fps=gif_fps)
    else:
        print('Showing animation')
        anim.show()

if __name__ == '__main__':
    main()
