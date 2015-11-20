from sys import argv
from py_neuralnet import genetic_learn_spark
from pyspark import SparkContext, SparkConf
from genetic_nnet import GeneticPendulum, parse_args


class SparkGeneticPendulum(GeneticPendulum):
    num_slices = 4

    def learn(self, sc):
        genetic_learn_spark(sc, self, self.nnet_size, self.pop_size, self.num_slices)


if __name__ == '__main__':
    conf = SparkConf().setAppName('Inverted Pendulum')
    sc = SparkContext(conf=conf)
    gp = parse_args(SparkGeneticPendulum())
    gp.learn(sc)
