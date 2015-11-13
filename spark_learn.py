from sys import argv
from py_neuralnet import genetic_learn_spark
from pyspark import SparkContext, SparkConf
from genetic_nnet import GeneticPendulum, parse_args


class SparkGeneticPendulum(GeneticPendulum):
    def learn(self, sc):
        genetic_learn_spark(sc, self.nnet_size, self.pop_size, self.evaluate, self.stop,
                            activation=self.nnet_activation)


if __name__ == '__main__':
    conf = SparkConf().setAppName('Inverted Pendulum')
    sc = SparkContext(conf=conf)
    gp = parse_args(SparkGeneticPendulum())
    gp.learn(sc)
