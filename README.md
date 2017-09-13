# Inverted Pendulum
Neural networks learning how to balance an inverted pendulum using evolutionary algorithms

Requires [matplotlib](http://matplotlib.org/) and the actual neural network code which can
be found in [another repository of mine](https://github.com/e-dorigatti/py_neuralnet). Just
clone it inside the directory in which you cloned this project.

The main script is [genetic_nnet.py](genetic_nnet.py) which takes parameter values as
command line arguments or uses [the default values](genetic_nnet.py#L13) if not specified.
The syntax is `name=value`; if the value is a list split items with commas, if it is a
dictionary split key value pairs with semicolon and split key from value with colon, 
for example`target=x:0;y:0`. For boolean false use the empty string as any other string
is interpreted as boolean true.

Every `checkpoint_interval` generations the script [dumps](genetic_nnet.py#L68) the
current population and a csv with a sample trial originating from the best performing
neural network found so far, and displays an animation of it, too!
You can see some samples [here](samples/) and animate them with the [animate.py](animate.py)
script; if you really like what you see you can also create a GIF animation from it.

![](https://cloud.githubusercontent.com/assets/5585926/10465064/404398a2-71ed-11e5-80ce-65af9b698ef7.gif)

There is also a parallel implementation of the algorithm which uses
[Apache Spark](http://spark.apache.org/); to run it download spark and use the included
`spark-submit` script to launch [spark_learn.py](spark_learn.py). If you need a
deployment more complex than local mode you will probably need to specify a couple of
parameters, at the very least `master`, `py-files` and `num_slices` (for better
parallelization, depends on CPU count and population size). For example:

```
find . -name '*.py' | xargs zip /tmp/genetic.zip
/path/to/spark/bin/spark-submit --master spark://somewhere:7077 \
                                --py-files /tmp/genetic.zip \
                                spark_learn.py num_slices=24
```

---

The file `dqn.py` contains my implementation of the Deep Q Network (which is not
deep, in this case), that learns to balance the pendulum with bang-bang controls.
The hyperparameters are not set to the optimal values, because my poor laptop is,
after all, only a laptop.

The algorithm was introduced in:

```
[Human-level control through deep reinforcement learning.](https://deepmind.com/research/dqn/)
V. Mnih, K. Kavukcuoglu, D. Silver, A. Rusu, J. Veness, M. Bellemare, A. Graves, M. Riedmiller, A. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis. 
Nature 518 (7540): 529-533 (2015)
```