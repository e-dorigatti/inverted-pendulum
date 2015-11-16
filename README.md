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

Every `checkpoint_interval` generations the script [dumps](genetic_nnet.py#L16) the
current population and a csv with a sample trial originating from the best performing
neural network found so far, and displays an animation of it, too!
You can see some samples [here](samples/) and animate them with the [animate.py](animate.py)
script; if you really like what you see you can also create a GIF animation from it.

![](https://cloud.githubusercontent.com/assets/5585926/10465064/404398a2-71ed-11e5-80ce-65af9b698ef7.gif)

There is also a parallel implementation of the algorithm which uses
[Apache Spark](http://spark.apache.org/); to run it download spark and use the included
`spark-submit` script to launch [spark_learn.py](spark_learn.py).
