This experiment serves to show that even in purely Gaussian setting, where the Categorical distribution is trivial, using the magnet still helps.

The underlying function is f(x,y) = (2x - 1) * (2y - 1) = 4xy - 2x - 2y + 1. The dynamics here should be the same as in f(x, y) = xy, but the equilibrium is shifted to 0.5 from 0.

TODO: Add the usual f(x, y) = xy as it is more canonical example

run with: 
python run_idealized.py {config} 
or
python train.py {config}

wo_magnet.yaml is no magnet, so the experiment fails to converge.
with_magnet.yaml changes only adds the magnet. It conerges

Note that in average strategies (target), both converge.
