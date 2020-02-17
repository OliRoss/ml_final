# Tasks

LFA Agent trainieren und Policy (.npy) speichern.
Parameter:
* gamma = 0.9
* poly_degree = 2
* step_size = 0.02
* num_episodes = 1000 (mit automatischer Speicherung bei 500)

Random seeds:

Raoul: 2611
Oli: 12345678

Best practice:
LFA: 
* random weights 
* first tanh
* then sample of gauss
* calculating log probs with clipped action
* gamma = 0.9
* poly_degree = 2
* step_size = 0.02

NN: 
* 8-512-128-4 
* learning the variance
* gamma = 0.9
* step_size = 0.001

* best policy so far (mo 14:07, avg. reward > +72): NN2020_17_02_10:17params_0.001_5000_0.9_None_best
