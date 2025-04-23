# Can I KMeans? 

Ok, this is nothing fancy! It's just for practice.
The code implement KMeans using (1) python lists, (2) numpy, and (3) torch. The goal is simply to remember the useful functions from torch and numpy. Also, I hear this is often an interview question.

_Note: Everything runs on CPU here!_

Here is how you can run the script:
`python3 kmeans.py -n 100 -d 10 -k 3`

And here is a sample output:
```
== Solving with basic python
Itt: 0 -- average distance: 0.8020002010509966
Itt: 1 -- average distance: 0.7359509831730455
Itt: 2 -- average distance: 0.7273557744878388
Itt: 3 -- average distance: 0.7235463810361091
Itt: 4 -- average distance: 0.7214399515420683
Itt: 5 -- average distance: 0.7192774531597346
Itt: 6 -- average distance: 0.7188732037953071
== Solving with numpy
Itt: 0 -- average distance: 0.8020002010509966
Itt: 1 -- average distance: 0.7359509831730453
Itt: 2 -- average distance: 0.7273557744878387
Itt: 3 -- average distance: 0.7235463810361089
Itt: 4 -- average distance: 0.7214399515420685
Itt: 5 -- average distance: 0.7192774531597347
Itt: 6 -- average distance: 0.7188732037953071
== Solving with torch
Itt: 0 -- average distance: 0.8020002129673958
Itt: 1 -- average distance: 0.7359509813785553
Itt: 2 -- average distance: 0.7273557758331299
Itt: 3 -- average distance: 0.7235463799536228
Itt: 4 -- average distance: 0.7214399561285972
Itt: 5 -- average distance: 0.7192774611711502
Itt: 6 -- average distance: 0.7188732054829597
```

As you can see all the outputs match which mean (in a very unscientific way) that our implementation is correct! 
