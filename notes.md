# Things to try
* Does the ML model need to predict the last arc that closes a circuit? Including this arc
feels a bit more elegant and seems to ever so slightly simplify representation, but in fact
we already know how to mechanically connect the last visited city back to the first city,
so maybe this is just needless extra complexity and the model could focus on a narrower
task?
* Step by step the model predicts the next city. Should it get city 1 for free, or have
to predict that when we haven't visited any cities yet we start at city 1?
    * Seems best to give the model city 1 for free, which is without loss of generality anyway.
    Otherwise we need to change the formulation of representing selected arcs; "we've
    been nowhere yet" and "we've started at city 1" both look identical with no arcs
    yet traversed.
* For a given TSP instance, should we shuffle the order of the cities to generate more
training data?
* For a given TSP instance, should we sort the cities to provide a more predictable
input => output mapping for the model to learn?