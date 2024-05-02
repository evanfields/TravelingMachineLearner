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
* In a symmetric TSP, there are two optimal paths for two directions of traversing the
optimal circuit. This might make training harder because the network doesn't know which
solution the training data has. Probably we want to somehow break the symmetry. We would
think that this forces the model to coin-flip guess the first city which is 1/10 of choices
so bakes in a minimum 5% error?
    * add a little nonsymmetric noise to all distance matrices
    * have a convention of favoring circuits that start at lower indices
    * braeking the symmetry seems to reduce training loss a tiny bit, like .26 to .24

# Best model so far
```
function _default_model(problem_size)
    input_size = 2 * problem_size^2
    output_size = problem_size
    return Flux.Chain(
        Flux.Dense(input_size, 200, Flux.gelu),
        Flux.Dense(200, 80, Flux.gelu),
        Flux.Dense(80, 35, Flux.gelu),
        Flux.Dense(35, output_size, identity),
        # logitcrossentropy loss, no softmax
    )
end
```
* This [with relu not gelu] is a wider variant of a previous model, which seems to help a tiny bit, maybe .28 => .25 train loss. Wider > deeper here?
* Swapping relu for gelu seems to help a bunch with train loss, down to like .14, presumably some performance penalty?
* Generating 40k rather than 30k training examples and adding another 80=>80 layer in the middle maybe gets train 
loss down to ~.12, which is modest but helpful?