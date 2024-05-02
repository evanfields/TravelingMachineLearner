module TravelingMachineLearner

import LinearAlgebra: norm
import Random: Xoshiro, GLOBAL_RNG


import StatsBase
import Flux
import Flux: gpu, cpu

const TSP_SIZE = 10

include("solve_tsp.jl")

##
# Helpers
##

"""Map a path through one or more cities (must be non-empty and start at city 1) to a
binary matrix of arc indicators. Roughly the inverse of `_extract_tour`."""
function _path_to_arc_mat(path, n_cities; type = Float32)
    @assert length(path) >= 1
    @assert path[1] == 1
    arc_indicators = zeros(type, n_cities, n_cities)
    for i in 2:length(path)
        arc_indicators[path[i-1], path[i]] = 1
    end
    return arc_indicators
end

function model_to_path(distmat, model; n_arcs = size(distmat, 1))
    path = [1]
    n_cities = size(distmat, 1)
    for _ in 1:n_arcs
        x = vcat(vec(distmat), vec(_path_to_arc_mat(path, n_cities)))
        next_city_weights = Flux.softmax(model(x)) # model returns logits
        next_city = StatsBase.sample(1:n_cities, StatsBase.Weights(next_city_weights))
        push!(path, next_city)
    end
    return path
end

##
# Data generation
##

"""Map a TSP instance with `TSP_SIZE` cities to `TSP_SIZE` points of training data.
Each observation is a single "next arc" step in one-hot encoded form. Return a tuple
(X, Y) where X has `2 * TSP_SIZE^2` rows and `TSP_SIZE` columns, and `Y` is a square
`TSP_SIZE` x `TSP_SIZE` matrix.

If `break_symmetry`, prefer path directions that go from city 1 (always the start) to low
index cities."""
function tsp_to_data(dm, type = Float32; break_symmetry = true)
    path = solve_tsp(dm)
    if break_symmetry
        cost = sum(dm[path[i], path[i+1]] for i in 1:(length(path) - 1))
        cost_rev = sum(dm[path[i+1], path[i]] for i in 1:(length(path) - 1))
        can_reverse = abs(cost - cost_rev) < 1e5
        if can_reverse && path[2] > path[end-1]
            path = reverse(path)
        end
    end
    n_cities = size(dm, 1)
    @assert path[1] == path[end] == 1
    @assert length(path) == n_cities + 1
    
    X = zeros(type, 2 * n_cities^2, n_cities)
    Y = zeros(type, n_cities, n_cities)
    for ind in 1:(length(path)-1) # each city we've "reached"; start at city 1 "for free"
        # goal here is to predict the city at path[ind + 1]
        arc_indicators = _path_to_arc_mat(path[1:ind], n_cities; type)
        X[:, ind] = vcat(vec(dm), vec(arc_indicators))
        Y[path[ind + 1], ind] = 1
    end
    return X, Y
end

"""Generate the Euclidean TSP for `n` points in the unit square. Return a named tuple
with keys :pts and :distmat. If `order_points`, arrange points so that lower indices
are more central."""
function generate_unit_square_tsp(n::Int, rng = GLOBAL_RNG; order_points = false)
    pts = rand(rng, 2, n)
    distmat = [norm(pts[:,i] - pts[:,j]) for i in 1:n, j in 1:n]
    if order_points
        order = sum(distmat, dims=1) |> vec |> sortperm
        distmat = distmat[order, order]
        pts = pts[:, order]
    end
    return (;pts, distmat)
end

"""Generate combined training data for `n_tsp` unit square Euclidean TSP instances, each
with `TSP_SIZE` points. Return a tuple `(X, Y)`."""
function generate_data(n_tsps, tsp_size = TSP_SIZE, rng = GLOBAL_RNG)
    instances = [
        tsp_to_data(generate_unit_square_tsp(tsp_size, rng).distmat)
        for _ in 1:n_tsps
    ]
    return (
        reduce(hcat, [inst[1] for inst in instances]), # X
        reduce(hcat, [inst[2] for inst in instances]), # Y
    )
end

function _default_model(problem_size)
    input_size = 2 * problem_size^2
    output_size = problem_size
    return Flux.Chain(
        Flux.Dense(input_size, 200, Flux.gelu),
        Flux.Dense(200, 80, Flux.gelu),
        Flux.Dense(80, 80, Flux.gelu),
        Flux.Dense(80, 35, Flux.gelu),
        Flux.Dense(35, output_size, identity),
        # logitcrossentropy loss, no softmax
    )
end

const INIT_MODEL = _default_model(TSP_SIZE)

function train_model(;
    tsp_size = TSP_SIZE,
    model = _default_model(tsp_size),
    loss_fn = Flux.Losses.logitcrossentropy,
    n_tsps = 1_000,
    rng = Xoshiro(47),
    data = generate_data(n_tsps, tsp_size, rng),
    n_epochs = 500,
    batch_size = 1_000,
    opt = Flux.Adam(.001),
    use_gpu = false,
    print_every = 25
)
    # set up data
    X, Y = data
    if use_gpu
        X, Y = gpu(X), gpu(Y)
    end
    train_data = Flux.DataLoader((X, Y); batchsize = batch_size, shuffle = true, rng)
    @info "Generated training data" size(X)

    # prepare for training
    loss(m, x, y) = loss_fn(m(x), y)
    if use_gpu
        model = gpu(model)
    end
    opt_state = Flux.setup(opt, model)
    @info "Ready for training" use_gpu
    train_losses = []
    push!(train_losses, cpu(loss(model, X, Y)))

    # main train loop
    for epoch in 1:n_epochs
        if epoch % print_every == 1
            @info "Start of epoch $(epoch)" train_losses[end]
        end
        GC.gc(false)
        Flux.train!(loss, model, train_data, opt_state)
        push!(train_losses, cpu(loss(model, X, Y)))
    end
    @info "Done training" train_losses[end]
    return cpu(model)
end


end # module TravelingMachineLearner
