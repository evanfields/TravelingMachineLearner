module TravelingMachineLearner
using LinearAlgebra: norm
const TSP_SIZE = 10

include("solve_tsp.jl")

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

"""Map a TSP instance with `TSP_SIZE` cities to `TSP_SIZE` points of training data.
Each observation is a single "next arc" step in one-hot encoded form. Return a tuple
(X, Y) where X has `2 * TSP_SIZE^2` rows and `TSP_SIZE` columns, and `Y` is a square
`TSP_SIZE` x `TSP_SIZE` matrix."""
function tsp_to_data(dm, type = Float32)
    path = solve_tsp(dm)
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

"""Generate the Euclidean distance matrix for `n` points in the unit square."""
function generate_unit_square_distmat(n::Int = TSP_SIZE)
    pts = rand(2, n)
    return [norm(pts[:,i] - pts[:,j]) for i in 1:n, j in 1:n]
end

"""Generate combined training data for `n_tsp` unit square Euclidean TSP instances, each
with `TSP_SIZE` points. Return a tuple `(X, Y)`."""
function generate_data(n_tsps, tsp_size = TSP_SIZE)
    instances = [tsp_to_data(generate_unit_square_distmat(tsp_size)) for _ in 1:n_tsps]
    return (
        reduce(hcat, [inst[1] for inst in instances]), # X
        reduce(hcat, [inst[2] for inst in instances]), # Y
    )
end


end # module TravelingMachineLearner
