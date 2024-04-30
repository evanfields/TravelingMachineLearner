import JuMP
import JuMP: @constraint, @objective, @variable
import Statistics: mean
import StaticArrays: SVector
import HiGHS
import Logging
import Logging: @logmsg

"Extract the subtour starting at a specified city. `bm` is a binary matrix. Returned tours
are not closed."
function _extract_tour(bm, firstcity = 1)
	tour = [firstcity]
	curcity = tour[end]
	nextcity = findfirst(bm[curcity,:])
	@assert !isnothing(nextcity)
	while !(nextcity in tour)
		push!(tour, nextcity)
		curcity = nextcity
		nextcity = findfirst(bm[curcity,:])
        @assert !isnothing(nextcity)
	end
	return tour
end

"extract several subtours, presumably to cut them all in a callback"
function _extract_multiple_tours(bm, maxtours = size(bm,1))
	available_cities = Set(1:size(bm,1))
	tours = Vector{Vector{Int}}()
	while !isempty(available_cities)
		firstcity = minimum(available_cities)
		tour = _extract_tour(bm, firstcity)
		setdiff!(available_cities, tour)
		push!(tours, tour)
	end
	numtours = min(maxtours, length(tours))
	return tours[1:numtours]
end

"Map a JuMP variable to a binary matrix."
_jump_to_bool(mat, thresh = .5) = JuMP.value.(mat) .>= thresh

"""Setup a JuMP model for solving a TSP instance, returning a tuple (model, variables)
where variables is the binary matrix of edge variables. Pre-cut length 2 subtours and
length-3 subtours where each edge has length `<= triangle_cut_thresh`. Note this
function does not assume that the input problem is symmetric, and more efficient
formulations are possible for known symmetric instances."""
function _setup_jump_model(
    distmat;
    triangle_cut_thresh = mean(distmat) / 10,
    opt = HiGHS.Optimizer
)
    n = size(distmat, 1)
    @assert size(distmat, 2) == n
    m = JuMP.Model(opt)
	JuMP.set_attribute(m, "output_flag", false)
	@variable(m, x[1:n, 1:n], Bin)

    # non-subtour constraints
	for i in 1:n
		@constraint(m, x[i, i] == 0) # no self-loop
		@constraint(m, sum(x[i, j] for j in 1:n) == 1) # leave each city once
		@constraint(m, sum(x[j, i] for j in 1:n) == 1) # enter each city once
	end

    # eliminate length 2 subtours/cycles
	if n > 2
		for i in 1:(n-1), j in (i+1):n
			if i == j
				continue
			end
			@constraint(m, x[i,j] + x[j,i] <= 1)
		end
	end

    # eliminate length 3 subtours
	if n > 3 && triangle_cut_thresh > 0.0
		cutoff = mean(distmat) * triangle_cut_thresh
		triangles = Set{SVector{3, Int}}()
		# find first short edge
		for i in 1:n, j in 1:n
			if i == j || distmat[i,j] > cutoff
				continue
			end
			# complete triangles around short edge
			for k in 1:n
				if k == i || k == j || distmat[j,k] > cutoff || distmat[k,i] > cutoff
					continue
				end
				push!(triangles, sort(SVector{3}([i, j, k])))
			end
		end
		# add all triangles
		for tri in triangles
			i, j, k = tri[1], tri[2], tri[3]
			@constraint(m, x[i,j] + x[j,i] + x[i,k] + x[k,i] + x[j,k] + x[k,j] <= 2)
		end
	end

    @objective(m, Min, sum(distmat[i,j] * x[i,j] for i in 1:n, j in 1:n))

	return m, x
end

"""
"Rotate" a closed circuit so that it starts (and ends) at `desired_start`. E.g.
`rotate_circuit([1,2,3,1], 2) == [2,3,1,2]`. `circuit` must be a closed path.
"""
function _rotate_path(circuit, desired_start)
    if first(circuit) != last(circuit)
        throw(DomainError(circuit, "Circuit passed to rotate_circuit is not closed."))
    end
    first(circuit) == desired_start && return copy(circuit)
    start_ind = findfirst(circuit .== desired_start)
    return vcat(
        circuit[start_ind:end],
        circuit[2:start_ind]
    )
end

"""Solve a TSP specified by a distance matrix. Return a closed list of integer indices
representing an optimal path starting at index 1. The returned list will also end with
index 1."""
function solve_tsp(
	distmat;
	log_level = Logging.Debug,
    triangle_cut_thresh = mean(distmat) / 10,
)
    model, arc_variables = _setup_jump_model(distmat; triangle_cut_thresh)
    num_cities = size(distmat, 1)
    @logmsg log_level "Built model"
    JuMP.optimize!(model)
    @logmsg log_level "First solve complete"
    tours = _extract_multiple_tours(_jump_to_bool(arc_variables))
    @logmsg log_level "Found $(length(tours)) [sub]tours"
    while length(tours) > 1
        for subtour in tours
            @logmsg log_level "Eliminating subtour" subtour
            @constraint(
                model,
                sum(arc_variables[i,j] for i in subtour, j in subtour) <= length(subtour) - 1
            )
        end
        JuMP.optimize!(model)
        @logmsg log_level "Resolve complete"
        tours = _extract_multiple_tours(_jump_to_bool(arc_variables))
        @logmsg log_level "Found $(length(tours)) [sub]tours"
    end
    path = _extract_tour(_jump_to_bool(arc_variables))
    push!(path, path[1]) # close the tour
    return _rotate_path(path, 1)
end
