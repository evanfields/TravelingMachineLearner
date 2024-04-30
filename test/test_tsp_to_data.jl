@testset "Four city TSP" begin
    # in this TSP the only reasonable path is [1,2,4,3,1]
    dm = [
        5 1 5 5;
        5 5 5 1;
        1 5 5 5;
        5 5 1 5
    ]
    X, Y = TML.tsp_to_data(dm)
    opt_path = [1,2,4,3,1]
    # build edge-indicator matrix step by step, confirming that
    # the X columns match
    edge_mat = zeros(4, 4)
    for i in 1:4
        expected_x = vcat(vec(dm), vec(edge_mat))
        @test X[:, i] == expected_x
        # prepare edge mat for next iteration with the next edge
        edge_mat[opt_path[i], opt_path[i+1]] = 1
        expected_y = zeros(4)
        expected_y[opt_path[i+1]] = 1
        @test Y[:, i] == expected_y
    end
end
