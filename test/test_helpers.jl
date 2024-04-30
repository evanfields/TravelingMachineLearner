@testset "_path_to_arc_mat" begin
    # Valid path
    path = [1, 2, 3, 4, 1]
    n_cities = 4
    expected_output = [
        0 1 0 0;
        0 0 1 0;
        0 0 0 1;
        1 0 0 0;
    ]
    @test TML._path_to_arc_mat(path, n_cities) == expected_output

    # Small path
    path = [1, 2]
    n_cities = 4
    expected_output = [
        0 1 0 0;
        0 0 0 0;
        0 0 0 0;
        0 0 0 0;
    ]
    @test TML._path_to_arc_mat(path, n_cities) == expected_output

    # Path with a single city
    path = [1]
    n_cities = 1
    expected_output = zeros(Float32, 1, 1)
    @test TML._path_to_arc_mat(path, n_cities) == expected_output

    # Invalid path - does not start at city 1
    path = [2, 3, 4]
    n_cities = 4
    @test_throws AssertionError TML._path_to_arc_mat(path, n_cities)
end