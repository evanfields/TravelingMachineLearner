@testset "Extracting a tour" begin
    # One tour in the input
    bm = Bool.([0 1 0; 0 0 1; 1 0 0])
    tour = TML._extract_tour(bm)
    @test tour == [1, 2, 3]

    # Two subtours in the input
    bm = Bool.([0 1 0 0; 1 0 0 0; 0 0 0 1; 0 0 1 0])
    tour = TML._extract_tour(bm)
    @test tour == [1, 2]

    # Starting not at firstcity = 1
    bm = Bool.([0 0 1; 1 0 0; 0 1 0])
    tour = TML._extract_tour(bm, 2)
    @test tour == [2, 1, 3]
end

@testset "Extracting multiple tours" begin
    # One tour in the input
    bm = Bool.([0 1 0; 0 0 1; 1 0 0])
    tours = TML._extract_multiple_tours(bm)
    @test length(tours) == 1
    @test tours[1] == [1, 2, 3]

    # Two tours in the input
    bm = Bool.([0 1 0 0; 1 0 0 0; 0 0 0 1; 0 0 1 0])
    tours = TML._extract_multiple_tours(bm)
    @test length(tours) == 2
    @test tours[1] == [1, 2]
    @test tours[2] == [3, 4]
end

@testset "Solving a small TSP instance" begin
    # Traditional 6-point Euclidean plane instance
    points = [
        (0, 0),
        (1, 0),
        (1, 1),
        (5, 0),
        (5, 1),
        (6, 0)
    ]
    distmat = [
        sqrt((points[i][1] - points[j][1])^2 + (points[i][2] - points[j][2])^2)
        for i in 1:length(points), j in 1:length(points)
    ]
    for triangle_cut_thresh in [0, 1.5] 
        tour = TML.solve_tsp(distmat; triangle_cut_thresh)
        @test length(tour) == 7  # Closed tour
        @test Set(tour) == Set(1:6)  # All cities visited
    end
end