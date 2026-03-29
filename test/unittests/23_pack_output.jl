@testset "pack_output and pack_output!" begin
    y = reshape(collect(1.0:18.0), 6, 3)

    packed = NNFoil.pack_output(y)
    @test packed.analysis_confidence == y[1, :]
    @test packed.CL == y[2, :]
    @test packed.CD == y[3, :]
    @test packed.CM == y[4, :]
    @test packed.Top_Xtr == y[5, :]
    @test packed.Bot_Xtr == y[6, :]

    y[2, 1] = 99.0
    @test packed.CL[1] == 99.0

    output = unit_output_buffer(size(y, 2))
    NNFoil.pack_output!(output, y)
    @test output.analysis_confidence == y[1, :]
    @test output.CL == y[2, :]
    @test output.CD == y[3, :]
    @test output.CM == y[4, :]
    @test output.Top_Xtr == y[5, :]
    @test output.Bot_Xtr == y[6, :]
end
