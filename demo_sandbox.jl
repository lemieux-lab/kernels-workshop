using Pkg

Pkg.activate(".")

using CUDA
using Flux
using BenchmarkTools

#(d*n)
function softmax_mine(mat)
    exps = exp.(mat)
    return exps./sum(exps,dims=1)
end

v = randn(20000,10000)

@btime softmax_mine(v)
@btime softmax_mine(cu(v))


function softmax_column_kernel(input, output)
    col_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if col_idx <= size(input,2)
        (@view output[:,col_idx]) .= exp.((@view input[:,col_idx]))
        sum_tmp = sum((@view output[:,col_idx]))
        (@view output[:,col_idx]) ./= sum_tmp
    end

    return nothing 
end

threads_per_block = 1024
blocks_per_grid = cld(size(v,2), threads_per_block)

@profview begin
    result = Matrix{Float32}(undef, size(v)...) |> cu
    @cuda threads=threads_per_block blocks=blocks_per_grid softmax_column_kernel(cu(v), result)
end



# On block per column, a chunk of rows per thread
#! 5x faster than previous kernel, unstable, no maximum subtraction
function softmax_kernel(input::CuDeviceArray{T}, sumsi::CuDeviceArray{T}, acc::CuDeviceArray{T}) where {T}
    # Thread index
    chunk_size = size(input,1) ÷ blockDim().x #! Only cuz the size is good

    i = (blockIdx().x - 1)* blockDim().x + threadIdx().x -1
    base_i = i * chunk_size
    j = (blockIdx().y - 1)* blockDim().y + threadIdx().y

    if j <= size(input,2)
        #! Unfinished attempt with shared arrays, not needed
        # chunk_sums = CuStaticSharedArray(T, 1024)
        for idx in 1:chunk_size
            if idx + base_i <= size(input, 1)
                acc[idx + base_i,j] = exp(input[idx + base_i,j])    
                # chunk_sums[i+1] += acc[idx + base_i,j] 
                sumsi[i+1,j] += acc[idx + base_i,j]         
            end
        end 
        
        sync_threads()
        
        if i==0
            sumsi[1,j] = sum(@view sumsi[:,j])
        end
        
        sync_threads()
        
        for idx in 1:chunk_size
            acc[idx+base_i,j] /= sumsi[1,j]
        end
    end
    return nothing    
end

v1 =  cu(randn(20000,10000))

threads_per_block = (1024, 1) # 1024
blocks_per_grid = (cld(1, threads_per_block[1]), cld(size(v1, 2),threads_per_block[2]))

@btime begin
    sums = CUDA.zeros(1024, 10240)
    accs = CUDA.zeros(size(v1))	
	@cuda threads=threads_per_block blocks=blocks_per_grid softmax_kernel(v1, sums, accs)
end
sums

sum(accs,dims=1)

softmax(v1) .- accs

maximum(softmax(v1) .- accs)
