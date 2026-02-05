using Pkg

Pkg.activate(".")

using CUDA
using Flux
using BenchmarkTools

#using Pluto
#Pluto.run(workspace_use_distributed=false)

function softmax_basic(x) 
	exps = exp.(x)
	return exps./sum(exps,dims=1)
end

#inner_max(x::X) where X = CUDA.max(x)

function softmax_kernel_sub_max(input::CuDeviceArray{T},acc::CuDeviceArray{T}) where {T}
    
    # Thread index
    i = (blockIdx().x - 1)* blockDim().x + threadIdx().x

    # Only in valid threads
    if i <= size(input,2)
        
        v_max = -1f8

        @view input[:,i]
        for i in @view input[:,i]
            if i > v_max
                v_max=i
            end
        end

        # Actual calculation
        (@view acc[:,i]) .= exp.((@view input[:,i]) .-v_max)
        sumsi = sum(view(acc,:,i))
        (@view acc[:,i]) ./= sumsi
    end
    
    return nothing

end


v1 = randn(19962,10000)

@btime begin
    threads_per_blockm = 1024 # 1024
    blocks_per_gridm = cld(size(v1, 2), threads_per_blockm)

    sumsm = CUDA.zeros(size(v1,2))
    accsm = CUDA.zeros(size(v1))

    @cuda threads=threads_per_blockm blocks=blocks_per_gridm softmax_kernel_sub_max(cu(v1), accsm)

end


#! CuDynamicSharedArray
#! Figure out how to get the online max. 
function softmax_kernel(input::CuDeviceArray{T}, sumsi::CuDeviceArray{T}, acc::CuDeviceArray{T}) where {T}
    # Thread index
    i = (blockIdx().x - 1)* blockDim().x + threadIdx().x
    j = (blockIdx().y - 1)* blockDim().y + threadIdx().y

    #sumsi = CuDynamicSharedArray(T, size(input,2))

    if i <= size(input,1) && j<=size(input,2)
        # Exponentiation and denominator addition
        acc[i,j] = exp(input[i,j])

        #! I may still need logsumexp?  
        #CUDA.atomic_add!(pointer(sumsi,j), acc[i,j])
    end
    
    return nothing
end

#! Shared Memory is very limited.
v1 = randn(Float32, (19962,16000));
@btime begin
	threads_per_block = (1024, 1) # 1024
	blocks_per_grid = (cld(size(v1, 1), threads_per_block[1]), cld(size(v1, 2),threads_per_block[2]))

	sums = CUDA.zeros(size(v1,2))
	accs = CUDA.zeros(size(v1))
	
	@cuda threads=threads_per_block blocks=blocks_per_grid softmax_kernel(cu(v1), sums, accs)
	
	# Grid-synced division, kind of cheating. 
	#res_sm = accs./sums[:,:]'
end

