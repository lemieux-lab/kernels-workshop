### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 69fc927d-adee-4c6d-aff6-a9a35350070d
using Pkg

# ╔═╡ 7a24781f-c7d5-4b17-85d8-f040eae59fa5
using CUDA

# ╔═╡ 2d61a06c-1d62-4979-b92f-784a92741ee6
CUDA.device()

# ╔═╡ cdc41743-80d0-449c-a43f-a4685a8d75e3
md"""
# Hello! :)
"""

# ╔═╡ afa298c8-f482-40ca-a73d-80ffab61c8e7
md"""
In Julia, GPU usage is already optimized for many processes through CUDA.jl. Simply by applying a function to a CuArray, operations (e.g. broadcasting and map-reducing) are executed on GPU-specialized code. Additionally, more complex tasks, such as operations in machine learning algorithms like self-attention, have optimized code through cuDNN.jl. 

These are done through GPU kernels, which implement functions that exploit the CUDA architecture of GPUs.

Today, we will show how to write such kernels, for when already-optimized kernels do not already exist.
 
"""

# ╔═╡ 2627ab83-84ed-4ed3-9dfc-95a1bdb96bcf
md"""
# GPU CUDA Architecture

![gpu](https://modal-cdn.com/gpu-glossary/light-green-cuda-programming-model.svg)
*https://modal.com/gpu-glossary/device-software/thread*

## Threads
Threads are the lowest level of the hierarchy (like in CPUs)! With GPUs, however, all threads within a **warp** should share the same task. While every thread that's called in the kernel executes the same code, they can process different parts of the data. 

Thus, threads can coordinate/sync with one another within a block using shared memory. Can be indexed (x, y, z).

## Blocks
Blocks are the smallest unit of thread coordination, wherein each block must execute independently and without order (in parallel, given enough resources). A single CUDA kernel launch produces one or more thread blocks that run asynchronously. 

Blocks are arbitrarily sized (up to 1028), but typically multiples of warp size (up to 32). Can also be indexed (x, y, z).

## Grids
Grids are made up of a collection of thread blocks, and this spans the entire GPU (basically global context of the kernel). Can be 1D, 2D, or 3D.

## Warps
A warp is a group of threads that are scheduled together and execute in parallel. Since blocks don't necessarily have to share the same task, warps are the unit of execution on the GPU. Thus, all threads within a warp will have the same task. 

This isn't actually part of the GPU architectural hierarchy in CUDA, but more an implementation detail that's useful to keep in mind for optimization.
"""

# ╔═╡ 1d38f704-0c74-41e0-ad24-e37e7a8dd66c
md"""
# Kernels

A kernel is essentially a function that launches/returns only **once** but is executed many times; once each by x number of threads. These occur in random order and simulataneously (these are what we assign to the warps!).

## Launching a kernel
"""

# ╔═╡ b496b692-d39b-4f25-ae71-f0a697681175
md"""
First, we can define our kernel function.
"""

# ╔═╡ 4a301a05-13c4-4561-be78-a766f05670de
function kernel()
	# do stuff here
	return nothing
end

# ╔═╡ a57c01b2-1c34-4c83-bda7-a7b1a4599bab
md"""
Then, we can launch it using `@cuda`. This launches a single thread.
"""

# ╔═╡ 003a2324-37c3-487c-967b-733ea815ccd6
@cuda kernel()

# ╔═╡ 6fb64c82-60e3-43ec-bf64-96acc016867c
md"""
We can also get an object out of the compiled kernel for additional info!
"""

# ╔═╡ 4f264902-ad19-4139-bda8-20645fcca984
k = @cuda launch=false kernel()

# ╔═╡ 5429e43c-a42b-4257-83be-02b6bdb99155
CUDA.registers(k)

# ╔═╡ 18dcd360-505c-4321-b11b-cd55c63e808e
md"""
The `launch=false` compiles the function without actually executing it. The result is a HostKernel object. 

Looking at the `.registers()` essentially shows the complexity of the kernel (fewer registers = more active threads at once).
"""

# ╔═╡ 8be67136-f565-4637-9209-3ce90934fd00
md"""
## Basic kernel operations

### Inputs/outputs
"""

# ╔═╡ fc3fc738-2ebc-49f3-93ab-050b56c9c7b2
md"""
GPU kernels cannot return values to the CPU like a regular function, so it must always be set to `return` or `return nothing`. So, to work with values, we can pass a `CuArray` aka writing our results to GPU arrays!

**Note:** Though a CuArray is given to the function when it is launched, the input is converted to a CuDeviceArray before execution.
"""

# ╔═╡ 80549d51-0948-49d6-8781-3c6368faa336
function log_kernel(input)
	data = input[1]
	input[1] = log(data)
	return nothing
end

# ╔═╡ 82c86b03-4378-4b8b-ac5d-843921fe6f37
a = CUDA.ones(Float32, 3)

# ╔═╡ 219230ef-5933-4a67-8db7-9e978483ed8b
@cuda log_kernel(a);

# ╔═╡ 096b5fbc-2ad3-41e1-a317-fbec7cef6555
a

# ╔═╡ 023bde30-4ac5-4b0c-9ee5-c55f9e401bd8
md"""
**FYI: if you need random numbers, you must use a GPU-compatiable RNG!**

Via: `@cushow rand()`
"""

# ╔═╡ b48a0560-692e-41ec-b36d-300a5e63ef97
md"""
With the example above, we took the log of only the first value in the array. 

**Note**: you usually want to only access/write into your global memory (`input`) once. Which is why we assign the variable `data`.
"""

# ╔═╡ 4b4a2b81-eef1-43e0-8388-17384b9de687
md"""
### Distributing across threads
Since we want to use multiple threads, we can use indexing to differentiate computations for each thread and block. Additionally, we can use `threads=` and `blocks=` when we launch `@cuda`.
![threadidx](https://developer-blogs.nvidia.com/wp-content/uploads/2017/01/Even-easier-intro-to-CUDA-image.png)
*https://developer.nvidia.com/blog/even-easier-introduction-cuda/*
"""

# ╔═╡ fed26c57-b123-4f90-acf6-2509f80ffbad
md"""
---
"""

# ╔═╡ df96c222-c395-4006-bc02-fefb8f8457bb
md"""
#### By threads
To process data that fits inside one block only, we can extract just the thread index using `threadIdx()`.
"""

# ╔═╡ fe781ab8-66ad-4bc9-aec3-92f701bf4e10
function thread_kernel(input)
	i = threadIdx().x
	j = threadIdx().y
	k = threadIdx().z

	x, y, z = size(input)
	if i <= x && j <= y && k <= z
		input[i, j, k] = i + j + k
	end
	return nothing
end

# ╔═╡ 8468f26f-dd40-4b10-8e90-b6f43b8ae3d1
b = CUDA.ones(Float32, (2, 2, 2))

# ╔═╡ 09f390a0-5ddb-4197-acbd-4a8bd2e4e418
@cuda thread_kernel(b);

# ╔═╡ 42f8eaca-9cf1-4759-823c-91be35172f18
b

# ╔═╡ e7d43a59-144b-4832-a080-3cda7ed14364
md"""
This is why we have to set the `threads=` argument!!! Without doing so, we only use default numbers; 1 block and 1 thread. Thus, the singular thread at (1, 1, 1) calculated 3.0.
"""

# ╔═╡ 0c193130-e7b8-49d9-b433-c72035a64d13
@cuda threads=size(b) thread_kernel(b);

# ╔═╡ 2a99ecea-2dbf-444c-88a6-9739b926ffcc
b

# ╔═╡ 7e8cbfb8-4c35-406b-93cd-ac34a8b5e22f
md"""
In specifying the threads, we get a cube of threads of size (2, 2, 2) so we get 8 threads total! However, we don't always set threads to the size of your data. This would be terrible as the maximum number of threads in a block is usually 1024, meaning that the amount of parallelism per block is limited. 
"""

# ╔═╡ fc49373a-4e6f-4a4c-984b-4286ccd7c0df
md"""
---
"""

# ╔═╡ 1fc6a81d-05af-46cb-a857-8bb06367ce49
md"""
**Ex. size limitation:**
"""

# ╔═╡ ede25d80-0342-45ee-81f1-80dc5c847aa2
big_3y = CUDA.ones(Float32, (2, 1025, 2))

# ╔═╡ 98a71850-e9e4-43d9-a0af-c20449530c58
@cuda threads=size(big_3y) thread_kernel(big_3y);

# ╔═╡ 0713aa0c-b124-4dda-a81f-4191da8d063c
md"""
---
"""

# ╔═╡ e681052e-e252-42c6-824a-615afaf97342
md"""
#### By blocks
To process data where one dimension (but not necessarily the other) fits inside one block only AND process rows and columns independently, we can assign each column to a block via `blockIdx()` and each row to a thread via `threadIdx()` (when looking at 2D inputs where n_rows <= 1024)!
"""

# ╔═╡ 8c25ebe0-80f2-4a51-b6fa-73cb7bfde088
function block_kernel(input)
    i = threadIdx().x
    j = blockIdx().x
	
    x, y = size(input)
	if i <= x && j <= y
		input[i, j] = i + j
	end
    return nothing
end

# ╔═╡ affb8d8f-4d94-46ca-adf3-4eaad33d3524
c = CUDA.ones(Float32, (2, 4))

# ╔═╡ 5e33bfd4-f891-4565-ab20-5a465d3d846e
@cuda threads=size(c, 1) blocks=size(c, 2) block_kernel(c);

# ╔═╡ e09f84fe-1824-4ff8-bdf3-1e55cf4cb96f
md"""
Here, we set blocks to the number of columns because we want to do column-wise calculations (like sample-wise!). Threads within a block can share memory, allowing values within a column (aka block) to work with one another.
"""

# ╔═╡ c9511d1a-e5a7-43c8-8bfa-6ba48c79a597
c

# ╔═╡ fd9312aa-9d93-45bc-ba49-f5a10aa5cba8
md"""
---
"""

# ╔═╡ 3f1c9e71-8052-4356-abd9-ba99229ce1ea
md"""
**Ex. size limitation:**
"""

# ╔═╡ 2c6c5385-ae13-4d7d-824c-4c19078a1e21
big_2x = CUDA.ones(Float32, (1025, 4))

# ╔═╡ add9d59d-ef03-4431-98c3-ca4c76ef6de0
@cuda threads=size(big_2x, 1) blocks=size(big_2x, 2) block_kernel(big_2x);

# ╔═╡ 3d49df46-852b-4c23-a17c-eae5ac6741b7
big_2y = CUDA.ones(Float32, (2, 1025))

# ╔═╡ a91e2c48-8a2b-4f29-8899-5cdc22dfca32
@cuda threads=size(big_2y, 1) blocks=size(big_2y, 2) block_kernel(big_2y);

# ╔═╡ df94bbe7-c8ce-4311-9829-f3452b86ce39
big_2y

# ╔═╡ 94edf947-5578-4716-96a2-513ebd934db7
md"""
---
"""

# ╔═╡ ea07a1b6-fb57-4f0d-a246-90b4e067721d
md"""
#### By index
Most commonly, we map our data into our grid by global indexing using `blockIdx()`, `blockDim()`, and `threadIdx()`. We assign each value a unique index, introducing the block dimensions so it knows to skip everything handled by previous blocks. In doing so, we are not limited by the dimension!
"""

# ╔═╡ a65f90c2-c7a8-49bc-ad8f-7d1a161a263d
function index_kernel(input)
	i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

	x, y = size(input)
	if i <= x && j <= y
		input[i, j] = i + j
	end
	return nothing
end

# ╔═╡ a7a62626-c414-4777-9f56-cdbf7913469d
d = CUDA.ones(Float32, (2, 2))

# ╔═╡ 94e4169b-ba67-46e6-86c5-d8f7dfd8c227
md"""
We calculate `cld()` to do ceiling division of the input dimension by the number of threads per block (32). Thus, we will have the minimum number of blocks needed to encapsulate all of the data.

This does mean that some threads will remain unused, i.e. when the total number of threads/block times the number of blocks is larger than the number of data points. Smart parameterization can help avoid this, but it's not always feasible to avoid it entirely. 
"""

# ╔═╡ de403ee8-b6e4-4bd1-b332-c4e92a481994
@cuda threads=(32, 32) blocks=(cld(size(d, 2), 32), cld(size(d, 2), 32)) index_kernel(d);

# ╔═╡ ad64c078-49bd-41d4-b6fa-415df11296cc
d

# ╔═╡ fbc39ba1-df25-4a3e-bd8a-79fa821db983
md"""
---
"""

# ╔═╡ e466469b-d704-445d-9017-672024d4393d
md"""
**Ex. no size limitation:**
"""

# ╔═╡ 09da573b-81da-498c-a42c-9e3d5a262eda
big_2xy = CUDA.ones(Float32, (1025, 1025))

# ╔═╡ e4f0c859-bb4a-4d9c-8789-ab336b831ac8
@cuda threads=(32, 32) blocks=(cld(size(big_2xy, 2), 32), cld(size(big_2xy, 2), 32)) index_kernel(big_2xy);

# ╔═╡ 3ab19663-2b92-4cb3-9664-0e23a59f432a
big_2xy

# ╔═╡ 915cf22c-5371-45fe-b833-c953a16ace06
md"""
## Calling functions
A thread can jump into helper functions as well! However, we must ensure that the helper function is specialized (ie. type stable) at compile time. This ensures that the GPU doesn't crash from having to figure out the types at runtime.
"""

# ╔═╡ 3e25a74b-f32c-4896-9c88-0b038cccb157
function calculations_unstable!(x, f)
	x .= f.(x)
end

# ╔═╡ 04541ad0-120a-488d-b31a-4381e43f5e89
function main_kernel_unstable(input, fxn)
	i = threadIdx().x
	calculations_unstable!(@view(input[i, :]), fxn)
	return nothing
end

# ╔═╡ cf3f80ec-8b34-463d-a20d-9cbab04a76b8
math_stuff(x) = x + 1

# ╔═╡ 87d33648-b45f-4209-9098-9d8b0ad70ca6
e = CUDA.ones(Float32, (2,2))

# ╔═╡ 1e0b00b7-5fc4-4e31-98b3-24df508886cc
@cuda threads=size(e, 1) main_kernel_unstable(e, math_stuff);

# ╔═╡ a76495a8-7927-4555-9919-6f4440fabe35
md"""
But if we redefine the `calculations()` function:
"""

# ╔═╡ 39f17b77-fa1a-4d3d-a73e-74ea03d15606
function calculations_stable!(x::X, f::F) where {X, F}
	x .= f.(x)
end

# ╔═╡ 13552532-1117-4bce-90a0-f3ae65060b1c
function main_kernel_stable(input, fxn)
	i = threadIdx().x
	calculations_stable!(@view(input[i, :]), fxn)
	return nothing
end

# ╔═╡ f89914d8-0ed8-44b7-9ddd-b3fbb9b952b6
@cuda threads=size(e, 1) main_kernel_stable(e, math_stuff);

# ╔═╡ d154b24d-7731-428f-b826-b60ad1540e93
e

# ╔═╡ 16dd03fe-61f7-475d-af43-3dd27cb821c3
md"""
## Synchronization
Sometimes, we need to sync threads within a block to ensure we don't overwrite data that another thread has worked on!
"""

# ╔═╡ 89ba0aef-0914-4249-8d8c-eed370e72a54
function sync_kernel(input)
	i = threadIdx().x
	j = blockDim().x
	data = input[i]
	sync_threads()
	input[j - i + 1] = data
	return nothing
end

# ╔═╡ bca69f3f-671a-4f67-a0c6-fc9962549ac7
f = CuArray([Vector(1:5) Vector(1:5) Vector(1:5)])

# ╔═╡ ed5e86f1-153e-4b49-9f19-dccab1b8eac0
@cuda threads=length(f) sync_kernel(f);

# ╔═╡ 5ea515ea-2cf2-4ae9-9f53-96178c2431e6
f

# ╔═╡ 9229435e-648f-4df3-a502-82dfc3d80b50
md"""
### Some additional uses of synchronization:


To verify or count conditions across threads (our predicates **pred**):
- `sync_threads_count(pred)`: returns the number of threads for which pred was true
- `sync_threads_and(pred)`: returns true if pred was true for all threads
- `sync_threads_or(pred)`: returns true if pred was true for any thread

To maintain multiple thread synchronizations via execution (i.e. have different sync_threads in different situations), we can use:
- `barrier_sync()`

To maintain multiple thread synchronizations via memory (e.g. to make sure parts of the memory are visible to other threads at the correct moment), we can use:
- `threadfence_block`: ensure memory ordering for all threads in the block
- `threadfence`: the same, but for all threads on the device
- `threadfence_system`: the same, but including host threads and threads on peer devices
"""

# ╔═╡ 8264c2e0-f916-4687-ad76-563a2612e831
md"""
## Shared memory
To communicate between threads, we can utilize static and dynamic shared arrays.
"""

# ╔═╡ 2ee211f9-02ee-41d6-ab45-d65121dcdeb6
md"""
### Static
For when we know the amount of shared memory beforehand.
"""

# ╔═╡ 54b2df6f-9d01-4e9c-b6a6-020fe8198237
function static_kernel(input::CuDeviceArray{T}) where T
	i = threadIdx().x # row
	j = blockIdx().x # col
	index = (j - 1) * 5 + i

	data = CuStaticSharedArray(T, 5)
	@inbounds begin
		data[5 - i + 1] = input[index]
		sync_threads()
		input[index] = data[i]
	end
	return nothing
end

# ╔═╡ 8adf51dd-e0b7-4d54-9166-5cb99e8a8ceb
md"""
Note: we can do `@inbounds` here to indicate when we know 100% that the index is in bounds. Otherwise, indexing a CuArray will do bounds checking by defualt and throwing the error can be very costly!
"""

# ╔═╡ 343345ab-b631-4c1d-bb9c-2b6fb6eda6c8
g = CuArray([Vector(1:5) Vector(1:5) Vector(1:5)])

# ╔═╡ b498f690-2a7d-4210-abfc-0ea485181344
@cuda threads=5 blocks=3 static_kernel(g);

# ╔═╡ 47831da4-44e9-4d7e-b436-a84450ee9111
g

# ╔═╡ 59363ff2-195d-4357-b8e0-bcdc678c95b5
md"""
Thus, different threads are able to modify the same array without interfering with one another.
"""

# ╔═╡ 130ed5ae-1655-4f4f-8fcc-58aa1f2578ea
md"""
### Dynamic
For when we don't know the amount of shared memory beforehand. 

Here, we pass the size of the shared memory (`shmem=`) **in bytes** as an argument to the kernel.
"""

# ╔═╡ adfebfac-589e-4eea-b9f7-44106c86e161
function dynamic_kernel(input::CuDeviceArray{T}) where T
	i = threadIdx().x # row
	j = blockIdx().x # col
	n = blockDim().x # n_rows
	index = (j - 1) * n + i

	data = CuDynamicSharedArray(T, n)
	@inbounds begin
		data[n - i + 1] = input[index]
		sync_threads()
		input[index] = data[i]
	end
	return nothing
end

# ╔═╡ 63b354e9-b035-46df-914f-436c64392a17
h = CuArray([Vector(1:5) Vector(1:5) Vector(1:5)])

# ╔═╡ f7f4f2a1-b90c-4ad2-8221-19e612e4dea7
@cuda threads=size(h, 1) blocks=size(h, 2) shmem=sizeof(h[:, 1]) dynamic_kernel(h);

# ╔═╡ d97a50c0-2eae-41b4-8737-054391bee8a6
md"""
Because we set the shared memory to the size of one column in h, we will share one column across all the threads in a block so that the threads do not interfere with each other.
"""

# ╔═╡ 56ec0f87-7455-40af-87e1-ee138804ced7
h

# ╔═╡ f5993620-a538-4cea-b182-1f702966db75
md"""
We can also introduce the parameter `offset`: the offset in  bytes from the start of the shared memory.
"""

# ╔═╡ 4ca8a843-f128-47e8-addc-c6c71c87b153
function dynamic_kernel_multi(input::CuDeviceArray{T}) where T
	i = threadIdx().x # row
	j = blockIdx().x # col
	n = blockDim().x # n_rows
	index = (j - 1) * n + i

	data = CuDynamicSharedArray(T, n)
	data2 = CuDynamicSharedArray(T, n, sizeof(data))

	@inbounds begin
		data[n - i + 1] = input[index]
		data2[n - i + 1] = input[index]
		sync_threads()
		input[index] = data[i]+data[i]
	end
	return nothing
end

# ╔═╡ 8a7d56a0-691e-4f87-a1d8-5099e260b363
h2 = CuArray([Vector(1:5) Vector(1:5) Vector(1:5)])

# ╔═╡ 18e3573d-dc9f-4549-b2b1-7a07e75776fa
@cuda threads=size(h2, 1) blocks=size(h2, 2) shmem=sizeof(h2[:, 1])*2 dynamic_kernel_multi(h2);

# ╔═╡ fc67e163-e8a7-4a7b-ad1b-bd4749f1ee2b
h2

# ╔═╡ 3d00f0f7-1a6a-4f01-a304-a57cb944c0ca
md"""
## Atomic operations
These are operations that execute read/modify/write in one step, such that when  working with shared memory, there are no interruptions. Essentially locks a piece of data while it's being operated on so nothing else can touch it!
"""

# ╔═╡ 3488c501-c9fa-4f2f-adae-1211e7b2b5f8
md"""
### Low-level
These take pointer inputs (via `pointer(CuArray)`). Some supported operations are:
- binary operations: `add, sub, or, xor, min, max, xchg, inc, dec`
- compare-and-swap: `cas`
"""

# ╔═╡ 31f51b5a-c3be-40be-a58c-144a8ea96def
function low_atomic_kernel(input)
	CUDA.atomic_add!(pointer(input), Float32(1))
	return nothing
end

# ╔═╡ 4db5c967-c09a-44c5-9b9c-b32e2491897d
o = CUDA.ones(Float32, 2, 2)

# ╔═╡ 1ab789a8-5a9b-444a-814d-013a4d7963fd
@cuda threads=size(o) low_atomic_kernel(o);

# ╔═╡ 1122c1ca-80ff-46e8-96aa-182859fd1eb9
o

# ╔═╡ 9506bda1-1871-416e-88f3-a9c3232ad0e0
md"""
Without using the atomics, all threads would have read the initial value (1.0) then simultaneously overwritten one another, which would have resulted in 2.0. We can see from the result that using `CUDA.atomic_add!()`, the threads are forced to essentially take turns and properly accumulate the sums to 5.0.

**Note:** Atomics other than CUDA.atomic_add!() are not supported for float values. 
"""

# ╔═╡ 2058c67c-d4cd-4140-9791-bb40dff0dbdb
md"""
### High-level
We can also use the `CUDA.@atomic` macro. This will automatically convert inputs to the appropriate type and other fallbacks but may have issues with the `Base.@atomic` macro...
"""

# ╔═╡ 027098a3-5dfe-46e2-b5ae-0f9772021428
function high_atomic_kernel(input)
	CUDA.@atomic input[1, 1] += 1
	return nothing
end

# ╔═╡ a1fef26f-e9a1-4e3c-b7c2-20586acdea70
p = CUDA.ones(Float32, 2, 2)

# ╔═╡ 03db2701-b097-474f-a0b6-7c040e9f19bf
@cuda threads=(2,2) high_atomic_kernel(p);

# ╔═╡ 720c6916-6afc-44da-86ab-2ba4d37615fd
p

# ╔═╡ 6cb978c6-8c25-46be-bc0f-3b17ffacab2f
md"""
Basically the exact same thing as the low-level but we can just write our code as usual instead of pointer-ing and type-ing, etc.
"""

# ╔═╡ 486919a0-f5c3-4d33-a3ea-770feb9bc01b
md"""
## Dynamic parallelism
For things like recursive functions, we can utilize dynamic parallelism. This essentially spawns a new grid of threads by launching an additional kernel through `@cuda ...` (kernel from inside a kernel)!
"""

# ╔═╡ 6ec3aee7-32d3-48d1-ab28-174dc0b68ed7
function update_kernel(input, i, j)
	input[i, j] -= 0.1
	return nothing
end

# ╔═╡ c07da5ff-0e5a-41de-92fa-749f6ac0d3a9
function check_kernel(input)
	i = threadIdx().x
	j = threadIdx().y
	while input[i, j] > 0.5
		@cuda dynamic=true threads=1 update_kernel(input, i, j)
		device_synchronize()
	end
	return nothing
end

# ╔═╡ 462e3e46-aa4b-4e15-b7cc-62f49a96edc3
q = CUDA.rand(Float32, 2, 2)

# ╔═╡ cb439496-9bd6-45b6-bb92-f067834d9e2f
@cuda threads=(2, 2) check_kernel(q);

# ╔═╡ 48218b16-bd2d-49f7-9e11-c0c68610db9e
q

# ╔═╡ 4318b91e-6a42-4de9-859a-93fc1fe14aaf
md"""
### Why `device_synchronize()`?
"""

# ╔═╡ a3ae6716-37cc-4b7d-94cf-0e2f80aa7b13
md"""
![parallel](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/parent-child-launch-nesting.png)

*https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html*
"""

# ╔═╡ 26b402d2-74bc-474b-a540-4df6505224ab
md"""
Essentially when we launch our parent grid (`check_kernel`), the second `@cuda` call will launch a child grid (`update_kernel`). Since they are asynchronous, the parent kernel will continue executing without waiting for the child to finish. 
"""

# ╔═╡ b19681d5-a220-4029-8729-37878a982f53
function check_kernel_bad(input)
	i = threadIdx().x
	j = threadIdx().y
	while input[i, j] > 0.5
		@cuda dynamic=true threads=1 update_kernel(input, i, j)
		# device_synchronize()
	end
	return nothing
end

# ╔═╡ 5134deb6-a93c-4c55-a26a-6f3683b5edbf
r = CUDA.rand(Float32, 2, 2) .+ 15

# ╔═╡ 79a8e7f4-7137-4cb2-9e80-3bcc9017add7
@cuda threads=(2, 2) check_kernel_bad(r);

# ╔═╡ b9ed8772-1564-47f1-ad18-4704a7b78071
r

# ╔═╡ 784d3995-879f-4501-bdfc-657bd309df63
md"""
This results in decrementing the value way too far because additional updates have been launched before the first child has finished modifying the value. Adding `device_synchronize` allows the parent to first wait for the child to fully complete its work before continuing with an additional while loop.
"""

# ╔═╡ aa08d3f8-c486-4dcc-bcd4-9c59ad97bb99
md"""
# Demo time! :D
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
Pkg = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[compat]
CUDA = "~5.9.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.7"
manifest_format = "2.0"
project_hash = "3fc610423ed3527af127b441755860ab16ce8f53"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

    [deps.AbstractFFTs.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "7e35fca2bdfba44d797c53dfe63a51fabf39bfc0"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.4.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "29bb0eb6f578a587a49da16564705968667f5fa8"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.1.2"

    [deps.Atomix.extensions]
    AtomixCUDAExt = "CUDA"
    AtomixMetalExt = "Metal"
    AtomixOpenCLExt = "OpenCL"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "0a6d6d072cb5f2baeba7667023075801f6ea4a7d"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.6.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Compiler_jll", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "Crayons", "DataFrames", "ExprTools", "GPUArrays", "GPUCompiler", "GPUToolbox", "KernelAbstractions", "LLVM", "LLVMLoopInfo", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "NVTX", "Preferences", "PrettyTables", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "StaticArrays", "Statistics", "demumble_jll"]
git-tree-sha1 = "756f031a1ef3137f497ee73ed595e4acf65d753f"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "5.9.3"

    [deps.CUDA.extensions]
    ChainRulesCoreExt = "ChainRulesCore"
    EnzymeCoreExt = "EnzymeCore"
    SparseMatricesCSRExt = "SparseMatricesCSR"
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.CUDA.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    SparseMatricesCSR = "a0a7dd2c-ebf4-11e9-1f05-cf50bc540ca1"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.CUDA_Compiler_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "CUDA_Runtime_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "b63428872a0f60d87832f5899369837cd930b76d"
uuid = "d1e2174e-dfdc-576e-b43e-73b79eb1aca8"
version = "0.3.0+0"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "2023be0b10c56d259ea84a94dbfc021aa452f2c6"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "13.0.2+0"

[[deps.CUDA_Runtime_Discovery]]
deps = ["Libdl"]
git-tree-sha1 = "f9a521f52d236fe49f1028d69e549e7f2644bb72"
uuid = "1af6417a-86b4-443c-805f-a4643ffb695f"
version = "1.0.0"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "92cd84e2b760e471d647153ea5efc5789fc5e8b2"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.19.2+0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d8928e9169ff76c6281f39a659f9bca3a573f24c"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.1"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e357641bb3e0638d353c4b29ea0e40ea644066a6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.3"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "ScopedValues", "Serialization", "Statistics"]
git-tree-sha1 = "8ddb438e956891a63a5367d7fab61550fc720026"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "11.2.6"

    [deps.GPUArrays.extensions]
    JLD2Ext = "JLD2"

    [deps.GPUArrays.weakdeps]
    JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "PrecompileTools", "Preferences", "Scratch", "Serialization", "TOML", "Tracy", "UUIDs"]
git-tree-sha1 = "c55c2f564230f1af2a975d927a814bcb47a63a50"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "1.7.3"

[[deps.GPUToolbox]]
deps = ["LLVM"]
git-tree-sha1 = "9e9186b09a13b7f094f87d1a9bb266d8780e1b1c"
uuid = "096a3bc2-3ced-46d0-87f4-dd12716f4bfc"
version = "1.0.0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.InlineStrings]]
git-tree-sha1 = "8f3d257792a522b4601c24a577954b0a8cd7334d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.5"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JuliaNVTXCallbacks_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "af433a10f3942e882d3c671aacb203e006a5808f"
uuid = "9c1d0b0a-7046-5b2e-a33f-ea22f176ac7e"
version = "0.2.1+0"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "b5a371fcd1d989d844a4354127365611ae1e305f"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.39"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "ce8614210409eaa54ed5968f4b50aa96da7ae543"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.4.4"
weakdeps = ["BFloat16s"]

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "8e76807afb59ebb833e9b131ebf1a8c006510f33"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.38+0"

[[deps.LLVMLoopInfo]]
git-tree-sha1 = "2e5c102cfc41f48ae4740c7eca7743cc7e7b75ea"
uuid = "8b046642-f1f6-4319-8d3c-209ddc03c586"
version = "1.0.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.LibTracyClient_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d2bc4e1034b2d43076b50f0e34ea094c2cb0a717"
uuid = "ad6e5548-8b26-5c9f-8ef3-ef0ad883f3a5"
version = "0.9.1+6"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NVTX]]
deps = ["Colors", "JuliaNVTXCallbacks_jll", "Libdl", "NVTX_jll"]
git-tree-sha1 = "6b573a3e66decc7fc747afd1edbf083ff78c813a"
uuid = "5da4648a-3479-48b8-97b9-01cb529c0a1f"
version = "1.0.1"

[[deps.NVTX_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "af2232f69447494514c25742ba1503ec7e9877fe"
uuid = "e98f9f5b-d649-5603-91fd-7774390e6439"
version = "3.2.2+0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "522f093a29b31a93e34eaea17ba055d850edea28"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.1"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "6b8e2f0bae3f678811678065c09571c1619da219"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.1.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "dbe5fd0b334694e905cb9fda73cd8554333c46e2"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.7.1"

[[deps.RandomNumbers]]
deps = ["Random"]
git-tree-sha1 = "c6ec94d2aaba1ab2ff983052cf6a606ca5985902"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.6.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "c3b2323466378a2ba15bea4b2f73b081e022f473"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.5.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "b8693004b385c842357406e3af647701fe783f98"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.15"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "725421ae8e530ec29bcbdddbe91ff8053421d023"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.1"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Tracy]]
deps = ["ExprTools", "LibTracyClient_jll", "Libdl"]
git-tree-sha1 = "73e3ff50fd3990874c59fef0f35d10644a1487bc"
uuid = "e689c965-62c8-4b79-b2c5-8359227902fd"
version = "0.1.6"

    [deps.Tracy.extensions]
    TracyProfilerExt = "TracyProfiler_jll"

    [deps.Tracy.weakdeps]
    TracyProfiler_jll = "0c351ed6-8a68-550e-8b79-de6f926da83c"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "b13c4edda90890e5b04ba24e20a310fbe6f249ff"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.3.0"
weakdeps = ["LLVM"]

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsLLVM = ["LLVM"]

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.demumble_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6498e3581023f8e530f34760d18f75a69e3a4ea8"
uuid = "1e29f10c-031c-5a83-9565-69cddfc27673"
version = "1.3.0+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╟─69fc927d-adee-4c6d-aff6-a9a35350070d
# ╠═7a24781f-c7d5-4b17-85d8-f040eae59fa5
# ╠═2d61a06c-1d62-4979-b92f-784a92741ee6
# ╟─cdc41743-80d0-449c-a43f-a4685a8d75e3
# ╟─afa298c8-f482-40ca-a73d-80ffab61c8e7
# ╟─2627ab83-84ed-4ed3-9dfc-95a1bdb96bcf
# ╟─1d38f704-0c74-41e0-ad24-e37e7a8dd66c
# ╟─b496b692-d39b-4f25-ae71-f0a697681175
# ╠═4a301a05-13c4-4561-be78-a766f05670de
# ╟─a57c01b2-1c34-4c83-bda7-a7b1a4599bab
# ╠═003a2324-37c3-487c-967b-733ea815ccd6
# ╟─6fb64c82-60e3-43ec-bf64-96acc016867c
# ╠═4f264902-ad19-4139-bda8-20645fcca984
# ╠═5429e43c-a42b-4257-83be-02b6bdb99155
# ╟─18dcd360-505c-4321-b11b-cd55c63e808e
# ╟─8be67136-f565-4637-9209-3ce90934fd00
# ╟─fc3fc738-2ebc-49f3-93ab-050b56c9c7b2
# ╠═80549d51-0948-49d6-8781-3c6368faa336
# ╠═82c86b03-4378-4b8b-ac5d-843921fe6f37
# ╠═219230ef-5933-4a67-8db7-9e978483ed8b
# ╠═096b5fbc-2ad3-41e1-a317-fbec7cef6555
# ╟─023bde30-4ac5-4b0c-9ee5-c55f9e401bd8
# ╟─b48a0560-692e-41ec-b36d-300a5e63ef97
# ╟─4b4a2b81-eef1-43e0-8388-17384b9de687
# ╟─fed26c57-b123-4f90-acf6-2509f80ffbad
# ╟─df96c222-c395-4006-bc02-fefb8f8457bb
# ╠═fe781ab8-66ad-4bc9-aec3-92f701bf4e10
# ╠═8468f26f-dd40-4b10-8e90-b6f43b8ae3d1
# ╠═09f390a0-5ddb-4197-acbd-4a8bd2e4e418
# ╠═42f8eaca-9cf1-4759-823c-91be35172f18
# ╟─e7d43a59-144b-4832-a080-3cda7ed14364
# ╠═0c193130-e7b8-49d9-b433-c72035a64d13
# ╠═2a99ecea-2dbf-444c-88a6-9739b926ffcc
# ╟─7e8cbfb8-4c35-406b-93cd-ac34a8b5e22f
# ╟─fc49373a-4e6f-4a4c-984b-4286ccd7c0df
# ╟─1fc6a81d-05af-46cb-a857-8bb06367ce49
# ╠═ede25d80-0342-45ee-81f1-80dc5c847aa2
# ╠═98a71850-e9e4-43d9-a0af-c20449530c58
# ╟─0713aa0c-b124-4dda-a81f-4191da8d063c
# ╟─e681052e-e252-42c6-824a-615afaf97342
# ╠═8c25ebe0-80f2-4a51-b6fa-73cb7bfde088
# ╠═affb8d8f-4d94-46ca-adf3-4eaad33d3524
# ╠═5e33bfd4-f891-4565-ab20-5a465d3d846e
# ╟─e09f84fe-1824-4ff8-bdf3-1e55cf4cb96f
# ╠═c9511d1a-e5a7-43c8-8bfa-6ba48c79a597
# ╟─fd9312aa-9d93-45bc-ba49-f5a10aa5cba8
# ╟─3f1c9e71-8052-4356-abd9-ba99229ce1ea
# ╠═2c6c5385-ae13-4d7d-824c-4c19078a1e21
# ╠═add9d59d-ef03-4431-98c3-ca4c76ef6de0
# ╠═3d49df46-852b-4c23-a17c-eae5ac6741b7
# ╠═a91e2c48-8a2b-4f29-8899-5cdc22dfca32
# ╠═df94bbe7-c8ce-4311-9829-f3452b86ce39
# ╟─94edf947-5578-4716-96a2-513ebd934db7
# ╟─ea07a1b6-fb57-4f0d-a246-90b4e067721d
# ╠═a65f90c2-c7a8-49bc-ad8f-7d1a161a263d
# ╠═a7a62626-c414-4777-9f56-cdbf7913469d
# ╟─94e4169b-ba67-46e6-86c5-d8f7dfd8c227
# ╠═de403ee8-b6e4-4bd1-b332-c4e92a481994
# ╠═ad64c078-49bd-41d4-b6fa-415df11296cc
# ╟─fbc39ba1-df25-4a3e-bd8a-79fa821db983
# ╟─e466469b-d704-445d-9017-672024d4393d
# ╠═09da573b-81da-498c-a42c-9e3d5a262eda
# ╠═e4f0c859-bb4a-4d9c-8789-ab336b831ac8
# ╠═3ab19663-2b92-4cb3-9664-0e23a59f432a
# ╟─915cf22c-5371-45fe-b833-c953a16ace06
# ╠═3e25a74b-f32c-4896-9c88-0b038cccb157
# ╠═04541ad0-120a-488d-b31a-4381e43f5e89
# ╠═cf3f80ec-8b34-463d-a20d-9cbab04a76b8
# ╠═87d33648-b45f-4209-9098-9d8b0ad70ca6
# ╠═1e0b00b7-5fc4-4e31-98b3-24df508886cc
# ╟─a76495a8-7927-4555-9919-6f4440fabe35
# ╠═39f17b77-fa1a-4d3d-a73e-74ea03d15606
# ╠═13552532-1117-4bce-90a0-f3ae65060b1c
# ╠═f89914d8-0ed8-44b7-9ddd-b3fbb9b952b6
# ╠═d154b24d-7731-428f-b826-b60ad1540e93
# ╟─16dd03fe-61f7-475d-af43-3dd27cb821c3
# ╠═89ba0aef-0914-4249-8d8c-eed370e72a54
# ╠═bca69f3f-671a-4f67-a0c6-fc9962549ac7
# ╠═ed5e86f1-153e-4b49-9f19-dccab1b8eac0
# ╠═5ea515ea-2cf2-4ae9-9f53-96178c2431e6
# ╟─9229435e-648f-4df3-a502-82dfc3d80b50
# ╟─8264c2e0-f916-4687-ad76-563a2612e831
# ╟─2ee211f9-02ee-41d6-ab45-d65121dcdeb6
# ╠═54b2df6f-9d01-4e9c-b6a6-020fe8198237
# ╟─8adf51dd-e0b7-4d54-9166-5cb99e8a8ceb
# ╠═343345ab-b631-4c1d-bb9c-2b6fb6eda6c8
# ╠═b498f690-2a7d-4210-abfc-0ea485181344
# ╠═47831da4-44e9-4d7e-b436-a84450ee9111
# ╟─59363ff2-195d-4357-b8e0-bcdc678c95b5
# ╟─130ed5ae-1655-4f4f-8fcc-58aa1f2578ea
# ╠═adfebfac-589e-4eea-b9f7-44106c86e161
# ╠═63b354e9-b035-46df-914f-436c64392a17
# ╠═f7f4f2a1-b90c-4ad2-8221-19e612e4dea7
# ╟─d97a50c0-2eae-41b4-8737-054391bee8a6
# ╠═56ec0f87-7455-40af-87e1-ee138804ced7
# ╟─f5993620-a538-4cea-b182-1f702966db75
# ╠═4ca8a843-f128-47e8-addc-c6c71c87b153
# ╠═8a7d56a0-691e-4f87-a1d8-5099e260b363
# ╠═18e3573d-dc9f-4549-b2b1-7a07e75776fa
# ╠═fc67e163-e8a7-4a7b-ad1b-bd4749f1ee2b
# ╟─3d00f0f7-1a6a-4f01-a304-a57cb944c0ca
# ╟─3488c501-c9fa-4f2f-adae-1211e7b2b5f8
# ╠═31f51b5a-c3be-40be-a58c-144a8ea96def
# ╠═4db5c967-c09a-44c5-9b9c-b32e2491897d
# ╠═1ab789a8-5a9b-444a-814d-013a4d7963fd
# ╠═1122c1ca-80ff-46e8-96aa-182859fd1eb9
# ╟─9506bda1-1871-416e-88f3-a9c3232ad0e0
# ╟─2058c67c-d4cd-4140-9791-bb40dff0dbdb
# ╠═027098a3-5dfe-46e2-b5ae-0f9772021428
# ╠═a1fef26f-e9a1-4e3c-b7c2-20586acdea70
# ╠═03db2701-b097-474f-a0b6-7c040e9f19bf
# ╠═720c6916-6afc-44da-86ab-2ba4d37615fd
# ╟─6cb978c6-8c25-46be-bc0f-3b17ffacab2f
# ╟─486919a0-f5c3-4d33-a3ea-770feb9bc01b
# ╠═6ec3aee7-32d3-48d1-ab28-174dc0b68ed7
# ╠═c07da5ff-0e5a-41de-92fa-749f6ac0d3a9
# ╠═462e3e46-aa4b-4e15-b7cc-62f49a96edc3
# ╠═cb439496-9bd6-45b6-bb92-f067834d9e2f
# ╠═48218b16-bd2d-49f7-9e11-c0c68610db9e
# ╟─4318b91e-6a42-4de9-859a-93fc1fe14aaf
# ╟─a3ae6716-37cc-4b7d-94cf-0e2f80aa7b13
# ╟─26b402d2-74bc-474b-a540-4df6505224ab
# ╠═b19681d5-a220-4029-8729-37878a982f53
# ╠═5134deb6-a93c-4c55-a26a-6f3683b5edbf
# ╠═79a8e7f4-7137-4cb2-9e80-3bcc9017add7
# ╠═b9ed8772-1564-47f1-ad18-4704a7b78071
# ╟─784d3995-879f-4501-bdfc-657bd309df63
# ╟─aa08d3f8-c486-4dcc-bcd4-9c59ad97bb99
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
