#ifndef INDEX_CUH
#define INDEX_CUH

#define TID_X threadIdx.x
#define TID_Y threadIdx.y
#define TID_Z threadIdx.z

#define BID_X blockIdx.x
#define BID_Y blockIdx.y
#define BID_Z blockIdx.z

#define GDIM_X gridDim.x
#define GDIM_Y gridDim.y
#define GDIM_Z gridDim.z

#define BDIM_X blockDim.x
#define BDIM_Y blockDim.y
#define BDIM_Z blockDim.z

#define TID_IN_BLOCK (TID_Z * (BDIM_Y * BDIM_X) + TID_Y * BDIM_X + TID_X)
#define NUM_THREADS_IN_BLOCK (BDIM_X * BDIM_Y * BDIM_Z)

#define BLOCK_ID_LINEAR (BID_X + GDIM_X * (BID_Y + GDIM_Y * BID_Z))
#define GLOBAL_TID (TID_IN_BLOCK + (BLOCK_ID_LINEAR * NUM_THREADS_IN_BLOCK))

#endif // INDEX_CUH
