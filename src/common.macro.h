
enum {  /* used in sorting of bulk particle when wall is created */
W_BULK,  /* remains in bulk */
W_WALL,  /* becomes wall particle */
W_DEEP   /* deep inside the wall */
};

#define allsync() do { CC(cudaDeviceSynchronize()); MC(MPI_Barrier(m::cart)); } while (0)
//#define allsync() do { CC(cudaDeviceSynchronize()); } while (0)
