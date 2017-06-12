
enum {  /* used in sorting of bulk particle when wall is created */
W_BULK,  /* remains in bulk */
W_WALL,  /* becomes wall particle */
W_DEEP   /* deep inside the wall */
};
/*
#define allsync() do {                                                  \
        CC(cudaDeviceSynchronize()); MC(MPI_Barrier(m::cart));          \
        if (m::rank == 0)                                               \
        fprintf(stderr, "%s : %d\n", __FILE__, __LINE__);               \
    } while (0)
