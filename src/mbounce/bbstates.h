namespace mbounce {
namespace sub {

#define make_enum(a) a ,
#define make_str(a) #a ,

#define bbstates(_)                                                 \
    _(BB_SUCCESS)   /* succesfully bounced                       */ \
    _(BB_NOCROSS)   /* did not cross the plane                   */ \
    _(BB_WTRIANGLE) /* [w]rong triangle                          */ \
    _(BB_HFAIL)     /* no time solution h                        */ \
    _(BB_HNEXT)     /* h triangle is not the first to be crossed */
    
enum BBState {
    bbstates(make_enum)
    NBBSTATES
};

namespace dbg {

#ifdef debug_output
static const char *bbstatenames[] = {bbstates(make_str)};

#define print_states(infos) do {                            \
        for (int c = 0; c < NBBSTATES; ++c)                 \
        printf("%-12s\t%d\n", bbstatenames[c], infos[c]);   \
    } while(0)

int bbstates_hst[NBBSTATES], dstep = 0;
__device__ int bbstates_dev[NBBSTATES];

__device__ __host__ void log_states(BBState s) {
#if DEVICE_FUNC
    atomicAdd(bbstates_dev + s, 1);
#else
    bbstates_hst[s] ++;
#endif
}

void ini_hst() {
    if (dstep % part_freq == 0)
        for (int c = 0; c < NBBSTATES; ++c) bbstates_hst[c] = 0;
}

void report_hst() {
    if ((++dstep) % part_freq == 0)
        print_states(bbstates_hst);
}

void ini_dev() {
    if (dstep % part_freq == 0) {
        const int zeros[NBBSTATES] = {0};
        CC(cudaMemcpyToSymbol(bbstates_dev, zeros, NBBSTATES*sizeof(int)));
    }
}

void report_dev() {
    if ((++dstep) % part_freq == 0) {
        int bbinfos[NBBSTATES];
        CC(cudaMemcpyFromSymbol(bbinfos, bbstates_dev, NBBSTATES*sizeof(int)));
        print_states(bbinfos);
    }
}

#else // debug_output

__device__ __host__ void log_states(BBState s) {}

void ini_hst() {}
void ini_dev() {}
void report_hst() {}
void report_dev() {}

#endif // debug_output

} // dbg

#undef bbstates
#undef make_enum
#undef make_str

} // sub
} // mbounce
