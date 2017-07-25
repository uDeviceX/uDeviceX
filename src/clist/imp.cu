#include <cstdio>
#include "common.h"
#include "common.cuda.h"
#include "clist/int.h"

namespace clist {
namespace dev {
#include "clist/dev.h"
}
void build(const Particle *pp, int n, /**/ int *start, int *count) {
    int *ids;
    CC(cudaMalloc(&ids, n*sizeof(int)));
    
    CC(cudaFree(ids));
}
}
