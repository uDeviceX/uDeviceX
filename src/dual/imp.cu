#include <stdio.h>
#include <conf.h>

#include "inc/conf.h"
#include "msg.h"
#include "cc.h"

#include "dual/type.h"
#include "dual/int.h"

namespace dual {
void alloc(I p, int n) {
    int *D, *DP;
    D = p.D; DP = p.DP;
    CC(cudaHostAlloc(&D, sizeof(int) * n, cudaHostAllocMapped));
    CC(cudaHostGetDevicePointer(&DP, D, 0));
}

void dealloc(I p) {
    int *D;
    D = p.D;
    CC(cudaFreeHost(D));
}
}
