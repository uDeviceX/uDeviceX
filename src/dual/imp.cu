#include <stdio.h>
#include <conf.h>

#include "inc/conf.h"
#include "msg.h"
#include "cc.h"
#include "inc/dev.h"

#include "dual/type.h"
#include "dual/int.h"

namespace dual {
void alloc(I *p, int n) {
    CC(cudaHostAlloc(&p->D, sizeof(int) * n, cudaHostAllocMapped));
    CC(cudaHostGetDevicePointer(&p->DP, p->D, 0));
}

void dealloc(I p) {
    int *D;
    D = p.D;
    CC(cudaFreeHost(D));
}
}
