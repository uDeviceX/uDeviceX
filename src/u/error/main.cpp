#include <stdio.h>
#include <stdlib.h>

#include "msg.h"
#include "mpi/glb.h"
#include "utils/error.h"

enum {UDX, MPI, CUDA, NKINDS};

void UDX_bar() {
    signal_error_extra("udx bar failed");
}

void MPI_bar() {
    signal_error_extra("mpi bar failed");
}

void CUDA_bar() {
    signal_error_extra("cuda bar failed");
}

void foo(int kind) {
    switch(kind) {
    case UDX:
        UC(UDX_bar());
        break;
    case MPI:
        UC(MPI_bar());
        break;
    case CUDA:
        UC(CUDA_bar());
        break;
    default:
        break;
    };
}

int main(int argc, char **argv) {
    m::ini(argc, argv);

    const char *ckind = getenv("ERR_KIND");
    int k = atoi(ckind);
    printf("%s %d\n", ckind, k);
    
    UC(foo(k));
    m::fin();
}
