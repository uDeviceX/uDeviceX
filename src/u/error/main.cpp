#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include <conf.h>
#include "inc/conf.h"

#include "mpi/glb.h"
#include "mpi/wrapper.h"

#include "d/api.h"

#include "conf/imp.h"

#include "utils/msg.h"
#include "utils/imp.h"
#include "utils/mc.h"
#include "utils/cc.h"
#include "utils/error.h"

enum {UDX_, MPI_, CUDA_, MAX_STACK_, NKINDS};

void MAX_STACK_bar() {
    int j = 0;
    for (int i = 0; i < 130; ++i) {
        error_stack_push(__FILE__, __LINE__);
        j ++;
    }
    printf("%d\n", j);
}

void UDX_bar() {
    ERR("udx bar failed");
}

void MPI_bar() {
    int c;
    MPI_Status *wrong_status = NULL;
    MC(m::Get_count(wrong_status, MPI_CHAR, &c));
}

void CUDA_bar() {
    int *wrong_pointer = NULL;
    CC(d::Memset(wrong_pointer, 0, 32));
}

void foo(int kind) {
    switch(kind) {
    case UDX_:
        UC(UDX_bar());
        break;
    case MPI_:
        UC(MPI_bar());
        break;
    case CUDA_:
        UC(CUDA_bar());
        break;
    case MAX_STACK_:
        UC(MAX_STACK_bar());
        break;
    default:
        break;
    };
}

int parse_kind(const Config *c) {
    const char *kind;
    int k = -1;
    conf_lookup_string(c, "err_kind", &kind);

    if      (same_str(kind, "udx"))      k = UDX_;
    else if (same_str(kind, "mpi"))      k = MPI_;
    else if (same_str(kind, "cuda"))     k = CUDA_;
    else if (same_str(kind, "maxstack")) k = MAX_STACK_;
    else ERR("unrecognised error kind <%s>", kind);

    return k;
}

int main(int argc, char **argv) {
    m::ini(&argc, &argv);
    Config *cfg;
    int k;
    conf_ini(&cfg);
    conf_read(argc, argv, cfg);

    k = parse_kind(cfg);
    
    UC(foo(k));

    conf_fin(cfg);
    m::fin();
}
