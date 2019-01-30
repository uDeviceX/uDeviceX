#include <stdio.h>
#include <float.h>
#include <mpi.h>

#include "utils/mc.h"
#include "utils/msg.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "utils/mc.h"

#include "algo/kahan_sum/imp.h"

void main0() {
    double input, sum, sum0;
    KahanSum *kahan_sum;
    kahan_sum_ini(&kahan_sum);
    sum0 = 0;
    while (scanf("%lf", &input) == 1) {
        kahan_sum_add(kahan_sum, input);
        sum0 += input;
    }
    sum = kahan_sum_get(kahan_sum);
    printf("%.17e   %.17e\n",sum, sum0);
    kahan_sum_fin(kahan_sum);
}

int main(int argc, char **argv) {
    int rank, size, dims[3];
    MPI_Comm cart;
    m::ini(&argc, &argv);
    m::get_dims(&argc, &argv, dims);
    m::get_cart(MPI_COMM_WORLD, dims, &cart);
    main0();

    MC(m::Comm_rank(cart, &rank));
    MC(m::Comm_size(cart, &size));
    msg_ini(rank);
    MC(m::Barrier(cart));
    m::fin();
}
