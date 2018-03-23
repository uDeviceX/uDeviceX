#include <stdio.h>
#include <float.h>
#include <mpi.h>

#include "utils/mc.h"
#include "utils/msg.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "utils/mc.h"

#include "algo/key_list/imp.h"

void main0() {
    KeyList *q, *p;
    key_list_ini(&q);
    key_list_append(q, "a");
    key_list_append(q, "b c");
    key_list_append(q, "d");

    key_list_copy(q, /**/ &p);

    msg_print("offset: %d", key_list_offset(q, "a"));
    msg_print("offset: %d", key_list_offset(q, "b c"));
    msg_print("offset: %d", key_list_offset(q, "d"));

    msg_print("width: %d", key_list_width(q, "b c"));
    msg_print("size: %d", key_list_size(q));

    key_list_mark(q, "a");
    key_list_mark(q, "b c");
    key_list_mark(q, "d");

    msg_print("marked: %d", key_list_marked(q));
    key_list_log(q);
    key_list_log(p);

    key_list_fin(p);
    key_list_fin(q);
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
