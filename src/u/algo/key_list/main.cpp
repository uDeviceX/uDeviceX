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
    KeyList_ini(&q);
    KeyList_append(q, "a");
    KeyList_append(q, "b c");
    KeyList_append(q, "d");

    KeyList_copy(q, /**/ &p);

    msg_print("offset: %d", KeyList_offset(q, "a"));
    msg_print("offset: %d", KeyList_offset(q, "b c"));
    msg_print("offset: %d", KeyList_offset(q, "d"));

    msg_print("width: %d", KeyList_width(q, "b c"));
    msg_print("size: %d", KeyList_size(q));

    KeyList_mark(q, "a");
    KeyList_mark(q, "b c");
    KeyList_mark(q, "d");

    msg_print("marked: %d", KeyList_marked(q));
    KeyList_log(q);
    KeyList_log(p);

    KeyList_fin(p);
    KeyList_fin(q);
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
