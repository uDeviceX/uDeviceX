#include <stdio.h>
#include <mpi.h>

#include "msg.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "mpi/basetags.h"
#include "glb.h"

#include "comm/imp.h"

void fill_bag(int val, int sz, int *ii) {
    for (int i = 0; i < sz; ++i) ii[i] = val;
}

void fill_bags(comm::Bags *b) {
    int c, i;
    for (i = 0; i < 26; ++i) {
        c = b->capacity[i] / 2;
        fill_bag(i, c, (int*) b->hst[i]);
        b->counts[i] = c;
    }
}

int main(int argc, char **argv) {
    m::ini(argc, argv);
    MSG("mpi size: %d", m::size);
    MSG("Comm unit test!");

    basetags::TagGen tg;
    comm::Bags sendB, recvB;
    comm::Stamp stamp;

    ini(/**/ &tg);
    ini_no_bulk(sizeof(int), 1, /**/ &sendB);
    ini_no_bulk(sizeof(int), 1, /**/ &recvB);
    ini(m::cart, /*io*/ &tg, /**/ &stamp);

    fill_bags(&sendB);

    post_recv(&recvB, &stamp);
    post_send(&sendB, &stamp);

    recv_counts(&stamp, /**/ &recvB);

    wait_recv(&stamp);
    wait_send(&stamp);
    
    fin(&sendB);
    fin(&recvB);
    fin(/**/ &stamp);
    
    m::fin();
}
