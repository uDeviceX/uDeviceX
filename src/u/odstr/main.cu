#include <stdio.h>
#include <assert.h>
#include <mpi.h>

#include "msg.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "mpi/basetags.h"
#include "glb.h"
#include "frag.h"

#include "comm/imp.h"
#include "distr/flu/imp.h"

int main(int argc, char **argv) {
    m::ini(argc, argv);
    MSG("mpi size: %d", m::size);
    MSG("Comm unit test!");

    basetags::TagGen tg;
    comm::hBags sendB, recvB;
    comm::Stamp stamp;

    ini(/**/ &tg);
    ini_no_bulk(sizeof(int), 26, /**/ &sendB);
    ini_no_bulk(sizeof(int), 26, /**/ &recvB);
    ini(m::cart, /*io*/ &tg, /**/ &stamp);

    fill_bags(&sendB);

    post_recv(&recvB, &stamp);
    post_send(&sendB, &stamp);

    wait_recv(&stamp, &recvB);
    wait_send(&stamp);

    compare(&sendB, &recvB);

    MSG("Passed");
    
    fin(&sendB);
    fin(&recvB);
    fin(/**/ &stamp);
    
    m::fin();
}
