# Hiwi for solvent distribution

See [sim.md](../../doc/sim.md)

## purpose

* distributes solvent particles across nodes
* builds cell lists

## ticket dependencies:

A->B reads B depends on A  
A->B->C reads C depends on B, which depends on A

### TicketD: 
* post_recv_pp->pack_pp->send_pp->bulk->recv_pp->unpack_pp->gather_pp
* post_recv_pp->post_recv_ii
* pack_pp->pack_ii
* send_pp->send_ii
* unpack_pp->unpack_ii

### TicketU:
* unpack_pp->unpack_ii
* unpack_pp->gather_pp
* unpack_ii->gather_ii

### TicketI:
* post_recv_ii->pack_ii->send_ii->recv_ii->unpack_ii
