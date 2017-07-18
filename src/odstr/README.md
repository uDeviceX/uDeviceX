# Hiwi for solvent distribution

See [sim.md](../../doc/sim.md)

## purpose

* distributes solvent particles across nodes
* builds cell lists

## ticket dependencies:

### TicketD: 
* post->pack->send->bulk->recv->unpack_pp
* unpack_pp->unpack_ii
* unpack_pp->gather_pp

### TicketU:
* unpack_pp->unpack_ii
* unpack_pp->gather_pp
* unpack_ii->gather_ii
