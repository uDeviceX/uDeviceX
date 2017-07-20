# Hiwi for solvent distribution

See [sim.md](../../doc/sim.md)

## purpose

* distributes solvent particles across nodes
* builds cell lists
* (optional) keep track of int data, such as global ids or tags

## usage

* all `_pp` functions, as well as `bulk`, are mandatory to distribute particles (see ticket dependencies below for the order of calls within a step).
* The `_ii` functions are optional and may be called only when global ids and/or tags are needed.
* for each int data, 2 tickets (`TicketI` and `TicketUI`) are needed. 

## ticket dependencies:

![alt text][graphviz/deps.png]

"A->B" reads "B depends on A"  
"A->B->C" reads "C depends on B, which depends on A"

### TicketD: 
* `post_recv_pp`->`pack_pp`->`send_pp`->`bulk`->`recv_pp`->`unpack_pp`->`gather_pp`
* `post_recv_pp`->`post_recv_ii`
* `pack_pp`->`pack_ii`
* `send_pp`->`send_ii`
* `unpack_pp`->`unpack_ii`

### TicketU:
* `unpack_pp`->`gather_pp`->`gather_ii`

### TicketUI:
* `unpack_ii`->`gather_ii`

### TicketI:
* `post_recv_ii`->`pack_ii`->`send_ii`->`recv_ii`->`unpack_ii`
