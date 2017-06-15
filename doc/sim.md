# sim.md

`sim` uses `hiwi`. Interfaces for `hiwi` are in
[int/](../src/int). One `hiwi` consists of several `struct`s and
functions. Interface for `hiwi` is in [int](../src/int).

`struct`s are the following:

* `Q` : quantities : states variables of the simulation. ex `pp`, `np`.

* `W` : work : work variables. ex: exchange buffers in distribute
  functions.

* `T1, T2`, ... : tickets : ex: `zip` variables for solvent

`sim` defines variables of types `Q`, `W`, `T` and calls functions
of `hiwi`. `sim` should follows some convention on how the functions
are called: 

* `w` is allocated by `hi::alloc_work()`
* `t` is allocated by `hi::alloc_ticket()`, `hi::alloc_ticket1()`
* `t` is not modified by `sim`

Functions of `hi::` can
* issue ticket : return `t`
* check ticket : receive `t` as an argument
* check and invalidate ticket : receive `t` and make it invalid

* modification of `q` by `sim` makes all tickets invalid

The system of ticket imposes a constrain on the order in whcih sim
call functions of `hi`.

## Notation
* `hi` : is a specific type of `hiwi`
* `w`, `q`, `t` : ariables of the `W`, `Q` and one of `T1`, `T2`, ...
