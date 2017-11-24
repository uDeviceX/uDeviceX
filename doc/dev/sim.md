# Intro
`sim` uses `hiwi`. Interfaces for `hiwi` are in
[int/](../src/int). One `hiwi` consists of several `struct`s and
functions.

`struct`s are the following:

* `Q` : quantities : states variables of the simulation. ex `pp`, `np`

* `W` : work : work variables. ex: exchange buffers in distribute
  functions

* `T1, T2`, ... : tickets : ex: `zip` variables for solvent

`sim` defines `w`, `q`, `t` and calls functions of `hiwi`. `sim` is
rectricted by the following rules:

* `w` is allocated by `hi::alloc_work()`
* `t` is allocated by `hi::alloc_ticket()`, `hi::alloc_ticket1()` and
  is not modified by `sim`

Functions of `hi::` can
* issue ticket : return `t`
* check ticket : receive `t` as an argument
* check and invalidate ticket : receive `t` and make it invalid

* direct modification of `q` by `sim` makes all tickets invalid

The system of ticket imposes a constrain on the order in whcih sim
call functions of `hi`.

# `hiwi`

`hiwi` is scattred in several files

* hdr/hi.h : declaration of host variables ([h]ea[d]e[r]) : no hdr for
  good `hiwi`
* imp/hi.h : [imp]limentation of host functions
* dev/hi.h : implimentation of [dev]ice functions
* int/hi.h : [int]erface

All files are included in [bund.cu](../src/bund.cu).

`int/hi.h` should "unpack/pack" `QWT` structures and call functions of `dev/hi.h`

# bund.cu

it includes all files of hiwi

	namespace hi {
	  int/hi.h
	  namespace i {
		hdr/hi.h
		imp/hi.h
		dev/hi.h
	  }
	}

# Notation
* `hi` : is a an example of `hiwi`
* `w`, `q`, `t` : variables of `QWT`
