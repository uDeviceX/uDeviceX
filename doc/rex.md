# rex

**rex** [R]BC [ex]change hiwi : despite the name it packs both RBCs
and "solids"

## halo
`halo` is composed of 26 `frag` : fragments

## purpose

- packs particles of objects (RBCs and solids) to `halo`
- sends each `frag` of `halo` to another processor
- each fragment interacts with local particles and objects
- sends a force of a fragment back.

`TicketPack` is an index which helps to pack particles

	 struct TicketPack {
		 int *counts, *starts;
		 int *offsets, *tstarts;
	 };


`TicketPinned` is a host copy of `TicketPack`

	struct TicketPinned { /* helps pack particles (hst) */
		int *tstarts;
		int *offsets;
	};


`buf` is a "packed" particles (dev)
`buf_pinned` is a copy of `buf` (hst)

`send_counts[i]` is a copy of the "last part" of `TicketPinned.offsets`

`remote[i]->pmessage` is for the "extra" particles
