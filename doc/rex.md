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
         int *counts, *starts, *offsets;
         int *tstarts;
     };

`TicketPinned` is a copy of `tstarts` and `offsets`
