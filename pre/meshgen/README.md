Mesh Generation
===============

Tools for generating simple triangle mesh

Compilation
-----------
```sh
make
```

Sphere
------
Usage:
```sh
./gensphere <out.ply> <R> <nsub2> <nsub3>
```
- `out.ply` : file name
- `R` : radius
- `nsub2` : number of subdivision of type 2 (1 triangle -> 4 triangles)
- `nsub3` : number of subdivision of type 3 (1 triangle -> 9 triangles)

Ellipsoid
---------
Usage:
```sh
./genellipse <out.ply> <a> <b> <c> <nsub2> <nsub3>
```
- `a`, `b`, `c`: principal axes in x, y, z directions
