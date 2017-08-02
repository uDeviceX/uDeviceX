# mbounce

mesh bounce hiwi

## purpose

* bounce particles on meshes
* transfer change of linear and angular momenta to the object

## datastructures

### mesh:

* faces indices and Particles for the mesh

### triangle cell list (parallel to mesh, i.e. no need to reorder the faces)

* cell starts
* cell counts
* triangle indices
