# meshbb

mesh bounce back

## purpose

find and perform collisions between Particles `pp` and a set of mesh.

## data structures

each particle has a list of a maximum of `MAX_COL` collisions. Each entry contains:

* collision time `tcol`
* triangle id `tid`
* collision coordinates `u`, `v` such that the collision is at rc = u * (B-A) + v * (C-A) for triangle ABC
* `s`: "sign", telling on which side of the triangle the particle should be
