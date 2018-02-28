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

## Momentum transfer

### Gather momentum

Each triangle gets a contribution of bounce back, stored in `Momentum` structure.
Linear Momentum is straight forward but angular momentum needs a referential coordinate.
We choose here the center of mass of the triangle as it is independent on the domain, therefore communication
do not need special care.

### collect momentum

Momentum should be transfered to the different objects.
For Rigid objets it needs to be shifted in the referencial of the object.
