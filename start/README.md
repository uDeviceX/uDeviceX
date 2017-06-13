# start/restart plan

There are foure "states" of udx:
* N : no wall
* B : before wall
* M : making wall
* A : after  wall

Is there any difference between N and B?  It the same loop with
`run0(driving_force0, wall_created, it)`. I will ignore N in the
following.

I want to have three version of udx: B, M, A. I will change them in
few steps. D means dumps restart files, R means read restart files.

## S0
B, M, A the same. They all have the same states

B: bma
M: bma
A: bma

## S1
B: bD
M: bma
A: bma

## S2
B: bD
M: bRm
A: bma

b in M is not needed.

## S3
B: bD
M: RmD
A: Ra

## S4
M: morphes into **interporcessor**
B: with b=0 (no `nsteps=0`) morphes into **preprocessor**
A: RaD will be a **udx** again
