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
few steps. D means dumps restart files, R means read restart files. I
want to have tests durining the steps.

## S0
B, M, A the same. They all have the same states

B: bma
M: bma
A: bma

## S1
B: bD
M: bma
A: bma

B dumps and stops. I cannot test.

## S2
B: bD
M: Rma
A: bma

I can connect B and M (M reads dumps from B) and test.

## S3
B: bD
M: RmD
A: Ra

I can connect B, M, A and test.

## S4
B: with no (`nsteps=0`) morphes into **pre-processor**
M: morphes into **inter-processor**
A: RaD will be **udx** and great again
