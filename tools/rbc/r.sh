#!/bin/bash

# Test call

r=1
f=0.1
sc=1 ang=0
sol=solid.out
rbc=rbc.out

plcmt.ro 10 20 30 $r     $f    $sc $ang     $sol $rbc
