#!/bin/bash

c=10.0
i=mir.sdf
o=smo.sdf
b=smo.bov

./sdf.smooth cubic $c $i  $o

sdf.2bov $o $b
