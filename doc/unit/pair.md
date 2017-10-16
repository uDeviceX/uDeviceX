# Intro

It is for test pair interactions.

# Compile

Run from src/

	u.conf0 u/pair

or from other directorie

	s=<path to src>
	echo run | u.conf $s u/pair $s/conf/test.h

Build

	u.make -j

# Run

	s=<path to src>
	./udx < $s/data/pair/2

Returns force between two particles

	-2.4 0 0

It also dumps pariticles to `stderr`

	[ 0 0 0 ] [ 0 0 0 ] [kc: 1 0]
	[ 0.1 0 0 ] [ 0 0 0 ] [kc: 0 1]

# Source

[u/pair/imp/main.h](../../src/u/pair/imp/main.h)

# Test

	u.test test/pair
