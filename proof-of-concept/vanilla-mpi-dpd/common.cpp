/*
 *  common.cpp
 *  Part of uDeviceX/vanilla-mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-11-07.
 *  Copyright 2015. All rights reserved.
 *
 */

#include "common.h"

bool Particle::initialized = false;

MPI_Datatype Particle::mytype;
