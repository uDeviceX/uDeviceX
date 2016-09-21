/*
 *  minmax-massimo.h
 *  Part of uDeviceX/mpi-dpd/
 *
 *  Created and authored by Massimo Bernaschi on 2015-03-24.
 *  Copyright 2015. All rights reserved.
 *
 */

#pragma once

#include "common.h"

void minmax(const Particle * const particles, int nparticles_per_body, int nbodies, 
		    float3 * minextents, float3 * maxextents, cudaStream_t stream);
