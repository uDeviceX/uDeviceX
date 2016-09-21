/*
 *  cell-factory.h
 *  Part of uDeviceX/cuda-dpd-sem/sem/
 *
 *  Created and authored by Diego Rossinelli on 2014-08-08.
 *  Copyright 2015. All rights reserved.
 *
 */

#pragma once

struct ParamsSEM
{
	float rcutoff, gamma, u0, rho, req, D, rc;
};

void cell_factory(int n, float * xyz, ParamsSEM& params);

