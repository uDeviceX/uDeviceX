/*
 *  cell-lists.h
 *  Part of uDeviceX/cuda-dpd-sem/
 *
 *  Created and authored by Diego Rossinelli on 2014-07-21.
 *  Copyright 2015. All rights reserved.
 *
 */

#pragma once

#include <utility>

void build_clists(float * const device_xyzuvw, int np, const float rc, 
		  const int xcells, const int ycells, const int zcells,
		  const float xdomainstart, const float ydomainstart, const float zdomainstart,
		  int * const host_order, int * device_cellsstart, int * device_cellscount,
		  std::pair<int, int *> * nonemptycells = NULL, cudaStream_t stream = 0, const float * const src_device_xyzuvw = NULL);
		  
void build_clists_vanilla(float * const device_xyzuvw, int np, const float rc, 
			  const int xcells, const int ycells, const int zcells,
			  const float xdomainstart, const float ydomainstart, const float zdomainstart,
			  int * const host_order, int * device_cellsstart, int * device_cellscount,
			  std::pair<int, int *> * nonemptycells = NULL, cudaStream_t stream = 0, const float * const src_device_xyzuvw = NULL);
