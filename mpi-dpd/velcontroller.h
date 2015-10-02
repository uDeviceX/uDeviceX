/*
 *  velcontroller.h
 *  ctc falcon
 *
 *  Created by Dmitry Alexeev on Sep 24, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "common.h"

class VelController
{
    MPI_Comm comm;
    int size, rank;

    SimpleDeviceBuffer<float3> vel;
    PinnedHostBuffer<float3> avgvel;
    int myxl[3], myxh[3], n[3];
    int total, globtot;

    float3 desired;
    float Kp, Ki, Kd;
    float3 s, old;

    int sampleid;

public:
    VelController(int xl[3], int xh[3], int mpicoos[3], float3 desired, MPI_Comm comm);

    void sample(const int * const cellsstart, const Particle* const p, cudaStream_t stream);
    float3 adjustF(cudaStream_t stream);
};




