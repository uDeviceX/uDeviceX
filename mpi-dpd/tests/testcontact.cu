/*
 *  main.cu
 *  ctc PANDA
 *
 *  Created by Dmitry Alexeev on Oct 20, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

/*
 *  main.cu
 *  Part of uDeviceX/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-11-14.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <cstdio>
#include <cassert>
#include <csignal>
#include <mpi.h>
#include <errno.h>
#include <vector>

#include <argument-parser.h>
#include <common.h>
#include <containers.h>
#include <contact.h>
#include <dpd-rng.h>
#include <solute-exchange.h>

using namespace std;

float tend;
bool walls, pushtheflow, doublepoiseuille, rbcs, ctcs, xyz_dumps, hdf5field_dumps, hdf5part_dumps, is_mps_enabled, adjust_message_sizes, contactforces;
int steps_per_report, steps_per_dump, wall_creation_stepid, nvtxstart, nvtxstop;

LocalComm localcomm;

static const float ljsigma = 0.5;
static const float ljsigma2 = ljsigma * ljsigma;

template<int s>
inline  float _viscosity_function(float x)
{
    return sqrtf(viscosity_function<s - 1>(x));
}

template<> inline  float _viscosity_function<1>(float x) { return sqrtf(x); }
template<> inline  float _viscosity_function<0>(float x){ return x; }

int main(int argc, char ** argv)
{
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceReset());

    {
        is_mps_enabled = false;

        const char * mps_variables[] = {
                "CRAY_CUDA_MPS",
                "CUDA_MPS",
                "CRAY_CUDA_PROXY",
                "CUDA_PROXY"
        };

        for(int i = 0; i < 4; ++i)
            is_mps_enabled |= getenv(mps_variables[i])!= NULL && atoi(getenv(mps_variables[i])) != 0;
    }

    int nranks, rank;
    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK( MPI_Comm_size(MPI_COMM_WORLD, &nranks) );
    MPI_CHECK( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
    MPI_Comm activecomm = MPI_COMM_WORLD;

    bool reordering = true;
    const char * env_reorder = getenv("MPICH_RANK_REORDER_METHOD");

    MPI_Comm cartcomm;
    int periods[] = {1, 1, 1};
    int ranks[] = {1, 1, 1};


    MPI_CHECK( MPI_Cart_create(activecomm, 3, ranks, periods, (int)reordering, &cartcomm) );
    activecomm = cartcomm;


    //RAII

    const double xs2 = 2.0 / XSIZE_SUBDOMAIN;
    const double ys2 = 2.0 / YSIZE_SUBDOMAIN;
    const double zs2 = 2.0 / ZSIZE_SUBDOMAIN;

    {
        MPI_CHECK(MPI_Barrier(activecomm));
        localcomm.initialize(activecomm);

        MPI_CHECK(MPI_Barrier(activecomm));

        // test here
        srand48(time(NULL));
        //srand48(0);

        int dens = 10;
        int l = 20;
        int n = l*l*l*dens;
        vector<Particle> ic(n);
        vector<Acceleration> acc(n);
        for (int i=0; i<n; i++)
            for (int d=0; d<3; d++)
                acc[i].a[d] = 0;
        vector<Acceleration> gpuacc(n);

        const int L[3] = { l, l, l };

        Logistic::KISS local_trunk = Logistic::KISS(7119 - rank, 187 + rank, 18278, 15674);

        int basex = XSIZE_SUBDOMAIN*1.0 - 0.5*l;
        int basey = YSIZE_SUBDOMAIN*1.0 - 0.5*l;
        int basez = ZSIZE_SUBDOMAIN*1.0 - 0.5*l;

        for(int iz = 0; iz < L[2]; iz++)
            for(int iy = 0; iy < L[1]; iy++)
                for(int ix = 0; ix < L[0]; ix++)
                    for(int l = 0; l < dens; ++l)
                    {
                        const int p = l + dens * (ix + L[0] * (iy + L[1] * iz));

//                        ic[p].x[0] = (ix + basex + XSIZE_SUBDOMAIN) % XSIZE_SUBDOMAIN - 0.5*XSIZE_SUBDOMAIN + 0.99 * drand48() + ((ix<floor(0.5*L[0])) ? 2 : 0);
//                        ic[p].x[1] = (iy + basey + YSIZE_SUBDOMAIN) % YSIZE_SUBDOMAIN - 0.5*YSIZE_SUBDOMAIN + 0.99 * drand48() + ((iy<floor(0.5*L[0])) ? 2 : 0);
//                        ic[p].x[2] = (iz + basez + ZSIZE_SUBDOMAIN) % ZSIZE_SUBDOMAIN - 0.5*ZSIZE_SUBDOMAIN + 0.99 * drand48() ;

                        ic[p].x[0] = (XSIZE_SUBDOMAIN + 8) * drand48() - 0.5*XSIZE_SUBDOMAIN - 4;
                        ic[p].x[1] = (YSIZE_SUBDOMAIN + 8) * drand48() - 0.5*YSIZE_SUBDOMAIN - 4;
                        ic[p].x[2] = (ZSIZE_SUBDOMAIN + 8) * drand48() - 0.5*ZSIZE_SUBDOMAIN - 4;
                        ic[p].u[0] = 0.5 - drand48();
                        ic[p].u[1] = 0.5 - drand48();
                        ic[p].u[2] = 0.5 - drand48();

                        //printf("%d: [%.3f %.3f %.3f]\n", p, ic[p].x[0], ic[p].x[1], ic[p].x[2]);
                    }

        float seed = local_trunk.get_float();

#pragma omp parallel for
        for (int i=0; i<n; i++)
            for (int j=0; j<n; j++)
                if (i != j)
                {
                    double _xr = ic[i].x[0] - ic[j].x[0];
                    double _yr = ic[i].x[1] - ic[j].x[1];
                    double _zr = ic[i].x[2] - ic[j].x[2];

                    while (_xr >= XSIZE_SUBDOMAIN) _xr -= XSIZE_SUBDOMAIN;
                    while (_xr < 0              ) _xr += XSIZE_SUBDOMAIN;
                    _xr = _xr - (int)(_xr * xs2) * XSIZE_SUBDOMAIN;
                    while (_yr >= YSIZE_SUBDOMAIN) _yr -= YSIZE_SUBDOMAIN;
                    while (_yr < 0              ) _yr += YSIZE_SUBDOMAIN;
                    _yr = _yr - (int)(_yr * ys2) * YSIZE_SUBDOMAIN;
                    while (_zr >= ZSIZE_SUBDOMAIN) _zr -= ZSIZE_SUBDOMAIN;
                    while (_zr < 0              ) _zr += ZSIZE_SUBDOMAIN;
                    _zr = _zr - (int)(_zr * zs2) * ZSIZE_SUBDOMAIN;


                    if (false)
                    {
                        int s = 0;
                        s += (ic[i].x[0] > 0.5*XSIZE_SUBDOMAIN-1 || ic[i].x[0] < -0.5*XSIZE_SUBDOMAIN+1);
                        s += (ic[i].x[1] > 0.5*YSIZE_SUBDOMAIN-1 || ic[i].x[1] < -0.5*YSIZE_SUBDOMAIN+1);
                        s += (ic[i].x[2] > 0.5*ZSIZE_SUBDOMAIN-1 || ic[i].x[2] < -0.5*ZSIZE_SUBDOMAIN+1);
                        s = (1 << s) - 1;

                        acc[i].a[0] = s;
                        acc[i].a[1] = s;
                        acc[i].a[2] = s;
                        continue;
                    }

                    const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;
                    const float invrij = rsqrtf(rij2);
                    const float rij = rij2 * invrij;

                    if (rij2 >= 1)
                        continue;

                    const float invr2 = invrij * invrij;
                    const float t2 = ljsigma2 * invr2;
                    const float t4 = t2 * t2;
                    const float t6 = t4 * t2;
                    const float lj = min(1e3f, max(0.f, 24.f * invrij * t6 * (2.f * t6 - 1.f)));

                    const float wr = _viscosity_function<0>(1.f - rij);

                    const float xr = _xr * invrij;
                    const float yr = _yr * invrij;
                    const float zr = _zr * invrij;

                    const float rdotv =
                            xr * (ic[i].u[0] - ic[j].u[0]) +
                            yr * (ic[i].u[1] - ic[j].u[1]) +
                            zr * (ic[i].u[2] - ic[j].u[2]);

                    const float myrandnr = 0;//Logistic::mean0var1(seed, i, j);

                    const float strength = lj + (- gammadpd * wr * rdotv + sigmaf * myrandnr) * wr;

                    const float xinteraction = strength * xr;
                    const float yinteraction = strength * yr;
                    const float zinteraction = strength * zr;

                    acc[i].a[0] += xinteraction;
                    acc[i].a[1] += yinteraction;
                    acc[i].a[2] += zinteraction;

//                    printf("%d [%.3f %.3f %.3f] coll %d [%.3f %.3f %.3f],   [%.3f %.3f %.3f] %f\n",
//                            i, ic[i].x[0], ic[i].x[1], ic[i].x[2],
//                            j, ic[j].x[0], ic[j].x[1], ic[j].x[2],
//                            _xr, _yr, _zr, rij2);
                }

        ParticleArray p;
        p.resize(n);

        CUDA_CHECK( cudaMemcpy(p.xyzuvw.data, &ic[0], n * sizeof(Particle), cudaMemcpyHostToDevice) );
        CUDA_CHECK( cudaMemset(p.axayaz.data, 0, n * sizeof(Acceleration)) );
        std::vector<ParticlesWrap> wsolutes;
        wsolutes.push_back(ParticlesWrap(p.xyzuvw.data, n, p.axayaz.data));

        ComputeContact contact(cartcomm);
        SoluteExchange solutex(cartcomm);

        solutex.attach_halocomputation(contact);
        contact.build_cells(wsolutes, 0);
        contact.bulk(wsolutes, 0);

        solutex.bind_solutes(wsolutes);
        solutex.pack_p(0);
        solutex.post_p(0, 0);
        solutex.recv_p(0);
        solutex.halo(0, 0);
        solutex.post_a();
        solutex.recv_a(0);

        CUDA_CHECK( cudaMemcpy(&gpuacc[0], p.axayaz.data, l*l*l*dens * sizeof(Acceleration), cudaMemcpyDeviceToHost) );
        CUDA_CHECK(cudaDeviceSynchronize());

        float linf = 0, l2 = 0;
        float fx = 0, fy = 0, fz = 0;
        float hfx = 0, hfy = 0, hfz = 0;

        for (int i=0; i<n; i++)
        {
            float diff = sqrt((acc[i].a[0] - gpuacc[i].a[0]) * ( acc[i].a[0] - gpuacc[i].a[0]) +
                    (acc[i].a[1] - gpuacc[i].a[1]) * ( acc[i].a[1] - gpuacc[i].a[1]) +
                    (acc[i].a[2] - gpuacc[i].a[2]) * ( acc[i].a[2] - gpuacc[i].a[2]));

            fx += gpuacc[i].a[0];
            fy += gpuacc[i].a[1];
            fz += gpuacc[i].a[2];

            hfx += acc[i].a[0];
            hfy += acc[i].a[1];
            hfz += acc[i].a[2];

            l2 += diff;
            linf = max(diff, linf);
            if (diff > 0.2) printf("%d:  CPU [%.5f %.5f %.5f]  GPU [%.5f %.5f %.5f]\n", i, acc[i].a[0], acc[i].a[1], acc[i].a[2], gpuacc[i].a[0], gpuacc[i].a[1], gpuacc[i].a[2]);
        }

        l2 /= n;
        printf("Linf:  %f,  L2:  %f,  F [%8f %8f %8f], CPU F [%8f %8f %8f]\n", linf, l2, fx, fy, fz,  hfx, hfy, hfz);

    }

    if (activecomm != cartcomm)
        MPI_CHECK(MPI_Comm_free(&activecomm));

    MPI_CHECK(MPI_Comm_free(&cartcomm));

    MPI_CHECK(MPI_Finalize());

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaDeviceReset());

    return 0;
}



