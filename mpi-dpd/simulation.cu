/*
 *  simulation.cu
 *  Part of uDeviceX/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2015-03-24.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <sys/stat.h>

#include "common.h"
#include "containers.h"
#include "solvent-exchange.h"
#include "dpd.h"
#include "wall.h"
#include "solute-exchange.h"
#include "fsi.h"
#include "contact.h"
#include "redistribute-particles.h"
#include "redistribute-rbcs.h"
#include "io.h"
#include "velcontroller.h"
#include "simulation.h"
#include "dpd-forces.h"
#include "last_bit_float.h"
#include "geom-wrapper.h"

#define NPMAX 5000000 /* TODO: */
float rbc_xx[NPMAX], rbc_yy[NPMAX], rbc_zz[NPMAX];
float sol_xx[NPMAX], sol_yy[NPMAX], sol_zz[NPMAX];
int   iotags[NPMAX];

__global__ void make_texture( float4 * __restrict xyzouvwo, ushort4 * __restrict xyzo_half, const float * __restrict xyzuvw, const uint n )
{
    extern __shared__ volatile float  smem[];
    const uint warpid = threadIdx.x / 32;
    const uint lane = threadIdx.x % 32;

    const uint i =  (blockIdx.x * blockDim.x + threadIdx.x ) & 0xFFFFFFE0U;

    const float2 * base = ( float2* )( xyzuvw +  i * 6 );
#pragma unroll 3
    for( uint j = lane; j < 96; j += 32 ) {
        float2 u = base[j];
        // NVCC bug: no operator = between volatile float2 and float2
        asm volatile( "st.volatile.shared.v2.f32 [%0], {%1, %2};" : : "r"( ( warpid * 96 + j )*8 ), "f"( u.x ), "f"( u.y ) : "memory" );
    }
    // SMEM: XYZUVW XYZUVW ...
    uint pid = lane / 2;
    const uint x_or_v = ( lane % 2 ) * 3;
    xyzouvwo[ i * 2 + lane ] = make_float4( smem[ warpid * 192 + pid * 6 + x_or_v + 0 ],
            smem[ warpid * 192 + pid * 6 + x_or_v + 1 ],
            smem[ warpid * 192 + pid * 6 + x_or_v + 2 ], 0 );
    pid += 16;
    xyzouvwo[ i * 2 + lane + 32] = make_float4( smem[ warpid * 192 + pid * 6 + x_or_v + 0 ],
            smem[ warpid * 192 + pid * 6 + x_or_v + 1 ],
            smem[ warpid * 192 + pid * 6 + x_or_v + 2 ], 0 );

    xyzo_half[i + lane] = make_ushort4( __float2half_rn( smem[ warpid * 192 + lane * 6 + 0 ] ),
            __float2half_rn( smem[ warpid * 192 + lane * 6 + 1 ] ),
            __float2half_rn( smem[ warpid * 192 + lane * 6 + 2 ] ), 0 );
    // }
}

void Simulation::_update_helper_arrays()
{
    CUDA_CHECK( cudaFuncSetCacheConfig( make_texture, cudaFuncCachePreferShared ) );

    const int np = particles->size;

    xyzouvwo.resize(2 * np);
    xyzo_half.resize(np);

    if (np)
        make_texture <<< (np + 1023) / 1024, 1024, 1024 * 6 * sizeof( float )>>>(xyzouvwo.data, xyzo_half.data, (float *)particles->xyzuvw.data, np );

    CUDA_CHECK(cudaPeekAtLastError());
}

/* set initial velocity of a particle */
void _ic_vel(float x, float y, float z, float* vx, float* vy, float* vz) {
  float gd = 0.5*desired_shrate; /* "gamma dot" */
  *vx = gd*z; *vy = 0; *vz = 0;
}

std::vector<Particle> Simulation::_ic() {
    srand48(rank);
    std::vector<Particle> pp;
    int  L[3] = {XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN};
    int iz, iy, ix, l, nd   = numberdensity;
    for (iz = 0; iz < L[2]; iz++)
        for (iy = 0; iy < L[1]; iy++)
            for (ix = 0; ix < L[0]; ix++) {
	      /* edge of a cell */
	      int xlo = -L[0]/2 + ix, ylo = -L[1]/2 + iy, zlo = -L[2]/2 + iz;
	      for (l = 0; l < nd; l++) {
		Particle p = Particle(); float dr = 0.99;
		float x = xlo + dr*drand48();
		float y = ylo + dr*drand48();
		float z = zlo + dr*drand48();
		p.x[0] = x; p.x[1] = y; p.x[2] = z;
		_ic_vel(x, y, z, &p.u[0], &p.u[1], &p.u[2]);
		pp.push_back(p);
	      }
            }
    fprintf(stderr, "(simulation.cu) generated %d\n solvent particles", pp.size());
    return pp;
}

void Simulation::_redistribute()
{
    double tstart = MPI_Wtime();

    redistribute.pack(particles->xyzuvw.data, particles->size, mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    if (rbcscoll)
        redistribute_rbcs.extent(rbcscoll->data(), rbcscoll->count(), mainstream);

    redistribute.send();

    if (rbcscoll)
        redistribute_rbcs.pack_sendcount(rbcscoll->data(), rbcscoll->count(), mainstream);

    redistribute.bulk(particles->size, cells.start, cells.count, mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    const int newnp = redistribute.recv_count(mainstream, host_idle_time);

    int nrbcs;
    if (rbcscoll)
        nrbcs = redistribute_rbcs.post();

    if (rbcscoll)
        rbcscoll->resize(nrbcs);

    newparticles->resize(newnp);
    xyzouvwo.resize(newnp * 2);
    xyzo_half.resize(newnp);

    redistribute.recv_unpack(newparticles->xyzuvw.data, xyzouvwo.data, xyzo_half.data, newnp, cells.start, cells.count, mainstream, host_idle_time);

    CUDA_CHECK(cudaPeekAtLastError());

    swap(particles, newparticles);

    if (rbcscoll)
        redistribute_rbcs.unpack(rbcscoll->data(), rbcscoll->count(), mainstream);

    CUDA_CHECK(cudaPeekAtLastError());
}

void Simulation::_remove_bodies_from_wall(CollectionRBC * coll)
{
    if (!coll || !coll->count())
        return;

    SimpleDeviceBuffer<int> marks(coll->pcount());

    SolidWallsKernel::fill_keys<<< (coll->pcount() + 127) / 128, 128 >>>(coll->data(), coll->pcount(), marks.data);

    vector<int> tmp(marks.size);
    CUDA_CHECK(cudaMemcpy(tmp.data(), marks.data, sizeof(int) * marks.size, cudaMemcpyDeviceToHost));

    const int nbodies = coll->count();
    const int nvertices = coll->get_nvertices();

    std::vector<int> tokill;
    for(int i = 0; i < nbodies; ++i)
    {
        bool valid = true;

        for(int j = 0; j < nvertices && valid; ++j)
            valid &= 0 == tmp[j + nvertices * i];

        if (!valid)
            tokill.push_back(i);
    }

    coll->remove(&tokill.front(), tokill.size());
    coll->clear_velocity();

    CUDA_CHECK(cudaPeekAtLastError());
}

void Simulation::_create_walls(const bool verbose, bool & termination_request)
{
    if (verbose)
        printf("creation of the walls...\n");

    int nsurvived = 0;
    ExpectedMessageSizes new_sizes;
    wall = new ComputeWall(cartcomm, particles->xyzuvw.data, particles->size, nsurvived, new_sizes, verbose);

    //adjust the message sizes if we're pushing the flow in x
    {
        const double xvelavg = getenv("XVELAVG") ? atof(getenv("XVELAVG")) : pushtheflow;
        const double yvelavg = getenv("YVELAVG") ? atof(getenv("YVELAVG")) : 0;
        const double zvelavg = getenv("ZVELAVG") ? atof(getenv("ZVELAVG")) : 0;

        for(int code = 0; code < 27; ++code)
        {
            const int d[3] = {
                (code % 3) - 1,
                ((code / 3) % 3) - 1,
                ((code / 9) % 3) - 1
            };

            const double IudotnI =
                fabs(d[0] * xvelavg) +
                fabs(d[1] * yvelavg) +
                fabs(d[2] * zvelavg) ;

            const float factor = 1 + IudotnI * dt * 10 * numberdensity;

            new_sizes.msgsizes[code] *= factor;
        }
    }

    particles->resize(nsurvived);
    particles->clear_velocity();
    cells.build(particles->xyzuvw.data, particles->size, 0, NULL, NULL);

    _update_helper_arrays();

    CUDA_CHECK(cudaPeekAtLastError());

    //remove cells touching the wall
    _remove_bodies_from_wall(rbcscoll);

    {
        H5PartDump sd("survived-particles->h5part", activecomm, cartcomm);
        Particle * p = new Particle[particles->size];

        CUDA_CHECK(cudaMemcpy(p, particles->xyzuvw.data, sizeof(Particle) * particles->size, cudaMemcpyDeviceToHost));

        sd.dump(p, particles->size);

        delete [] p;
    }
}

void Simulation::_forces()
{
    double tstart = MPI_Wtime();

    SolventWrap wsolvent(particles->xyzuvw.data, particles->size, particles->axayaz.data, cells.start, cells.count);

    std::vector<ParticlesWrap> wsolutes;

    if (rbcscoll)
        wsolutes.push_back(ParticlesWrap(rbcscoll->data(), rbcscoll->pcount(), rbcscoll->acc()));

    fsi.bind_solvent(wsolvent);

    solutex.bind_solutes(wsolutes);

    particles->clear_acc(mainstream);

    if (rbcscoll)
        rbcscoll->clear_acc(mainstream);

    dpd.pack(particles->xyzuvw.data, particles->size, cells.start, cells.count, mainstream);

    solutex.pack_p(mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    if (contactforces)
        contact.build_cells(wsolutes, mainstream);

    dpd.local_interactions(particles->xyzuvw.data, xyzouvwo.data, xyzo_half.data, particles->size, particles->axayaz.data,
            cells.start, cells.count, mainstream);

    if (!doublepoiseuille) {
        velcont1->push(cells.start, particles->xyzuvw.data, particles->axayaz.data, mainstream);
        velcont2->push(cells.start, particles->xyzuvw.data, particles->axayaz.data, mainstream);
    }

    dpd.post(particles->xyzuvw.data, particles->size, mainstream, downloadstream);

    solutex.post_p(mainstream, downloadstream);

    CUDA_CHECK(cudaPeekAtLastError());

    if (rbcscoll && wall)
        wall->interactions(rbcscoll->data(), rbcscoll->pcount(), rbcscoll->acc(), NULL, NULL, mainstream);

    if (wall)
        wall->interactions(particles->xyzuvw.data, particles->size, particles->axayaz.data,
                cells.start, cells.count, mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    dpd.recv(mainstream, uploadstream);

    solutex.recv_p(uploadstream);

    solutex.halo(uploadstream, mainstream);

    dpd.remote_interactions(particles->xyzuvw.data, particles->size, particles->axayaz.data, mainstream, uploadstream);

    fsi.bulk(wsolutes, mainstream);

    if (contactforces)
        contact.bulk(wsolutes, mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    if (rbcscoll)
        CudaRBC::forces_nohost(mainstream, rbcscoll->count(), (float *)rbcscoll->data(), (float *)rbcscoll->acc());

    CUDA_CHECK(cudaPeekAtLastError());

    solutex.post_a();

    solutex.recv_a(mainstream);
    CUDA_CHECK(cudaPeekAtLastError());
}

void Simulation::_datadump(const int idtimestep)
{
    double tstart = MPI_Wtime();

    pthread_mutex_lock(&mutex_datadump);

    while (datadump_pending)
        pthread_cond_wait(&done_datadump, &mutex_datadump);

    int n = particles->size;

    if (rbcscoll)
        n += rbcscoll->pcount();

    particles_datadump.resize(n);
    accelerations_datadump.resize(n);

#include "simulation.hack.h"

    CUDA_CHECK(cudaMemcpyAsync(particles_datadump.data, particles->xyzuvw.data, sizeof(Particle) * particles->size, cudaMemcpyDeviceToHost,0));
    CUDA_CHECK(cudaMemcpyAsync(accelerations_datadump.data, particles->axayaz.data, sizeof(Acceleration) * particles->size, cudaMemcpyDeviceToHost,0));

    int start = particles->size;

    if (rbcscoll)
    {
        CUDA_CHECK(cudaMemcpyAsync(particles_datadump.data + start, rbcscoll->xyzuvw.data, sizeof(Particle) * rbcscoll->pcount(), cudaMemcpyDeviceToHost, 0));
        CUDA_CHECK(cudaMemcpyAsync(accelerations_datadump.data + start, rbcscoll->axayaz.data, sizeof(Acceleration) * rbcscoll->pcount(), cudaMemcpyDeviceToHost, 0));

        start += rbcscoll->pcount();
    }

    CUDA_CHECK(cudaEventRecord(evdownloaded, 0));

    datadump_idtimestep = idtimestep;
    datadump_nsolvent = particles->size;
    datadump_nrbcs = rbcscoll ? rbcscoll->pcount() : 0;
    datadump_pending = true;

    pthread_cond_signal(&request_datadump);
#if defined(_SYNC_DUMPS_)
    while (datadump_pending)
        pthread_cond_wait(&done_datadump, &mutex_datadump);
#endif

    pthread_mutex_unlock(&mutex_datadump);
}

void Simulation::_datadump_async()
{
    int iddatadump = 0, rank;
    int curr_idtimestep = -1;

    MPI_Comm myactivecomm, mycartcomm;

    MPI_CHECK(MPI_Comm_dup(activecomm, &myactivecomm) );
    MPI_CHECK(MPI_Comm_dup(cartcomm, &mycartcomm) );

    printf("ALLPARTICLES CHECK\n");
    H5PartDump dump_part("allparticles.h5part", activecomm, cartcomm), *dump_part_solvent = NULL;
    H5FieldDump dump_field(cartcomm);

    MPI_CHECK(MPI_Comm_rank(myactivecomm, &rank));
    MPI_CHECK(MPI_Barrier(myactivecomm));

    while (true)
    {
        pthread_mutex_lock(&mutex_datadump);
        async_thread_initialized = 1;

        while (!datadump_pending)
            pthread_cond_wait(&request_datadump, &mutex_datadump);

        pthread_mutex_unlock(&mutex_datadump);

        if (curr_idtimestep == datadump_idtimestep)
            if (simulation_is_done)
                break;

        CUDA_CHECK(cudaEventSynchronize(evdownloaded));

        const int n = particles_datadump.size;
        Particle * p = particles_datadump.data;
        Acceleration * a = accelerations_datadump.data;

        {
            diagnostics(myactivecomm, mycartcomm, p, n, dt, datadump_idtimestep, a);
        }

        if (hdf5part_dumps)
        {
            if (!dump_part_solvent && walls && datadump_idtimestep >= wall_creation_stepid)
            {
                dump_part.close();

                dump_part_solvent = new H5PartDump("solvent-particles->h5part", activecomm, cartcomm);
            }

            if (dump_part_solvent)
                dump_part_solvent->dump(p, n);
            else
                dump_part.dump(p, n);
        }

        if (hdf5field_dumps && (datadump_idtimestep % steps_per_hdf5dump == 0))
        {
            dump_field.dump(activecomm, p, datadump_nsolvent, datadump_idtimestep);
        }

        /* LINA: this is to not to dump the beginning
           if (datadump_idtimestep >= 600/dt) */
        {
            if (rbcscoll)
                CollectionRBC::dump(myactivecomm, mycartcomm, p + datadump_nsolvent, a + datadump_nsolvent, datadump_nrbcs, iddatadump);
        }

        curr_idtimestep = datadump_idtimestep;

        pthread_mutex_lock(&mutex_datadump);

        if (simulation_is_done)
        {
            pthread_mutex_unlock(&mutex_datadump);
            break;
        }

        datadump_pending = false;

        pthread_cond_signal(&done_datadump);

        pthread_mutex_unlock(&mutex_datadump);

        ++iddatadump;
    }

    if (dump_part_solvent)
        delete dump_part_solvent;

    CUDA_CHECK(cudaEventDestroy(evdownloaded));
}

void Simulation::_update_and_bounce()
{
    double tstart = MPI_Wtime();
    particles->update_stage2_and_1(1, driving_acceleration, mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    if (rbcscoll)
        rbcscoll->update_stage2_and_1(rbc_mass, driving_acceleration, mainstream);

    CUDA_CHECK(cudaPeekAtLastError());
    if (wall)
    {
        tstart = MPI_Wtime();
        wall->bounce(particles->xyzuvw.data, particles->size, mainstream);

        if (rbcscoll)
            wall->bounce(rbcscoll->data(), rbcscoll->pcount(), mainstream);
    }

    CUDA_CHECK(cudaPeekAtLastError());
}

Simulation::Simulation(MPI_Comm cartcomm, MPI_Comm activecomm, bool (*check_termination)()) :
    cartcomm(cartcomm), activecomm(activecomm),
    cells(XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN),
    rbcscoll(NULL), wall(NULL),
    redistribute(cartcomm),  redistribute_rbcs(cartcomm),
    dpd(cartcomm), fsi(cartcomm), contact(cartcomm), solutex(cartcomm),
    check_termination(check_termination),
    driving_acceleration(0), host_idle_time(0), nsteps((int)(tend / dt)),
    datadump_pending(false), simulation_is_done(false)
{
    MPI_CHECK( MPI_Comm_size(activecomm, &nranks) );
    MPI_CHECK( MPI_Comm_rank(activecomm, &rank) );

    solutex.attach_halocomputation(fsi);

    if (contactforces)
        solutex.attach_halocomputation(contact);

    int dims[3], periods[3], coords[3];
    MPI_CHECK( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );

    int xl1[3] = {0, 0, 3};
    int xh1[3] = {XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, 8};
    velcont1 = new VelController(xl1, xh1, coords,
            make_float3(-desired_shrate*(ZSIZE_SUBDOMAIN/2-8), 0, 0), activecomm);

    int xl2[3] = {0, 0, ZSIZE_SUBDOMAIN-8};
    int xh2[3] = {XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN-3};
    velcont2 = new VelController(xl2, xh2, coords,
            make_float3(desired_shrate*(ZSIZE_SUBDOMAIN/2-8), 0, 0), activecomm);

    {
        particles = &particles_pingpong[0];
        newparticles = &particles_pingpong[1];

        vector<Particle> ic = _ic();

        for(int c = 0; c < 2; ++c)
        {
            particles_pingpong[c].resize(ic.size());

            particles_pingpong[c].origin = make_float3((0.5 + coords[0]) * XSIZE_SUBDOMAIN,
                    (0.5 + coords[1]) * YSIZE_SUBDOMAIN,
                    (0.5 + coords[2]) * ZSIZE_SUBDOMAIN);

            particles_pingpong[c].globalextent = make_float3(dims[0] * XSIZE_SUBDOMAIN,
                    dims[1] * YSIZE_SUBDOMAIN,
                    dims[2] * ZSIZE_SUBDOMAIN);
        }

        CUDA_CHECK(cudaMemcpy(particles->xyzuvw.data, &ic.front(), sizeof(Particle) * ic.size(), cudaMemcpyHostToDevice));

        cells.build(particles->xyzuvw.data, particles->size, 0, NULL, NULL);

        _update_helper_arrays();
    }

    CUDA_CHECK(cudaStreamCreate(&mainstream));
    CUDA_CHECK(cudaStreamCreate(&uploadstream));
    CUDA_CHECK(cudaStreamCreate(&downloadstream));

    if (rbcs)
    {
        rbcscoll = new CollectionRBC(cartcomm);
        rbcscoll->setup("rbcs-ic.txt");
    }

#ifndef _NO_DUMPS_
    //setting up the asynchronous data dumps
    {
        CUDA_CHECK(cudaEventCreate(&evdownloaded, cudaEventDisableTiming | cudaEventBlockingSync));

        particles_datadump.resize(particles->size * 1.5);
        accelerations_datadump.resize(particles->size * 1.5);

        int rc = pthread_mutex_init(&mutex_datadump, NULL);
        rc |= pthread_cond_init(&done_datadump, NULL);
        rc |= pthread_cond_init(&request_datadump, NULL);
        async_thread_initialized = 0;
        rc |= pthread_create(&thread_datadump, NULL, datadump_trampoline, this);

        while (1)
        {
            pthread_mutex_lock(&mutex_datadump);
            int done = async_thread_initialized;
            pthread_mutex_unlock(&mutex_datadump);

            if (done)
                break;
        }

        if (rc)
        {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }
#endif
}

void Simulation::_lockstep()
{
    double tstart = MPI_Wtime();

    SolventWrap wsolvent(particles->xyzuvw.data, particles->size, particles->axayaz.data, cells.start, cells.count);

    std::vector<ParticlesWrap> wsolutes;

    if (rbcscoll)
        wsolutes.push_back(ParticlesWrap(rbcscoll->data(), rbcscoll->pcount(), rbcscoll->acc()));

    fsi.bind_solvent(wsolvent);

    solutex.bind_solutes(wsolutes);

    particles->clear_acc(mainstream);

    if (rbcscoll)
        rbcscoll->clear_acc(mainstream);

    solutex.pack_p(mainstream);

    dpd.pack(particles->xyzuvw.data, particles->size, cells.start, cells.count, mainstream);

    dpd.local_interactions(particles->xyzuvw.data, xyzouvwo.data, xyzo_half.data, particles->size, particles->axayaz.data,
            cells.start, cells.count, mainstream);
    if (!doublepoiseuille) {
        velcont1->push(cells.start, particles->xyzuvw.data, particles->axayaz.data, mainstream);
        velcont2->push(cells.start, particles->xyzuvw.data, particles->axayaz.data, mainstream);
    }

    if (contactforces)
        contact.build_cells(wsolutes, mainstream);

    solutex.post_p(mainstream, downloadstream);

    dpd.post(particles->xyzuvw.data, particles->size, mainstream, downloadstream);

    CUDA_CHECK(cudaPeekAtLastError());

    if (wall)
        wall->interactions(particles->xyzuvw.data, particles->size, particles->axayaz.data,
                cells.start, cells.count, mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    dpd.recv(mainstream, uploadstream);

    solutex.recv_p(uploadstream);

    solutex.halo(uploadstream, mainstream);

    dpd.remote_interactions(particles->xyzuvw.data, particles->size, particles->axayaz.data, mainstream, uploadstream);

    fsi.bulk(wsolutes, mainstream);

    if (contactforces)
        contact.bulk(wsolutes, mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    if (rbcscoll)
        CudaRBC::forces_nohost(mainstream, rbcscoll->count(), (float *)rbcscoll->data(), (float *)rbcscoll->acc());

    CUDA_CHECK(cudaPeekAtLastError());

    solutex.post_a();

    particles->update_stage2_and_1(1, driving_acceleration, mainstream);

    if (wall)
        wall->bounce(particles->xyzuvw.data, particles->size, mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    redistribute.pack(particles->xyzuvw.data, particles->size, mainstream);

    redistribute.send();

    redistribute.bulk(particles->size, cells.start, cells.count, mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    if (rbcscoll && wall)
        wall->interactions(rbcscoll->data(), rbcscoll->pcount(), rbcscoll->acc(), NULL, NULL, mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    solutex.recv_a(mainstream);

    if (rbcscoll)
        rbcscoll->update_stage2_and_1(rbc_mass, driving_acceleration, mainstream);


    const int newnp = redistribute.recv_count(mainstream, host_idle_time);

    CUDA_CHECK(cudaPeekAtLastError());

    if (rbcscoll)
        redistribute_rbcs.extent(rbcscoll->data(), rbcscoll->count(), mainstream);

    if (rbcscoll)
        redistribute_rbcs.pack_sendcount(rbcscoll->data(), rbcscoll->count(), mainstream);

    newparticles->resize(newnp);
    xyzouvwo.resize(newnp * 2);
    xyzo_half.resize(newnp);

    redistribute.recv_unpack(newparticles->xyzuvw.data, xyzouvwo.data, xyzo_half.data, newnp, cells.start, cells.count, mainstream, host_idle_time);

    CUDA_CHECK(cudaPeekAtLastError());

    swap(particles, newparticles);

    int nrbcs;
    if (rbcscoll)
        nrbcs = redistribute_rbcs.post();

    if (rbcscoll)
        rbcscoll->resize(nrbcs);

    CUDA_CHECK(cudaPeekAtLastError());

    if (rbcscoll)
        redistribute_rbcs.unpack(rbcscoll->data(), rbcscoll->count(), mainstream);

    CUDA_CHECK(cudaPeekAtLastError());
}


void Simulation::run()
{
    if (rank == 0 && !walls)
        printf("the simulation begins now and it consists of %.3e steps\n", (double)nsteps);


    _redistribute();
    _forces();

    if (!walls && pushtheflow)
        driving_acceleration = hydrostatic_a;

    particles->update_stage1(1, driving_acceleration, mainstream);

    if (rbcscoll)
        rbcscoll->update_stage1(rbc_mass, driving_acceleration, mainstream);

    int it;


    for(it = 0; it < nsteps; ++it)
    {
        const bool verbose = it > 0 && rank == 0;
        _redistribute();

lockstep_check:

        const bool lockstep_OK =
            !(walls && it >= wall_creation_stepid && wall == NULL) &&
            !(it % steps_per_dump == 0) &&
            !(it + 1 == nsteps);

        if (lockstep_OK)
        {
            _lockstep();
            {
                if (!doublepoiseuille) {
                    if (it % 10 == 0)
                    {
                        velcont1->sample(cells.start, particles->xyzuvw.data, mainstream);
                        velcont2->sample(cells.start, particles->xyzuvw.data, mainstream);
                    }

                    if (it % 500 == 0)
                    {
                        velcont1->adjustF(mainstream);
                        velcont2->adjustF(mainstream);
                        driving_acceleration = 0;
                    }
                }
            }

            ++it;

            goto lockstep_check;
        }

        if (walls && it >= wall_creation_stepid && wall == NULL)
        {
            CUDA_CHECK(cudaDeviceSynchronize());

            bool termination_request = false;

            _create_walls(verbose, termination_request);

            _redistribute();

            if (termination_request)
                break;


            if (pushtheflow)
                driving_acceleration = hydrostatic_a;

            if (rank == 0)
                printf("the simulation begins now and it consists of %.3e steps\n", (double)(nsteps - it));
        }

        _forces();

#ifndef _NO_DUMPS_
        if (it % steps_per_dump == 0)
            _datadump(it);
#endif
        _update_and_bounce();
        {
            if (!doublepoiseuille) {
                if (it % 10 == 0)
                {
                    velcont1->sample(cells.start, particles->xyzuvw.data, mainstream);
                    velcont2->sample(cells.start, particles->xyzuvw.data, mainstream);
                }

                if (it % 500 == 0)
                {
                    velcont1->adjustF(mainstream);
                    velcont2->adjustF(mainstream);
                    driving_acceleration = 0;
                }
            }
        }
    }


    simulation_is_done = true;
    fflush(stdout);
}

Simulation::~Simulation()
{
#ifndef _NO_DUMPS_
    pthread_mutex_lock(&mutex_datadump);

    datadump_pending = true;
    pthread_cond_signal(&request_datadump);

    pthread_mutex_unlock(&mutex_datadump);

    pthread_join(thread_datadump, NULL);
#endif

    CUDA_CHECK(cudaStreamDestroy(mainstream));
    CUDA_CHECK(cudaStreamDestroy(uploadstream));
    CUDA_CHECK(cudaStreamDestroy(downloadstream));

    if (wall)
        delete wall;

    if (rbcscoll)
        delete rbcscoll;
}
