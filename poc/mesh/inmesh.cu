#include "mesh.h"
#include "collision.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

#define CC(ans)                                             \
    do { cudaAssert((ans), __FILE__, __LINE__); } while (0)
inline void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPU assert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        abort();
    }
}

#define DEVICE

#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost

// #define VANILLA
// #define SHARED
// #define TEXTURE
#define TEXTURE_SHARED

#include "tex.h"

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <N> <file.ply>\n", argv[0]);
        exit(1);
    }

    const int N = atoi(argv[1]);
    
    srand48(123456);
    
    std::vector<int> tt;
    std::vector<float> vv;

    mesh::read_ply(argv[2], tt, vv);

    const int nv = vv.size() / 3;
    const int nt = tt.size() / 3;
    
    // generate points
    
    float xlo, xhi, ylo, yhi, zlo, zhi;

    xlo = xhi = vv[0];
    ylo = yhi = vv[1];
    zlo = zhi = vv[2];
    
    for (int i = 0; i < nv; ++i)
    {
        const float x = vv[3*i + 0], y = vv[3*i + 1], z = vv[3*i + 2];

#define highest(a, b) do {a = a < b ? b : a; } while(0)
#define  lowest(a, b) do {a = a < b ? a : b; } while(0)

        lowest(xlo, x); highest(xhi, x);
        lowest(ylo, y); highest(yhi, y);
        lowest(zlo, z); highest(zhi, z);
    }

    printf("Extents: %f %f, %f %f, %f %f\n", xlo, xhi, ylo, yhi, zlo, zhi);

    float *rr  = new float[3*N];
    int *inout = new int[N];

    for (int i = 0; i < N; ++i)
    {
        rr[i*3 + 0] = xlo + drand48() * (xhi - xlo);
        rr[i*3 + 1] = ylo + drand48() * (yhi - ylo);
        rr[i*3 + 2] = zlo + drand48() * (zhi - zlo);
    }
    
    // compute inout

#ifdef DEVICE

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float *d_vv = NULL, *d_rr = NULL; int *d_tt = NULL, *d_inout = NULL;
    CC(cudaMalloc(&d_vv, 3 * nv * sizeof(float)));
    CC(cudaMalloc(&d_tt, 3 * nt * sizeof(int)));
    CC(cudaMalloc(&d_rr, 3 * N  * sizeof(float)));
    CC(cudaMalloc(&d_inout, N   * sizeof(int)));
    
    CC(cudaMemcpy(d_vv, vv.data(), 3 * nv * sizeof(float), H2D));
    CC(cudaMemcpy(d_tt, tt.data(), 3 * nt * sizeof(int),   H2D));
    CC(cudaMemcpy(d_rr, rr, 3 * N * sizeof(float), H2D));

#if defined( TEXTURE ) || defined( TEXTURE_SHARED )
    float4 *vvzip; int4 *ttzip;
    CC(cudaMalloc(&vvzip, nv * sizeof(float4)));
    CC(cudaMalloc(&ttzip, nt * sizeof(int4)));

    tex::zip4(d_tt, nt, /**/ ttzip);
    tex::zip4(d_vv, nv, /**/ vvzip);

    cudaTextureObject_t ttto, vvto;

    tex::maketexzip(ttzip, nt, /**/ &ttto);
    tex::maketexzip(vvzip, nv, /**/ &vvto);
    
#endif
    
    cudaEventRecord(start);

#if defined( VANILLA )
    collision::in_mesh_dev(d_rr, N, d_vv, nv, d_tt, nt, /**/ d_inout);
#elif defined( SHARED )
    collision::in_mesh_dev_shared(d_rr, N, d_vv, nv, d_tt, nt, /**/ d_inout);
#elif defined( TEXTURE )
    collision::in_mesh_dev_tex(d_rr, N, vvto, nv, ttto, nt, /**/ d_inout);
#elif defined( TEXTURE_SHARED )
    collision::in_mesh_dev_tex_shared(d_rr, N, vvto, nv, ttto, nt, /**/ d_inout);
#endif

    cudaEventRecord(stop);
    
    CC(cudaMemcpy(inout, d_inout, N * sizeof(int), D2H));
    
    CC(cudaFree(d_vv)); CC(cudaFree(d_tt));
    CC(cudaFree(d_rr)); CC(cudaFree(d_inout));

#if defined( TEXTURE ) || defined( TEXTURE_SHARED )
    CC(cudaFree(vvzip)); CC(cudaFree(ttzip));
#endif
    
    cudaEventSynchronize(stop);
    float tms = 0;
    cudaEventElapsedTime(&tms, start, stop);
    fprintf(stderr, "Took %f ms for %d particles\n", tms, N);
#else
    collision::in_mesh(rr, N, vv.data(), tt.data(), nt, /**/ inout);
#endif
    
    // dump
    
    FILE *fin = fopen("parts_in.3D", "w");
    FILE *fout = fopen("parts_out.3D", "w");

    fprintf(fin, "x y z inout\n");
    fprintf(fout, "x y z inout\n");
    
    for (int i = 0; i < N; ++i)
    fprintf(inout[i] == -1 ? fout : fin, "%.6e %.6e %.6e %e\n", rr[3*i + 0], rr[3*i + 1], rr[3*i + 2], inout[i] == -1 ? 1.f : 0.f);

    delete[] inout;
    delete[] rr;
    
    fclose(fin);
    fclose(fout);
}

/*

# nTEST: collision.t0
# make clean && make -j
# ./inmesh 10000 data/cow.ply
# cat parts_in.3D | sed -n '2,10000p' > parts.out.3D

# nTEST: collision.t1
# make clean && make -j
# ./inmesh 10000 data/sphere.ply
# cat parts_in.3D | sed -n '2,10000p' > parts.out.3D

*/
