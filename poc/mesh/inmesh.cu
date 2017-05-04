#include "mesh.h"
#include "collision.h"

#define N 10000

int main(int argc, char **argv)
{
    srand48(123456);
    
    std::vector<int> tt;
    std::vector<float> vv;

    mesh::read_ply(argv[1], tt, vv);

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

    float rr[3*N];
    int inout[N];

    for (int i = 0; i < N; ++i)
    {
        rr[i*3 + 0] = xlo + drand48() * (xhi - xlo);
        rr[i*3 + 1] = ylo + drand48() * (yhi - ylo);
        rr[i*3 + 2] = zlo + drand48() * (zhi - zlo);
    }
    
    // compute inout

    collision::in_mesh(rr, N, vv.data(), tt.data(), nt, /**/ inout);

    // dump
    
    FILE *fin = fopen("parts_in.3D", "w");
    FILE *fout = fopen("parts_out.3D", "w");

    fprintf(fin, "x y z inout\n");
    fprintf(fout, "x y z inout\n");
    
    for (int i = 0; i < N; ++i)
    fprintf(inout[i] ? fout : fin, "%.6e %.6e %.6e %e\n", rr[3*i + 0], rr[3*i + 1], rr[3*i + 2], (float) inout[i]);
    
    fclose(fin);
    fclose(fout);
}

/*

# nTEST: collision.t0
# make clean && make -j
# ./inmesh data/cow.ply
# cat parts_in.3D | sed -n '2,10000p' > parts.out.3D

*/
