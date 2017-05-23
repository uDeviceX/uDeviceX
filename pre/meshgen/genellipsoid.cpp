#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ply.h"
#include "mesh.h"

using std::vector;

int main(int argc, char **argv)
{
    if (argc != 7)
    {
        fprintf(stderr, "usage : %s <out.ply> <a> <b> <c> <nsub2> <nsub3>\n", argv[0]);
        exit(1);
    }
    
    vector<int> tt;
    vector<float> vv;
    icosahedron(tt, vv);

    const float a = atof(argv[2]);
    const float b = atof(argv[3]);
    const float c = atof(argv[4]);
    const int n2 = atoi(argv[5]);
    const int n3 = atoi(argv[6]);

    for (int i = 0; i < n2; ++i)
    {
        subdivide2(tt, vv);
        scale_to_usphere(vv);
    }

    for (int i = 0; i < n3; ++i)
    {
        subdivide3(tt, vv);
        scale_to_usphere(vv);
    }

    for (uint i = 0; i < vv.size()/3; ++i)
    {
        float *x = vv.data() + 3*i;
        x[0] *= a;
        x[1] *= b;
        x[2] *= c;
    }
    
    write_ply(argv[1], tt, vv);
    
    return 0;
}
