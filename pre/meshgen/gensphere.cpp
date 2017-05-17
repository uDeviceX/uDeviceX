#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ply.h"
#include "mesh.h"

using std::vector;

int main(int argc, char **argv)
{
    if (argc != 5)
    {
        fprintf(stderr, "usage : %s <out.ply> <R> <nsub2> <nsub3>\n", argv[0]);
        exit(1);
    }
    
    vector<int> tt;
    vector<float> vv;
    icosahedron(tt, vv);

    const float R = atof(argv[2]);
    const int n2 = atoi(argv[3]);
    const int n3 = atoi(argv[4]);

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

    scale(vv, R);
    
    write_ply(argv[1], tt, vv);
    
    return 0;
}
