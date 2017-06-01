#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#include "ply.h"
#include "mesh.h"

using std::vector;

void map2disk(int n, float *vv)
{
    for (int i = 0; i < n; ++i)
    {
        float *r = vv + 3*i;
        const float x = r[0];
        const float y = r[1];
        
        float theta = atan2(y, x);
        theta += M_PI/6;
        theta = theta > 0 ? theta : 2*M_PI + theta;
        
        const float T = 2.0*M_PI/3.0;
        if (theta > T) theta -= T;
        if (theta > T) theta -= T;

        const float s = 2*cos(theta - M_PI/3);

        r[0] *= s;
        r[1] *= s;
    }
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "usage : %s <out.ply>\n", argv[0]);
        exit(1);
    }
    
    vector<int> tt;
    vector<float> vv;

#define pp push_back
#define pp3(VV, a, b, c) do{VV.pp(a); VV.pp(b); VV.pp(c);} while(0)

    pp3(vv, 0, 1, 0);
    pp3(vv, +0.5*sqrt(3), -0.5, 0);
    pp3(vv, -0.5*sqrt(3), -0.5, 0);    
#undef pp3
#undef pp

    tt.push_back(0); tt.push_back(1); tt.push_back(2);

    for (int i = 0; i < 3; ++i)
    subdivide2(tt, vv);

    map2disk(vv.size()/3, vv.data());

    int c = 1;
    while(c != 0)
    {
        c = flip_edges(tt, vv);
        printf("flipped %d edges\n", c);
    }
    
    write_ply(argv[1], tt, vv);
    
    return 0;
}
