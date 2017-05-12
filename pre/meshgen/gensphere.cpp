#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#include "ply.h"

using std::vector;

void icosahedron(vector<int>& tt, vector<float>& vv)
{
    const float phi = 0.5 * (1. + sqrt(5.));
    
    tt.clear(); vv.clear();

#define pp push_back
#define pp3(VV, a, b, c) do{VV.pp(a); VV.pp(b); VV.pp(c);} while(0)
    
    pp3(vv, -1, phi ,0);
    pp3(vv, 1, phi, 0);
    pp3(vv, -1, -phi, 0);
    pp3(vv, 1, -phi, 0);

    pp3(vv, 0, -1, phi);
    pp3(vv, 0, 1, phi);
    pp3(vv, 0, -1, -phi);
    pp3(vv, 0, 1, -phi);

    pp3(vv, phi, 0, -1);
    pp3(vv, phi, 0, 1);
    pp3(vv, -phi, 0, -1);
    pp3(vv, -phi, 0, 1);

    pp3(tt, 0, 11, 5);
    pp3(tt, 0, 5, 1);
    pp3(tt, 0, 1, 7);
    pp3(tt, 0, 7, 10);
    pp3(tt, 0, 10, 11);
    
    pp3(tt, 1, 5, 9);
    pp3(tt, 5, 11, 4);
    pp3(tt, 11, 10, 2);
    pp3(tt, 10, 7, 6);
    pp3(tt, 7, 1, 8);
    
    pp3(tt, 3, 9, 4);
    pp3(tt, 3, 4, 2);
    pp3(tt, 3, 2, 6);
    pp3(tt, 3, 6, 8);
    pp3(tt, 3, 8, 9);
    
    pp3(tt, 4, 9, 5);
    pp3(tt, 2, 4, 11);
    pp3(tt, 6, 2, 10);
    pp3(tt, 8, 6, 7);
    pp3(tt, 9, 8, 1);

#undef pp3
#undef pp
}

int main(int argc, char **argv)
{
    vector<int> tt;
    vector<float> vv;
    icosahedron(tt, vv);

    write_ply("test.ply", tt, vv);
    
    return 0;
}
