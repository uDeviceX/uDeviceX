#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ply.h"
#include "mesh.h"

using std::vector;

int main(int argc, char **argv)
{
    vector<int> tt;
    vector<float> vv;
    icosahedron(tt, vv);

    write_ply("test.ply", tt, vv);
    
    return 0;
}
