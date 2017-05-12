#include <cstdio>
#include <cstdlib>
#include <cassert>
#include "ply.h"

using std::vector;

void write_ply(const char *fname, const std::vector<int>& tt, const std::vector<float>& vv)
{
    const int nt = tt.size() / 3;
    const int nv = vv.size() / 3;

    FILE * f = fopen(fname, "w");

    assert(f != NULL);
    
    fprintf(f, "ply\n");
    fprintf(f, "format ascii 1.0\n");
    fprintf(f, "element vertex %d\n", nv);
    fprintf(f, "property float x\n");
    fprintf(f, "property float y\n");
    fprintf(f, "property float z\n");
    fprintf(f, "element face %d\n", nt);
    fprintf(f, "property list int int vertex_index\n");
    fprintf(f, "end_header\n");
    
    for (int i = 0; i < nv; ++i)
    fprintf(f, "%f %f %f\n", vv[3*i + 0], vv[3*i + 1], vv[3*i + 2]);
    
    for (int i = 0; i < nt; ++i)
    fprintf(f, "3 %d %d %d\n", tt[3*i + 0], tt[3*i + 1], tt[3*i + 2]);
    
    fclose(f);
}
