#include "mesh.h"

int main(int argc, char **argv)
{
    std::vector<int> tt;
    std::vector<float> vv;

    mesh::read_ply(argv[2], tt, vv);
    
    mesh::write_ply(argv[1], tt.data(), tt.size()/3, vv.data(), vv.size()/3);
}
/*

# nTEST: ply.t0
# make clean && make -j
# ./ply_io test.out.ply data/cow.ply

*/
