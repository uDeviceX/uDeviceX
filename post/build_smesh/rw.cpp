#include <cstdio>
#include <cstdlib>
#include "ply.h"

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "usage: %s <in.ply> <out.ply>\n", argv[0]);
        exit(1);
    }
    
    std::vector<float> vv;
    std::vector<int> tt;

    read_ply(argv[1], tt, vv);
    write_ply(argv[2], tt, vv);
    
    return 0;
}
