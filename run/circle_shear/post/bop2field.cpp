#include <cstdlib>
#include <cstdio>

#include "reader.h"

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        fprintf(stderr, "usage: %s <out.txt> <in1.bop> <in2.bop> ...\n", argv[0]);
        exit(1);
    }

    ReadData d;
    init(&d);
    {
        const int nd = argc-2;
        ReadData *dd = new ReadData[nd];

        for (int i = 0; i < nd; ++i)
        {
            init(dd + i);
            read(argv[2+i], dd + i);
        }

        concatenate(nd, dd, /**/ &d);
        for (int i = 0; i < nd; ++i) finalize(dd + i);
        delete[] dd;
    }
    summary(&d);

    FILE *f = fopen(argv[1], "w");
    
    fclose(f);

    finalize(&d);
    
    return 0;
}
