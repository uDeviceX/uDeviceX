#include <cstdio>
#include <cstdlib>
#include "reader.h"

int main(int argc, char **argv) {

    if (argc < 4) {
        fprintf(stderr, "Usage: po.diffCoeff <inpp-*.bop> -- <inii-*.bop>\n");
        exit(1);
    }
    
    ReadData dpp, dii;    
    init(&dpp); init(&dii);

    
    
    finalize(&dpp); finalize(&dii);
    
    return 0;
}
