#include <stdio.h>
#include <assert.h>
#include "frag/common.h"
#include "frag/to.h"
int main() {
    enum {X, Y, Z};
    #include "2.h"
}
