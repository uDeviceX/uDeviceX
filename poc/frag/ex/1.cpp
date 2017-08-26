#include <assert.h>
#include "frag/common.h"
#include "frag/to.h"
#include "frag/fro.h"

int main() {
    enum {X, Y, Z};
    #include "1.h"
}
