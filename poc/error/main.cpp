#include "h.h"

#define EC(f) \
    do {                                        \
        f;                                      \
        if (status() != 0) {                    \
            line(__LINE__);                     \
            file(__FILE__);                     \
        }                                       \
    } while (0)

int main() {
    EC(fun());
    extra("d: %d", 42);
    format();
}
