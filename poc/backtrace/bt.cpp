#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>

#define PRINT_TRACE() do {                      \
    void *array[10];                            \
    size_t size, i;                             \
    char **strings;                             \
                                                \
    size = backtrace (array, 10);               \
    strings = backtrace_symbols (array, size);  \
                                                \
    for (i = 0; i < size; i++)                  \
        printf ("%s\n", strings[i]);            \
                                                \
    free (strings);                             \
    } while(0)


/* A dummy function to make the backtrace more interesting. */
void foo (void) {
    PRINT_TRACE();
}

void bar (void) {
    foo();
}

int main (void) {
  bar();
  return 0;
}
