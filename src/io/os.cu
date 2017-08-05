#include <sys/stat.h>
#include <stdio.h>
#include "io/os.h"

namespace os {
void mkdir(const char *path) {
    mode_t mode;
    int rc;
    mode = S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH;
    rc = ::mkdir(path, mode);

    if (!rc) {
        fprintf(stderr, "os.cu: mkdir: cannot create directory ‘%s’\n", path);
        exit(1);
    }
}
}
