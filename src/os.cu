#include <sys/stat.h>
#include <stdio.h>
#include <errno.h>
#include "os.h"

namespace os {
void mkdir(const char *path) {
    mode_t mode;
    int rc, ok;
    mode = S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH;
    rc = ::mkdir(path, mode);
    ok = (rc == 0 || rc == EEXIST);
    if (!ok) {
        fprintf(stderr, "os.cu: mkdir: cannot create directory ‘%s’\n", path);
        fprintf(stderr, "errno: %d\n", errno);
        exit(1);
    }
}
}
