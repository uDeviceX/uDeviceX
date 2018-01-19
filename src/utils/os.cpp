#include <stdio.h>
#include <stdlib.h>

#include <sys/stat.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>

#include "msg.h"
#include "utils/error.h"

#include "os.h"

void os_mkdir(const char *path) {
    mode_t mode;
    int rc, ok;
    mode = S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH;
    rc = ::mkdir(path, mode);
    ok = (rc == 0 || errno == EEXIST);
    if (!ok) {
        msg_print("os_mkdir: cannot create directory ‘%s’", path);
        ERR("errno: %d\n", errno);
        exit(1);
    }
}

long os_time() { return time(NULL); }
void os_srand(long int seedval) { srand48(seedval); }
double os_drand() { return drand48(); }
void os_sleep(unsigned int seconds) { sleep(seconds); }
