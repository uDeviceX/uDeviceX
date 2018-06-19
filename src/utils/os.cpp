#include <stdio.h>
#include <stdlib.h>

#include <sys/stat.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#include <malloc.h>

#include "msg.h"
#include "utils/error.h"

#include "os.h"

static void safe_mkdir(const char *path)  {
    mode_t mode;
    int rc, ok;
    mode = S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH;
    rc = mkdir(path, mode);
    ok = (rc == 0 || errno == EEXIST);
    if (!ok) {
        msg_print("os_mkdir: cannot create directory ‘%s’", path);
        ERR("errno: %d\n", errno);
        exit(1);
    }
}

#define END '\0'
#define SEP '/'

void os_mkdir(const char *path0) {
    char path[FILENAME_MAX];
    unsigned int len;
    char *p;

    len = snprintf(path, sizeof(path), "%s", path0);

    if (len >= sizeof(path)) ERR("Path is too long: %s", path0);
    if (len == 0) ERR("empty path");

    if (path[len-1] == SEP) path[len-1] = END;

    for (p = path; *p != END; ++p) {
        if (*p == SEP) {
            *p = END;
            UC(safe_mkdir(path));
            *p = SEP;
        }
    }
    UC(safe_mkdir(path));
}

#undef SEP
#undef END

long os_time() { return time(NULL); }
void os_srand(long int seedval) { srand48(seedval); }
double os_drand() { return drand48(); }
void os_sleep(unsigned int seconds) { sleep(seconds); }
void os_malloc_stats() { os_malloc_stats(); }
