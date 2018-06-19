#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <malloc.h>

#include "imp.h"
#include "utils/msg.h"
#include "utils/error.h"

// static size_t tot = 0;

static void format_bytes(size_t sz, char *s) {
    int GB, MB, kB, B;
    GB = sz >> 30; sz -= GB << 30;
    MB = sz >> 20; sz -= MB << 20;
    kB = sz >> 10; sz -= kB << 10;
    B  = sz;
    sprintf(s, "%d GB + %d MB + %d kB + %d B", GB, MB, kB, B);
}

// static void log_size(size_t sz) {
//     char s[FILENAME_MAX], t[FILENAME_MAX];
//     if (sz < (1 << 20)) return;
//     format_bytes(sz, s);
//     format_bytes(tot, t);
//     msg_print("allocate %s -> total %s", s, t);
    
//     // error_print_stack();
// }
void emalloc(size_t size, /**/ void **data) {
    *data = malloc(size);
    // tot += size;
    // log_size(size);
    if (NULL == *data) {
        char s[FILENAME_MAX];
        format_bytes(size, s);
        msg_print("standard library: %s", strerror(errno));
        malloc_stats();
        ERR("Failed to allocate array of size %s\n", s);
    }
}

void efree(void *ptr) {
    free(ptr);
    ptr = NULL;
}

void *ememcpy(void *dest, const void *src, size_t n) {
    if (NULL == dest) ERR("NULL == dest");
    if (NULL == src) ERR("NULL == src");
    return memcpy(dest, src, n);
}

void efopen(const char *fname, const char *mode, /**/ FILE **pq) {
    FILE *q;
    q = fopen(fname, mode);
    if (NULL == q)
        ERR("Could not open file <%s> with mode <%s>", fname, mode);
    *pq = q;
}

void efclose(FILE *f) {
    if (fclose(f) != 0) ERR("Failed to close");
}

void efread(void *ptr, size_t size, size_t nmemb, FILE *stream) {
    size_t nmemb0;
    nmemb0 = fread(ptr, size, nmemb, stream);
    if (nmemb != nmemb0)
        ERR("`fread` failed: nmemb0=%ld   !=    nmemb=%lds, size=%ld",
            nmemb0, nmemb, size);
}

void efwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream) {
    size_t nmemb0;
    nmemb0 = fwrite(ptr, size, nmemb, stream);
    if (nmemb != nmemb0)
        ERR("`fwrite` failed: nmemb0=%ld   !=    nmemb=%lds, size=%ld",
            nmemb0, nmemb, size);
}

void efgets(char *s, int size, FILE *stream) {
    char *p;
    p = fgets(s, size, stream);
    if (p == NULL)
        ERR("`fgets` failed: size=%d", size);
}

bool same_str(const char *a, const char *b) {
    return 0 == strcmp(a, b);
}

const char* get_filename_ext(const char *filename) {
    const char *dot = strrchr(filename, '.');
    if (!dot || dot == filename) return "";
    return dot + 1;
}
