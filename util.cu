#include <stdio.h>
#include <string.h>
#include <ctype.h>

enum {MAX_STRING_SIZE = 4096};

int util_eq(const char *a, const char *b) {
    return strncmp(a, b, MAX_STRING_SIZE) == 0;
}

char *util_fgets(char *s, FILE *stream) {
    char *c;
    if (fgets(s, MAX_STRING_SIZE, stream) == NULL)
        return NULL;
    if ((c = strchr(s, '\n')) != NULL)
        *c = '\0';
    return s;
}

enum {NO, YES};
static int commentp(const char *s) {
    while (*(s++) != '\0') {
        if (*s == '#') return YES;
        if (!isblank(*s)) return NO;
    }
    return YES;
}

char *util_comment_fgets(char *s, FILE *stream) {
    do if (util_fgets(s, stream) == NULL)
           return NULL;
    while (commentp(s) == YES);
    return s;
}
