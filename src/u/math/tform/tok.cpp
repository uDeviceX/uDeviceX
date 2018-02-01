#include <string.h>
#include <stdio.h>

#include "utils/error.h"
#include "utils/imp.h"

void tok_cat(int n, char **ss, /**/ char *a) {
    int i;
    char *s;
    a[0] = '\0';
    for (i = 0; i < n; ++i) {
        s = ss[i];
        if (i > 0) strcat(a, " ");
        strcat(a, s);
    }
}

int cnt0(char *s, const char *del) {
    char *tok;
    int c;
    tok = s;
    c = 0;
    while ((tok = strtok(tok, del)) != NULL) {
        c++;
        tok = NULL;
    }
    return c;
}

int cnt(const char *s0, const char *del) {
    char *s;
    int c;
    s = strdup(s0);
    c = cnt0(s, del);
    EFREE(s);
    return c;
}

void tok_ini0(char *s, char *del, /**/ int *pc, char ***pv) {
    int i, c;
    char *tok;
    char **v;
    c = cnt(s, del);
    EMALLOC(c, &v);
    tok = s; i = 0;
    while ((tok = strtok(tok, del)) != NULL) {
        v[i++] = strdup(tok);
        tok = NULL;
    }
    *pc = c; *pv = v;
}

void tok_ini(const char *s0, char *del, /**/ int *pc, char ***pv) {
    char *s;
    s = strdup(s0);
    tok_ini0(s, del, /**/ pc, pv);
    EFREE(s);
}

void tok_fin(int c, char **pv) {
    int i;
    for (i = 0; i < c; i++) EFREE(pv[i]);
    EFREE(pv);
}
