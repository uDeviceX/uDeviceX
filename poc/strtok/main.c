#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void concatenate(int n, char **ss, /**/ char *a) {
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
    free(s);
    return c;
}

void tok_ini0(char *s, char *del, /**/ int *pc, char ***pv) {
    int i, c;
    char *tok;
    char **v;
    c = cnt(s, del);
    v = malloc(c*sizeof(v[0]));
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
    free(s);
}

void tok_fin(int c, char **pv) {
    int i;
    for (i = 0; i < c; i++) free(pv[i]);
    free(pv);
}

int main() {
    char a[2048];
    char s[] = "a bcde";
    char delim[] = " \t";
    int i, c;
    char **v;
    tok_ini(s, delim, /**/ &c, &v);
    for (i = 0; i < c; i++)
        printf("tok: %s\n", v[i]);
    concatenate(c, v, /**/ a);
    printf("a: %s\n", a);
    tok_fin(c, v);
    return 0;
}
