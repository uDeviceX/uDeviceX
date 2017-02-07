#ifndef RBC_UTILS_H
#define RBC_UTILS_H
#include <cstdlib>
#include <cstdio>
#include <cstring>

float env2f(const char* n) { /* read a float from env. */
  char* v_ch = getenv(n);
  if (v_ch == NULL) {
    fprintf(stderr, "(rw) ERROR: environment variable `%s' should be set\n", n);
    exit(2);
  }
  return atof(v_ch);
}

int env2d(const char* n) { /* read an integer from env. */
  char* v_ch = getenv(n);
  if (v_ch == NULL) {
    fprintf(stderr, "(rw) ERROR: environment variable `%s' should be set\n", n);
    exit(2);
  }
  return atoi(v_ch);
}

FILE* safe_fopen(const char* fn, const char *mode) {
  FILE* fd = fopen(fn, mode);
  if (fd == NULL) {
    fprintf(stderr, "(rbc_utils) ERROR: cannot open file: %s\n", fn);
    exit(2);
  }
  return fd;
}

void safe_fread(void *ptr, size_t size, size_t nmemb, FILE *stream) {
  size_t rc = fread(ptr, size, nmemb, stream);
  if (rc == 0) {
    fprintf(stderr, "(rbc_utils) ERROR: cannot read data\n");
    exit(2);
  }
}

char* trim(char* s) { /* remove trailing and leading blanks, tabs,
			   new lines */
  /* trailing */
  int n;
  for (n = std::strlen(s)-1; n >= 0; n--)
    if (s[n] != ' ' && s[n] != '\t' && s[n] != '\n')
      break;
  s[n+1] = '\0';

  /* and leading */
  while (s[0] != '\0' && (s[0] == ' ' || s[0] == '\t')) s++;
  return s;
}

#endif
