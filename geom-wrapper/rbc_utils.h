#ifndef RBC_UTILS_H
#define RBC_UTILS_H

float env2f(const char* n); /* read a float from env. */
int   env2d(const char* n); /* read an integer from env. */
FILE* safe_fopen(const char* fn, const char *mode);
void  safe_fread(void *ptr, size_t size, size_t nmemb, FILE *stream);
char* trim(char* s);

template <typename NType>
long gnp(FILE* fd, long nvar) { /* return a number of points in the file
				   (assuming NType and `NVAR' per lines) */
  long sz;
  fseek(fd, 0L, SEEK_END); sz = ftell(fd); fseek(fd, 0, SEEK_SET);
  return sz / sizeof(NType) / nvar;
}


#endif
