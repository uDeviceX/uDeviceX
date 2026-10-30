#pragma once

#include <stddef.h>
#include <stdio.h>

void emalloc(size_t size, void **data);
void efree(void *ptr);
void *ememcpy(void *dest, const void *src, size_t n);

void efopen(const char *fname, const char *mode, FILE **stream);
void efclose(FILE *stream);
void efread(void *ptr, size_t size, size_t nmemb, FILE *stream);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream);
void efgets(char *s, int size, FILE *stream);
