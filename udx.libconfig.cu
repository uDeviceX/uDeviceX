/* begin bundled libconfig */
/* begin libconfig/lib/libconfig.h */
/* ----------------------------------------------------------------------------
   libconfig - A library for processing structured configuration files
   Copyright (C) 2005-2018  Mark A Lindner

   This file is part of libconfig.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License
   as published by the Free Software Foundation; either version 2.1 of
   the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Library General Public
   License along with this library; if not, see
   <http://www.gnu.org/licenses/>.
   ----------------------------------------------------------------------------
*/

#ifndef __libconfig_h
#define __libconfig_h

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#if defined(LIBCONFIG_STATIC)
#define LIBCONFIG_API
#elif defined(LIBCONFIG_EXPORTS)
#define LIBCONFIG_API __declspec(dllexport)
#else /* ! LIBCONFIG_EXPORTS */
#define LIBCONFIG_API __declspec(dllimport)
#endif /* LIBCONFIG_STATIC */
#else /* ! WIN32 */
#define LIBCONFIG_API
#endif /* WIN32 */

#define LIBCONFIG_VER_MAJOR    1
#define LIBCONFIG_VER_MINOR    7
#define LIBCONFIG_VER_REVISION 0

#include <stdio.h>

#define CONFIG_TYPE_NONE    0
#define CONFIG_TYPE_GROUP   1
#define CONFIG_TYPE_INT     2
#define CONFIG_TYPE_INT64   3
#define CONFIG_TYPE_FLOAT   4
#define CONFIG_TYPE_STRING  5
#define CONFIG_TYPE_BOOL    6
#define CONFIG_TYPE_ARRAY   7
#define CONFIG_TYPE_LIST    8

#define CONFIG_FORMAT_DEFAULT  0
#define CONFIG_FORMAT_HEX      1

#define CONFIG_OPTION_AUTOCONVERT                     0x01
#define CONFIG_OPTION_SEMICOLON_SEPARATORS            0x02
#define CONFIG_OPTION_COLON_ASSIGNMENT_FOR_GROUPS     0x04
#define CONFIG_OPTION_COLON_ASSIGNMENT_FOR_NON_GROUPS 0x08
#define CONFIG_OPTION_OPEN_BRACE_ON_SEPARATE_LINE     0x10
#define CONFIG_OPTION_ALLOW_SCIENTIFIC_NOTATION       0x20
#define CONFIG_OPTION_FSYNC                           0x40
#define CONFIG_OPTION_ALLOW_OVERRIDES                 0x80

#define CONFIG_TRUE  (1)
#define CONFIG_FALSE (0)

typedef union config_value_t
{
  int ival;
  long long llval;
  double fval;
  char *sval;
  struct config_list_t *list;
} config_value_t;

typedef struct config_setting_t
{
  char *name;
  short type;
  short format;
  config_value_t value;
  struct config_setting_t *parent;
  struct config_t *config;
  void *hook;
  unsigned int line;
  const char *file;
} config_setting_t;

typedef enum
{
  CONFIG_ERR_NONE = 0,
  CONFIG_ERR_FILE_IO = 1,
  CONFIG_ERR_PARSE = 2
} config_error_t;

typedef struct config_list_t
{
  unsigned int length;
  config_setting_t **elements;
} config_list_t;

typedef const char ** (*config_include_fn_t)(struct config_t *,
                                             const char *,
                                             const char *,
                                             const char **);

typedef struct config_t
{
  config_setting_t *root;
  void (*destructor)(void *);
  int options;
  unsigned short tab_width;
  unsigned short float_precision;
  unsigned short default_format;
  const char *include_dir;
  config_include_fn_t include_fn;
  const char *error_text;
  const char *error_file;
  int error_line;
  config_error_t error_type;
  const char **filenames;
  void *hook;
} config_t;

extern LIBCONFIG_API int config_read(config_t *config, FILE *stream);
extern LIBCONFIG_API void config_write(const config_t *config, FILE *stream);

extern LIBCONFIG_API void config_set_default_format(config_t *config,
                                                    short format);

extern LIBCONFIG_API void config_set_options(config_t *config, int options);
extern LIBCONFIG_API int config_get_options(const config_t *config);

extern LIBCONFIG_API void config_set_option(config_t *config, int option,
                                            int flag);
extern LIBCONFIG_API int config_get_option(const config_t *config, int option);

extern LIBCONFIG_API int config_read_string(config_t *config, const char *str);

extern LIBCONFIG_API int config_read_file(config_t *config,
                                          const char *filename);
extern LIBCONFIG_API int config_write_file(config_t *config,
                                           const char *filename);

extern LIBCONFIG_API void config_set_destructor(config_t *config,
                                                void (*destructor)(void *));
extern LIBCONFIG_API void config_set_include_dir(config_t *config,
                                                 const char *include_dir);
extern LIBCONFIG_API void config_set_include_func(config_t *config,
                                                  config_include_fn_t func);

extern LIBCONFIG_API void config_set_float_precision(config_t *config,
                                                     unsigned short digits);
extern LIBCONFIG_API unsigned short config_get_float_precision(
  const config_t *config);

extern LIBCONFIG_API void config_set_tab_width(config_t *config,
                                               unsigned short width);
extern LIBCONFIG_API unsigned short config_get_tab_width(
  const config_t *config);

extern LIBCONFIG_API void config_set_hook(config_t *config, void *hook);

#define config_get_hook(C) ((C)->hook)

extern LIBCONFIG_API void config_init(config_t *config);
extern LIBCONFIG_API void config_destroy(config_t *config);
extern LIBCONFIG_API void config_clear(config_t *config);

extern LIBCONFIG_API int config_setting_get_int(
  const config_setting_t *setting);
extern LIBCONFIG_API long long config_setting_get_int64(
  const config_setting_t *setting);
extern LIBCONFIG_API double config_setting_get_float(
  const config_setting_t *setting);
extern LIBCONFIG_API int config_setting_get_bool(
  const config_setting_t *setting);
extern LIBCONFIG_API const char *config_setting_get_string(
  const config_setting_t *setting);

extern LIBCONFIG_API int config_setting_lookup_int(
  const config_setting_t *setting, const char *name, int *value);
extern LIBCONFIG_API int config_setting_lookup_int64(
  const config_setting_t *setting, const char *name, long long *value);
extern LIBCONFIG_API int config_setting_lookup_float(
  const config_setting_t *setting, const char *name, double *value);
extern LIBCONFIG_API int config_setting_lookup_bool(
  const config_setting_t *setting, const char *name, int *value);
extern LIBCONFIG_API int config_setting_lookup_string(
  const config_setting_t *setting, const char *name, const char **value);

extern LIBCONFIG_API int config_setting_set_int(config_setting_t *setting,
                                                int value);
extern LIBCONFIG_API int config_setting_set_int64(config_setting_t *setting,
                                                  long long value);
extern LIBCONFIG_API int config_setting_set_float(config_setting_t *setting,
                                                  double value);
extern LIBCONFIG_API int config_setting_set_bool(config_setting_t *setting,
                                                 int value);
extern LIBCONFIG_API int config_setting_set_string(config_setting_t *setting,
                                                   const char *value);

extern LIBCONFIG_API int config_setting_set_format(config_setting_t *setting,
                                                   short format);
extern LIBCONFIG_API short config_setting_get_format(
  const config_setting_t *setting);

extern LIBCONFIG_API int config_setting_get_int_elem(
  const config_setting_t *setting, int idx);
extern LIBCONFIG_API long long config_setting_get_int64_elem(
  const config_setting_t *setting, int idx);
extern LIBCONFIG_API double config_setting_get_float_elem(
  const config_setting_t *setting, int idx);
extern LIBCONFIG_API int config_setting_get_bool_elem(
  const config_setting_t *setting, int idx);
extern LIBCONFIG_API const char *config_setting_get_string_elem(
  const config_setting_t *setting, int idx);

extern LIBCONFIG_API config_setting_t *config_setting_set_int_elem(
  config_setting_t *setting, int idx, int value);
extern LIBCONFIG_API config_setting_t *config_setting_set_int64_elem(
  config_setting_t *setting, int idx, long long value);
extern LIBCONFIG_API config_setting_t *config_setting_set_float_elem(
  config_setting_t *setting, int idx, double value);
extern LIBCONFIG_API config_setting_t *config_setting_set_bool_elem(
  config_setting_t *setting, int idx, int value);
extern LIBCONFIG_API config_setting_t *config_setting_set_string_elem(
  config_setting_t *setting, int idx, const char *value);

extern LIBCONFIG_API const char **config_default_include_func(
    config_t *config, const char *include_dir, const char *path,
    const char **error);

extern LIBCONFIG_API int config_setting_is_scalar(
    const config_setting_t *setting);

extern LIBCONFIG_API int config_setting_is_aggregate(
    const config_setting_t *setting);

#define /* const char * */ config_get_include_dir(/* const config_t * */ C) \
  ((C)->include_dir)

#define /* void */ config_set_auto_convert(/* config_t * */ C, F) \
  config_set_option((C), CONFIG_OPTION_AUTOCONVERT, (F))
#define /* int */ config_get_auto_convert(/* const config_t * */ C) \
  config_get_option((C), CONFIG_OPTION_AUTOCONVERT)

#define /* int */ config_setting_type(/* const config_setting_t * */ S) \
  ((S)->type)

#define /* int */ config_setting_is_group(/* const config_setting_t * */ S) \
  ((S)->type == CONFIG_TYPE_GROUP)
#define /* int */ config_setting_is_array(/* const config_setting_t * */ S) \
  ((S)->type == CONFIG_TYPE_ARRAY)
#define /* int */ config_setting_is_list(/* const config_setting_t * */ S) \
  ((S)->type == CONFIG_TYPE_LIST)

#define /* int */ config_setting_is_number(/* const config_setting_t * */ S) \
  (((S)->type == CONFIG_TYPE_INT)                                       \
   || ((S)->type == CONFIG_TYPE_INT64)                                  \
   || ((S)->type == CONFIG_TYPE_FLOAT))

#define /* const char * */ config_setting_name( \
  /* const config_setting_t * */ S)             \
  ((S)->name)

#define /* config_setting_t * */ config_setting_parent( \
  /* const config_setting_t * */ S)                     \
  ((S)->parent)

#define /* int */ config_setting_is_root(       \
  /* const config_setting_t * */ S)             \
  ((S)->parent ? CONFIG_FALSE : CONFIG_TRUE)

extern LIBCONFIG_API int config_setting_index(const config_setting_t *setting);

extern LIBCONFIG_API int config_setting_length(
  const config_setting_t *setting);
extern LIBCONFIG_API config_setting_t *config_setting_get_elem(
  const config_setting_t *setting, unsigned int idx);

extern LIBCONFIG_API config_setting_t *config_setting_get_member(
  const config_setting_t *setting, const char *name);

extern LIBCONFIG_API config_setting_t *config_setting_add(
  config_setting_t *parent, const char *name, int type);
extern LIBCONFIG_API int config_setting_remove(config_setting_t *parent,
                                               const char *name);
extern LIBCONFIG_API int config_setting_remove_elem(config_setting_t *parent,
                                                    unsigned int idx);
extern LIBCONFIG_API void config_setting_set_hook(config_setting_t *setting,
                                                  void *hook);

#define config_setting_get_hook(S) ((S)->hook)

extern LIBCONFIG_API config_setting_t *config_lookup(const config_t *config,
                                                     const char *path);
extern LIBCONFIG_API config_setting_t *config_setting_lookup(
  config_setting_t *setting, const char *path);

extern LIBCONFIG_API int config_lookup_int(const config_t *config,
                                           const char *path, int *value);
extern LIBCONFIG_API int config_lookup_int64(const config_t *config,
                                             const char *path,
                                             long long *value);
extern LIBCONFIG_API int config_lookup_float(const config_t *config,
                                             const char *path, double *value);
extern LIBCONFIG_API int config_lookup_bool(const config_t *config,
                                            const char *path, int *value);
extern LIBCONFIG_API int config_lookup_string(const config_t *config,
                                              const char *path,
                                              const char **value);

#define /* config_setting_t * */ config_root_setting( \
  /* const config_t * */ C)                           \
  ((C)->root)

#define  /* void */ config_set_default_format(/* config_t * */ C,       \
                                              /* short */ F)            \
  (C)->default_format = (F)

#define /* short */ config_get_default_format(/* config_t * */ C)       \
  ((C)->default_format)

#define /* unsigned short */ config_setting_source_line(   \
  /* const config_setting_t * */ S)                        \
  ((S)->line)

#define /* const char */ config_setting_source_file(    \
  /* const config_setting_t * */ S)                     \
  ((S)->file)

#define /* const char * */ config_error_text(/* const config_t * */ C)  \
  ((C)->error_text)

#define /* const char * */ config_error_file(/* const config_t * */ C)  \
  ((C)->error_file)

#define /* int */ config_error_line(/* const config_t * */ C)   \
  ((C)->error_line)

#define /* config_error_t */ config_error_type(/* const config_t * */ C) \
  ((C)->error_type)

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __libconfig_h */
/* end libconfig/lib/libconfig.h */
/* begin libconfig/lib/strbuf.h */
/* ----------------------------------------------------------------------------
   libconfig - A library for processing structured configuration files
   Copyright (C) 2005-2020  Mark A Lindner

   This file is part of libconfig.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License
   as published by the Free Software Foundation; either version 2.1 of
   the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Library General Public
   License along with this library; if not, see
   <http://www.gnu.org/licenses/>.
   ----------------------------------------------------------------------------
*/

#ifndef __libconfig_strbuf_h
#define __libconfig_strbuf_h

#include <string.h>
#include <sys/types.h>

typedef struct
{
  char *string;
  size_t length;
  size_t capacity;
} strbuf_t;

void libconfig_strbuf_append_string(strbuf_t *buf, const char *s);

void libconfig_strbuf_append_char(strbuf_t *buf, char c);

char *libconfig_strbuf_release(strbuf_t *buf);

#endif /* __libconfig_strbuf_h */
/* end libconfig/lib/strbuf.h */
/* begin libconfig/lib/strvec.h */
/* ----------------------------------------------------------------------------
   libconfig - A library for processing structured configuration files
   Copyright (C) 2005-2020  Mark A Lindner

   This file is part of libconfig.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License
   as published by the Free Software Foundation; either version 2.1 of
   the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Library General Public
   License along with this library; if not, see
   <http://www.gnu.org/licenses/>.
   ----------------------------------------------------------------------------
*/

#ifndef __libconfig_strvec_h
#define __libconfig_strvec_h

#include <string.h>
#include <sys/types.h>

typedef struct
{
  const char **strings;
  const char **end;
  size_t length;
  size_t capacity;
} strvec_t;

extern void libconfig_strvec_append(strvec_t *vec, const char *s);

extern const char **libconfig_strvec_release(strvec_t *vec);

extern void libconfig_strvec_delete(const char * const *vec);

#endif /* __libconfig_strvec_h */
/* end libconfig/lib/strvec.h */
/* begin libconfig/lib/util.h */
/* ----------------------------------------------------------------------------
   libconfig - A library for processing structured configuration files
   Copyright (C) 2005-2018  Mark A Lindner

   This file is part of libconfig.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License
   as published by the Free Software Foundation; either version 2.1 of
   the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Library General Public
   License along with this library; if not, see
   <http://www.gnu.org/licenses/>.
   ----------------------------------------------------------------------------
*/

#include <string.h>
#include <sys/types.h>

#define __new(T) (T *)calloc(1, sizeof(T)) /* zeroed */
#define __delete(P) free((void *)(P))
#define __zero(P) memset((void *)(P), 0, sizeof(*P))

extern long long libconfig_parse_integer(const char *s, int *ok);
extern unsigned long long libconfig_parse_hex64(const char *s);

extern void libconfig_format_double(double val, int precision, int sci_ok,
                                    char *buf, size_t buflen);

/* end libconfig/lib/util.h */
/* begin libconfig/lib/wincompat.h */
/* ----------------------------------------------------------------------------
   libconfig - A library for processing structured configuration files
   Copyright (C) 2005-2018  Mark A Lindner

   This file is part of libconfig.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License
   as published by the Free Software Foundation; either version 2.1 of
   the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Library General Public
   License along with this library; if not, see
   <http://www.gnu.org/licenses/>.
   ----------------------------------------------------------------------------
*/

#ifndef __wincompat_h
#define __wincompat_h

#include <limits.h>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) \
  || defined(WIN64) || defined(_WIN64) || defined(__WIN64__)

/* Prevent warnings about redefined malloc/free in generated code. */
#ifndef _STDLIB_H
#define _STDLIB_H
#endif

#include <malloc.h>

#ifdef _MSC_VER
#pragma warning (disable: 4996)
#endif

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#define fileno _fileno

#if _MSC_VER <= 1800
#define snprintf  _snprintf
#endif

#if !defined(__MINGW32__) && _MSC_VER < 1800
#define atoll     _atoi64
#define strtoull  _strtoui64
#define strtoll   _strtoi64
#endif

#if !defined(S_ISDIR) && defined(S_IFMT) && defined(S_IFDIR)
#define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)
#endif

#endif

#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__) \
  || defined(WIN64) || defined(_WIN64) || defined(__WIN64__) \
  || defined(__MINGW32__))

#define INT64_FMT "%I64d"
#define UINT64_FMT "%I64u"

#define INT64_HEX_FMT "%I64X"

#define FILE_SEPARATOR "\\"

#else /* defined(WIN32) || defined(__MINGW32__) */

#define INT64_FMT "%lld"
#define UINT64_FMT "%llu"

#define INT64_HEX_FMT "%llX"

#define FILE_SEPARATOR "/"

#endif /* defined(WIN32) || defined(__MINGW32__) */

#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__) \
  || defined(WIN64) || defined(_WIN64) || defined(__WIN64__)) \
  && ! defined(__MINGW32__)

#define INT64_CONST(I)  (I ## i64)
#define UINT64_CONST(I) (I ## Ui64)

#ifndef INT_MAX
#define INT_MAX (2147483647)
#endif

#ifndef INT_MIN
#define INT_MIN (-2147483647-1)
#endif

#include <Shlwapi.h>
#define IS_RELATIVE_PATH(P) \
  (PathIsRelativeA(P))

extern int fsync(int fd);

#else /* defined(WIN32/WIN64) && ! defined(__MINGW32__) */

#define INT64_CONST(I)  (I ## LL)
#define UINT64_CONST(I) (I ## ULL)

#define IS_RELATIVE_PATH(P) \
  ((P)[0] != '/')

#include <unistd.h> /* for fsync() */

#endif /* defined(WIN32/WIN64) && ! defined(__MINGW32__) */

#endif /* __wincompat_h */
/* end libconfig/lib/wincompat.h */
/* begin libconfig/lib/scanctx.h */
/* ----------------------------------------------------------------------------
   libconfig - A library for processing structured configuration files
   Copyright (C) 2005-2020  Mark A Lindner

   This file is part of libconfig.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License
   as published by the Free Software Foundation; either version 2.1 of
   the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Library General Public
   License along with this library; if not, see
   <http://www.gnu.org/licenses/>.
   ----------------------------------------------------------------------------
*/

#ifndef __libconfig_scanctx_h
#define __libconfig_scanctx_h

#include <stdio.h>
#include <string.h>
#include <sys/types.h>


#define MAX_INCLUDE_DEPTH 10

struct include_stack_frame
{
  /*
   * Member strings are not owned; they are pointers into scan_context's
   * filenames vector.
   */
  const char **files;
  const char **current_file;
  FILE *current_stream;
  void *parent_buffer;
};

struct scan_context
{
  config_t *config;
  const char *top_filename;
  struct include_stack_frame include_stack[MAX_INCLUDE_DEPTH];
  int stack_depth;
  strbuf_t string;
  strvec_t filenames;
};

extern void libconfig_scanctx_init(struct scan_context *ctx,
                                   const char *top_filename);
extern const char **libconfig_scanctx_cleanup(struct scan_context *ctx);

/*
 * Pushes a new frame onto the include stack, and returns an open stream to the
 * first file in the include list, if any.
 *
 * ctx - The scan context
 * prev_buffer - The current input buffer, to be restored when this frame is
 * popped
 * path - The string argument to the @include directive, to be expanded into a
 * list of zero or more filenames using the function ctx->config->include_fn
 * error - A pointer at which to store a static error message, if any.
 *
 * On success, the new frame will be pushed and the stream to the first file
 * will be returned.
 *
 * On failure, the frame will not be pushed and NULL will be returned. If
 * *error is NULL, it means there are no files in the list. Otherwise, it
 * points to an error and parsing should be aborted.
 */
extern FILE *libconfig_scanctx_push_include(struct scan_context *ctx,
                                            void *prev_buffer,
                                            const char *path,
                                            const char **error);

/*
 * Returns the next include file in the current include stack frame.
 *
 * Returns NULL on failure or if there are no more files left in the current
 * frame. If there was an error, sets *error.
 */
extern FILE *libconfig_scanctx_next_include_file(struct scan_context *ctx,
                                                 const char **error);

/*
 * Pops a frame off the include stack.
 */
extern void *libconfig_scanctx_pop_include(struct scan_context *ctx);

#define libconfig_scanctx_append_string(C, S) \
  libconfig_strbuf_append_string(&((C)->string), (S))

#define libconfig_scanctx_append_char(C, X) \
  libconfig_strbuf_append_char(&((C)->string), (X))

extern char *libconfig_scanctx_take_string(struct scan_context *ctx);

extern const char *libconfig_scanctx_current_filename(struct scan_context *ctx);

#endif /* __libconfig_scanctx_h */
/* end libconfig/lib/scanctx.h */
/* begin libconfig/lib/parsectx.h */
/* ----------------------------------------------------------------------------
   libconfig - A library for processing structured configuration files
   Copyright (C) 2005-2020  Mark A Lindner

   This file is part of libconfig.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License
   as published by the Free Software Foundation; either version 2.1 of
   the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Library General Public
   License along with this library; if not, see
   <http://www.gnu.org/licenses/>.
   ----------------------------------------------------------------------------
*/

#ifndef __libconfig_parsectx_h
#define __libconfig_parsectx_h


struct parse_context
{
  config_t *config;
  config_setting_t *parent;
  config_setting_t *setting;
  char *name;
  strbuf_t string;
};

#define libconfig_parsectx_init(C) \
  __zero(C)
#define libconfig_parsectx_cleanup(C) \
  __delete(libconfig_strbuf_release(&((C)->string)))

#define libconfig_parsectx_append_string(C, S) \
  libconfig_strbuf_append_string(&((C)->string), (S))
#define libconfig_parsectx_take_string(C) \
  libconfig_strbuf_release(&((C)->string))

#endif /* __libconfig_parsectx_h */
/* end libconfig/lib/parsectx.h */
/* begin libconfig/lib/grammar.h */
/* A Bison parser, made by GNU Bison 3.0.4.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

#ifndef YY_LIBCONFIG_YY_GRAMMAR_H_INCLUDED
# define YY_LIBCONFIG_YY_GRAMMAR_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int libconfig_yydebug;
#endif

/* Token type.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    TOK_BOOLEAN = 258,
    TOK_INTEGER = 259,
    TOK_HEX = 260,
    TOK_INTEGER64 = 261,
    TOK_HEX64 = 262,
    TOK_FLOAT = 263,
    TOK_STRING = 264,
    TOK_NAME = 265,
    TOK_EQUALS = 266,
    TOK_NEWLINE = 267,
    TOK_ARRAY_START = 268,
    TOK_ARRAY_END = 269,
    TOK_LIST_START = 270,
    TOK_LIST_END = 271,
    TOK_COMMA = 272,
    TOK_GROUP_START = 273,
    TOK_GROUP_END = 274,
    TOK_SEMICOLON = 275,
    TOK_GARBAGE = 276,
    TOK_ERROR = 277
  };
#endif
/* Tokens.  */
#define TOK_BOOLEAN 258
#define TOK_INTEGER 259
#define TOK_HEX 260
#define TOK_INTEGER64 261
#define TOK_HEX64 262
#define TOK_FLOAT 263
#define TOK_STRING 264
#define TOK_NAME 265
#define TOK_EQUALS 266
#define TOK_NEWLINE 267
#define TOK_ARRAY_START 268
#define TOK_ARRAY_END 269
#define TOK_LIST_START 270
#define TOK_LIST_END 271
#define TOK_COMMA 272
#define TOK_GROUP_START 273
#define TOK_GROUP_END 274
#define TOK_SEMICOLON 275
#define TOK_GARBAGE 276
#define TOK_ERROR 277

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED

union YYSTYPE
{
#line 76 "grammar.y" /* yacc.c:1909  */

  int ival;
  long long llval;
  double fval;
  char *sval;

#line 105 "grammar.h" /* yacc.c:1909  */
};

typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif



int libconfig_yyparse (void *scanner, struct parse_context *ctx, struct scan_context *scan_ctx);

#endif /* !YY_LIBCONFIG_YY_GRAMMAR_H_INCLUDED  */
/* end libconfig/lib/grammar.h */
/* begin libconfig/lib/scanner.h */
#ifndef libconfig_yyHEADER_H
#define libconfig_yyHEADER_H 1
#define libconfig_yyIN_HEADER 1

#line 5 "scanner.h"

#line 7 "scanner.h"

#define  YY_INT_ALIGNED short int

/* A lexical scanner generated by flex */

#define FLEX_SCANNER
#define YY_FLEX_MAJOR_VERSION 2
#define YY_FLEX_MINOR_VERSION 6
#define YY_FLEX_SUBMINOR_VERSION 4
#if YY_FLEX_SUBMINOR_VERSION > 0
#define FLEX_BETA
#endif

#ifdef yy_create_buffer
#define libconfig_yy_create_buffer_ALREADY_DEFINED
#else
#define yy_create_buffer libconfig_yy_create_buffer
#endif

#ifdef yy_delete_buffer
#define libconfig_yy_delete_buffer_ALREADY_DEFINED
#else
#define yy_delete_buffer libconfig_yy_delete_buffer
#endif

#ifdef yy_scan_buffer
#define libconfig_yy_scan_buffer_ALREADY_DEFINED
#else
#define yy_scan_buffer libconfig_yy_scan_buffer
#endif

#ifdef yy_scan_string
#define libconfig_yy_scan_string_ALREADY_DEFINED
#else
#define yy_scan_string libconfig_yy_scan_string
#endif

#ifdef yy_scan_bytes
#define libconfig_yy_scan_bytes_ALREADY_DEFINED
#else
#define yy_scan_bytes libconfig_yy_scan_bytes
#endif

#ifdef yy_init_buffer
#define libconfig_yy_init_buffer_ALREADY_DEFINED
#else
#define yy_init_buffer libconfig_yy_init_buffer
#endif

#ifdef yy_flush_buffer
#define libconfig_yy_flush_buffer_ALREADY_DEFINED
#else
#define yy_flush_buffer libconfig_yy_flush_buffer
#endif

#ifdef yy_load_buffer_state
#define libconfig_yy_load_buffer_state_ALREADY_DEFINED
#else
#define yy_load_buffer_state libconfig_yy_load_buffer_state
#endif

#ifdef yy_switch_to_buffer
#define libconfig_yy_switch_to_buffer_ALREADY_DEFINED
#else
#define yy_switch_to_buffer libconfig_yy_switch_to_buffer
#endif

#ifdef yypush_buffer_state
#define libconfig_yypush_buffer_state_ALREADY_DEFINED
#else
#define yypush_buffer_state libconfig_yypush_buffer_state
#endif

#ifdef yypop_buffer_state
#define libconfig_yypop_buffer_state_ALREADY_DEFINED
#else
#define yypop_buffer_state libconfig_yypop_buffer_state
#endif

#ifdef yyensure_buffer_stack
#define libconfig_yyensure_buffer_stack_ALREADY_DEFINED
#else
#define yyensure_buffer_stack libconfig_yyensure_buffer_stack
#endif

#ifdef yylex
#define libconfig_yylex_ALREADY_DEFINED
#else
#define yylex libconfig_yylex
#endif

#ifdef yyrestart
#define libconfig_yyrestart_ALREADY_DEFINED
#else
#define yyrestart libconfig_yyrestart
#endif

#ifdef yylex_init
#define libconfig_yylex_init_ALREADY_DEFINED
#else
#define yylex_init libconfig_yylex_init
#endif

#ifdef yylex_init_extra
#define libconfig_yylex_init_extra_ALREADY_DEFINED
#else
#define yylex_init_extra libconfig_yylex_init_extra
#endif

#ifdef yylex_destroy
#define libconfig_yylex_destroy_ALREADY_DEFINED
#else
#define yylex_destroy libconfig_yylex_destroy
#endif

#ifdef yyget_debug
#define libconfig_yyget_debug_ALREADY_DEFINED
#else
#define yyget_debug libconfig_yyget_debug
#endif

#ifdef yyset_debug
#define libconfig_yyset_debug_ALREADY_DEFINED
#else
#define yyset_debug libconfig_yyset_debug
#endif

#ifdef yyget_extra
#define libconfig_yyget_extra_ALREADY_DEFINED
#else
#define yyget_extra libconfig_yyget_extra
#endif

#ifdef yyset_extra
#define libconfig_yyset_extra_ALREADY_DEFINED
#else
#define yyset_extra libconfig_yyset_extra
#endif

#ifdef yyget_in
#define libconfig_yyget_in_ALREADY_DEFINED
#else
#define yyget_in libconfig_yyget_in
#endif

#ifdef yyset_in
#define libconfig_yyset_in_ALREADY_DEFINED
#else
#define yyset_in libconfig_yyset_in
#endif

#ifdef yyget_out
#define libconfig_yyget_out_ALREADY_DEFINED
#else
#define yyget_out libconfig_yyget_out
#endif

#ifdef yyset_out
#define libconfig_yyset_out_ALREADY_DEFINED
#else
#define yyset_out libconfig_yyset_out
#endif

#ifdef yyget_leng
#define libconfig_yyget_leng_ALREADY_DEFINED
#else
#define yyget_leng libconfig_yyget_leng
#endif

#ifdef yyget_text
#define libconfig_yyget_text_ALREADY_DEFINED
#else
#define yyget_text libconfig_yyget_text
#endif

#ifdef yyget_lineno
#define libconfig_yyget_lineno_ALREADY_DEFINED
#else
#define yyget_lineno libconfig_yyget_lineno
#endif

#ifdef yyset_lineno
#define libconfig_yyset_lineno_ALREADY_DEFINED
#else
#define yyset_lineno libconfig_yyset_lineno
#endif

#ifdef yyget_column
#define libconfig_yyget_column_ALREADY_DEFINED
#else
#define yyget_column libconfig_yyget_column
#endif

#ifdef yyset_column
#define libconfig_yyset_column_ALREADY_DEFINED
#else
#define yyset_column libconfig_yyset_column
#endif

#ifdef yywrap
#define libconfig_yywrap_ALREADY_DEFINED
#else
#define yywrap libconfig_yywrap
#endif

#ifdef yyget_lval
#define libconfig_yyget_lval_ALREADY_DEFINED
#else
#define yyget_lval libconfig_yyget_lval
#endif

#ifdef yyset_lval
#define libconfig_yyset_lval_ALREADY_DEFINED
#else
#define yyset_lval libconfig_yyset_lval
#endif

#ifdef yyalloc
#define libconfig_yyalloc_ALREADY_DEFINED
#else
#define yyalloc libconfig_yyalloc
#endif

#ifdef yyrealloc
#define libconfig_yyrealloc_ALREADY_DEFINED
#else
#define yyrealloc libconfig_yyrealloc
#endif

#ifdef yyfree
#define libconfig_yyfree_ALREADY_DEFINED
#else
#define yyfree libconfig_yyfree
#endif

/* First, we deal with  platform-specific or compiler-specific issues. */

/* begin standard C headers. */
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>

/* end standard C headers. */

/* flex integer type definitions */

#ifndef FLEXINT_H
#define FLEXINT_H

/* C99 systems have <inttypes.h>. Non-C99 systems may or may not. */

#if defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L

/* C99 says to define __STDC_LIMIT_MACROS before including stdint.h,
 * if you want the limit (max/min) macros for int types.
 */
#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS 1
#endif

#include <inttypes.h>
typedef int8_t flex_int8_t;
typedef uint8_t flex_uint8_t;
typedef int16_t flex_int16_t;
typedef uint16_t flex_uint16_t;
typedef int32_t flex_int32_t;
typedef uint32_t flex_uint32_t;
#else
typedef signed char flex_int8_t;
typedef short int flex_int16_t;
typedef int flex_int32_t;
typedef unsigned char flex_uint8_t;
typedef unsigned short int flex_uint16_t;
typedef unsigned int flex_uint32_t;

/* Limits of integral types. */
#ifndef INT8_MIN
#define INT8_MIN               (-128)
#endif
#ifndef INT16_MIN
#define INT16_MIN              (-32767-1)
#endif
#ifndef INT32_MIN
#define INT32_MIN              (-2147483647-1)
#endif
#ifndef INT8_MAX
#define INT8_MAX               (127)
#endif
#ifndef INT16_MAX
#define INT16_MAX              (32767)
#endif
#ifndef INT32_MAX
#define INT32_MAX              (2147483647)
#endif
#ifndef UINT8_MAX
#define UINT8_MAX              (255U)
#endif
#ifndef UINT16_MAX
#define UINT16_MAX             (65535U)
#endif
#ifndef UINT32_MAX
#define UINT32_MAX             (4294967295U)
#endif

#ifndef SIZE_MAX
#define SIZE_MAX               (~(size_t)0)
#endif

#endif /* ! C99 */

#endif /* ! FLEXINT_H */

/* begin standard C++ headers. */

/* TODO: this is always defined, so inline it */
#define yyconst const

#if defined(__GNUC__) && __GNUC__ >= 3
#define yynoreturn __attribute__((__noreturn__))
#else
#define yynoreturn
#endif

/* An opaque pointer. */
#ifndef YY_TYPEDEF_YY_SCANNER_T
#define YY_TYPEDEF_YY_SCANNER_T
typedef void* yyscan_t;
#endif

/* For convenience, these vars (plus the bison vars far below)
   are macros in the reentrant scanner. */
#define yyin yyg->yyin_r
#define yyout yyg->yyout_r
#define yyextra yyg->yyextra_r
#define yyleng yyg->yyleng_r
#define yytext yyg->yytext_r
#define yylineno (YY_CURRENT_BUFFER_LVALUE->yy_bs_lineno)
#define yycolumn (YY_CURRENT_BUFFER_LVALUE->yy_bs_column)
#define yy_flex_debug yyg->yy_flex_debug_r

/* Size of default input buffer. */
#ifndef YY_BUF_SIZE
#ifdef __ia64__
/* On IA-64, the buffer size is 16k, not 8k.
 * Moreover, YY_BUF_SIZE is 2*YY_READ_BUF_SIZE in the general case.
 * Ditto for the __ia64__ case accordingly.
 */
#define YY_BUF_SIZE 32768
#else
#define YY_BUF_SIZE 16384
#endif /* __ia64__ */
#endif

#ifndef YY_TYPEDEF_YY_BUFFER_STATE
#define YY_TYPEDEF_YY_BUFFER_STATE
typedef struct yy_buffer_state *YY_BUFFER_STATE;
#endif

#ifndef YY_TYPEDEF_YY_SIZE_T
#define YY_TYPEDEF_YY_SIZE_T
typedef size_t yy_size_t;
#endif

#ifndef YY_STRUCT_YY_BUFFER_STATE
#define YY_STRUCT_YY_BUFFER_STATE
struct yy_buffer_state
	{
	FILE *yy_input_file;

	char *yy_ch_buf;		/* input buffer */
	char *yy_buf_pos;		/* current position in input buffer */

	/* Size of input buffer in bytes, not including room for EOB
	 * characters.
	 */
	int yy_buf_size;

	/* Number of characters read into yy_ch_buf, not including EOB
	 * characters.
	 */
	int yy_n_chars;

	/* Whether we "own" the buffer - i.e., we know we created it,
	 * and can realloc() it to grow it, and should free() it to
	 * delete it.
	 */
	int yy_is_our_buffer;

	/* Whether this is an "interactive" input source; if so, and
	 * if we're using stdio for input, then we want to use getc()
	 * instead of fread(), to make sure we stop fetching input after
	 * each newline.
	 */
	int yy_is_interactive;

	/* Whether we're considered to be at the beginning of a line.
	 * If so, '^' rules will be active on the next match, otherwise
	 * not.
	 */
	int yy_at_bol;

    int yy_bs_lineno; /**< The line count. */
    int yy_bs_column; /**< The column count. */

	/* Whether to try to fill the input buffer when we reach the
	 * end of it.
	 */
	int yy_fill_buffer;

	int yy_buffer_status;

	};
#endif /* !YY_STRUCT_YY_BUFFER_STATE */

void yyrestart ( FILE *input_file , yyscan_t yyscanner );
void yy_switch_to_buffer ( YY_BUFFER_STATE new_buffer , yyscan_t yyscanner );
YY_BUFFER_STATE yy_create_buffer ( FILE *file, int size , yyscan_t yyscanner );
void yy_delete_buffer ( YY_BUFFER_STATE b , yyscan_t yyscanner );
void yy_flush_buffer ( YY_BUFFER_STATE b , yyscan_t yyscanner );
void yypush_buffer_state ( YY_BUFFER_STATE new_buffer , yyscan_t yyscanner );
void yypop_buffer_state ( yyscan_t yyscanner );

YY_BUFFER_STATE yy_scan_buffer ( char *base, yy_size_t size , yyscan_t yyscanner );
YY_BUFFER_STATE yy_scan_string ( const char *yy_str , yyscan_t yyscanner );
YY_BUFFER_STATE yy_scan_bytes ( const char *bytes, int len , yyscan_t yyscanner );

void *yyalloc ( yy_size_t , yyscan_t yyscanner );
void *yyrealloc ( void *, yy_size_t , yyscan_t yyscanner );
void yyfree ( void * , yyscan_t yyscanner );

#define libconfig_yywrap(yyscanner) (/*CONSTCOND*/1)
#define YY_SKIP_YYWRAP

#define yytext_ptr yytext_r

#ifdef YY_HEADER_EXPORT_START_CONDITIONS
#define INITIAL 0
#define SINGLE_LINE_COMMENT 1
#define MULTI_LINE_COMMENT 2
#define STRING 3
#define INCLUDE 4

#endif

#ifndef YY_NO_UNISTD_H
/* Special case for "unistd.h", since it is non-ANSI. We include it way
 * down here because we want the user's section 1 to have been scanned first.
 * The user has a chance to override it with an option.
 */
#include <unistd.h>
#endif

#define YY_EXTRA_TYPE struct scan_context *

int yylex_init (yyscan_t* scanner);

int yylex_init_extra ( YY_EXTRA_TYPE user_defined, yyscan_t* scanner);

/* Accessor methods to globals.
   These are made visible to non-reentrant scanners for convenience. */

int yylex_destroy ( yyscan_t yyscanner );

int yyget_debug ( yyscan_t yyscanner );

void yyset_debug ( int debug_flag , yyscan_t yyscanner );

YY_EXTRA_TYPE yyget_extra ( yyscan_t yyscanner );

void yyset_extra ( YY_EXTRA_TYPE user_defined , yyscan_t yyscanner );

FILE *yyget_in ( yyscan_t yyscanner );

void yyset_in  ( FILE * _in_str , yyscan_t yyscanner );

FILE *yyget_out ( yyscan_t yyscanner );

void yyset_out  ( FILE * _out_str , yyscan_t yyscanner );

			int yyget_leng ( yyscan_t yyscanner );

char *yyget_text ( yyscan_t yyscanner );

int yyget_lineno ( yyscan_t yyscanner );

void yyset_lineno ( int _line_number , yyscan_t yyscanner );

int yyget_column  ( yyscan_t yyscanner );

void yyset_column ( int _column_no , yyscan_t yyscanner );

YYSTYPE * yyget_lval ( yyscan_t yyscanner );

void yyset_lval ( YYSTYPE * yylval_param , yyscan_t yyscanner );

/* Macros after this point can all be overridden by user definitions in
 * section 1.
 */

#ifndef YY_SKIP_YYWRAP
#ifdef __cplusplus
extern "C" int yywrap ( yyscan_t yyscanner );
#else
extern int yywrap ( yyscan_t yyscanner );
#endif
#endif

#ifndef yytext_ptr
static void yy_flex_strncpy ( char *, const char *, int , yyscan_t yyscanner);
#endif

#ifdef YY_NEED_STRLEN
static int yy_flex_strlen ( const char * , yyscan_t yyscanner);
#endif

#ifndef YY_NO_INPUT

#endif

/* Amount of stuff to slurp up with each read. */
#ifndef YY_READ_BUF_SIZE
#ifdef __ia64__
/* On IA-64, the buffer size is 16k, not 8k */
#define YY_READ_BUF_SIZE 16384
#else
#define YY_READ_BUF_SIZE 8192
#endif /* __ia64__ */
#endif

/* Number of entries by which start-condition stack grows. */
#ifndef YY_START_STACK_INCR
#define YY_START_STACK_INCR 25
#endif

/* Default declaration of generated scanner - a define so the user can
 * easily add parameters.
 */
#ifndef YY_DECL
#define YY_DECL_IS_OURS 1

extern int yylex \
               (YYSTYPE * yylval_param , yyscan_t yyscanner);

#define YY_DECL int yylex \
               (YYSTYPE * yylval_param , yyscan_t yyscanner)
#endif /* !YY_DECL */

/* yy_get_previous_state - get the state just before the EOB char was reached */

#undef YY_NEW_FILE
#undef YY_FLUSH_BUFFER
#undef yy_set_bol
#undef yy_new_buffer
#undef yy_set_interactive
#undef YY_DO_BEFORE_ACTION

#ifdef YY_DECL_IS_OURS
#undef YY_DECL_IS_OURS
#undef YY_DECL
#endif

#ifndef libconfig_yy_create_buffer_ALREADY_DEFINED
#undef yy_create_buffer
#endif
#ifndef libconfig_yy_delete_buffer_ALREADY_DEFINED
#undef yy_delete_buffer
#endif
#ifndef libconfig_yy_scan_buffer_ALREADY_DEFINED
#undef yy_scan_buffer
#endif
#ifndef libconfig_yy_scan_string_ALREADY_DEFINED
#undef yy_scan_string
#endif
#ifndef libconfig_yy_scan_bytes_ALREADY_DEFINED
#undef yy_scan_bytes
#endif
#ifndef libconfig_yy_init_buffer_ALREADY_DEFINED
#undef yy_init_buffer
#endif
#ifndef libconfig_yy_flush_buffer_ALREADY_DEFINED
#undef yy_flush_buffer
#endif
#ifndef libconfig_yy_load_buffer_state_ALREADY_DEFINED
#undef yy_load_buffer_state
#endif
#ifndef libconfig_yy_switch_to_buffer_ALREADY_DEFINED
#undef yy_switch_to_buffer
#endif
#ifndef libconfig_yypush_buffer_state_ALREADY_DEFINED
#undef yypush_buffer_state
#endif
#ifndef libconfig_yypop_buffer_state_ALREADY_DEFINED
#undef yypop_buffer_state
#endif
#ifndef libconfig_yyensure_buffer_stack_ALREADY_DEFINED
#undef yyensure_buffer_stack
#endif
#ifndef libconfig_yylex_ALREADY_DEFINED
#undef yylex
#endif
#ifndef libconfig_yyrestart_ALREADY_DEFINED
#undef yyrestart
#endif
#ifndef libconfig_yylex_init_ALREADY_DEFINED
#undef yylex_init
#endif
#ifndef libconfig_yylex_init_extra_ALREADY_DEFINED
#undef yylex_init_extra
#endif
#ifndef libconfig_yylex_destroy_ALREADY_DEFINED
#undef yylex_destroy
#endif
#ifndef libconfig_yyget_debug_ALREADY_DEFINED
#undef yyget_debug
#endif
#ifndef libconfig_yyset_debug_ALREADY_DEFINED
#undef yyset_debug
#endif
#ifndef libconfig_yyget_extra_ALREADY_DEFINED
#undef yyget_extra
#endif
#ifndef libconfig_yyset_extra_ALREADY_DEFINED
#undef yyset_extra
#endif
#ifndef libconfig_yyget_in_ALREADY_DEFINED
#undef yyget_in
#endif
#ifndef libconfig_yyset_in_ALREADY_DEFINED
#undef yyset_in
#endif
#ifndef libconfig_yyget_out_ALREADY_DEFINED
#undef yyget_out
#endif
#ifndef libconfig_yyset_out_ALREADY_DEFINED
#undef yyset_out
#endif
#ifndef libconfig_yyget_leng_ALREADY_DEFINED
#undef yyget_leng
#endif
#ifndef libconfig_yyget_text_ALREADY_DEFINED
#undef yyget_text
#endif
#ifndef libconfig_yyget_lineno_ALREADY_DEFINED
#undef yyget_lineno
#endif
#ifndef libconfig_yyset_lineno_ALREADY_DEFINED
#undef yyset_lineno
#endif
#ifndef libconfig_yyget_column_ALREADY_DEFINED
#undef yyget_column
#endif
#ifndef libconfig_yyset_column_ALREADY_DEFINED
#undef yyset_column
#endif
#ifndef libconfig_yywrap_ALREADY_DEFINED
#undef yywrap
#endif
#ifndef libconfig_yyget_lval_ALREADY_DEFINED
#undef yyget_lval
#endif
#ifndef libconfig_yyset_lval_ALREADY_DEFINED
#undef yyset_lval
#endif
#ifndef libconfig_yyget_lloc_ALREADY_DEFINED
#undef yyget_lloc
#endif
#ifndef libconfig_yyset_lloc_ALREADY_DEFINED
#undef yyset_lloc
#endif
#ifndef libconfig_yyalloc_ALREADY_DEFINED
#undef yyalloc
#endif
#ifndef libconfig_yyrealloc_ALREADY_DEFINED
#undef yyrealloc
#endif
#ifndef libconfig_yyfree_ALREADY_DEFINED
#undef yyfree
#endif
#ifndef libconfig_yytext_ALREADY_DEFINED
#undef yytext
#endif
#ifndef libconfig_yyleng_ALREADY_DEFINED
#undef yyleng
#endif
#ifndef libconfig_yyin_ALREADY_DEFINED
#undef yyin
#endif
#ifndef libconfig_yyout_ALREADY_DEFINED
#undef yyout
#endif
#ifndef libconfig_yy_flex_debug_ALREADY_DEFINED
#undef yy_flex_debug
#endif
#ifndef libconfig_yylineno_ALREADY_DEFINED
#undef yylineno
#endif
#ifndef libconfig_yytables_fload_ALREADY_DEFINED
#undef yytables_fload
#endif
#ifndef libconfig_yytables_destroy_ALREADY_DEFINED
#undef yytables_destroy
#endif
#ifndef libconfig_yyTABLES_NAME_ALREADY_DEFINED
#undef yyTABLES_NAME
#endif

#line 203 "scanner.l"

#line 717 "scanner.h"
#undef libconfig_yyIN_HEADER
#endif /* libconfig_yyHEADER_H */
/* end libconfig/lib/scanner.h */
#ifndef YY_BUFFER_NEW
#define YY_BUFFER_NEW 0
#define YY_BUFFER_NORMAL 1
#define YY_BUFFER_EOF_PENDING 2
#endif

/* begin libconfig/lib/strbuf.c */
/* ----------------------------------------------------------------------------
   libconfig - A library for processing structured configuration files
   Copyright (C) 2005-2020  Mark A Lindner

   This file is part of libconfig.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License
   as published by the Free Software Foundation; either version 2.1 of
   the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Library General Public
   License along with this library; if not, see
   <http://www.gnu.org/licenses/>.
   ----------------------------------------------------------------------------
*/


#include <string.h>
#include <stdlib.h>

#define STRING_BLOCK_SIZE 64

/* ------------------------------------------------------------------------- */

void libconfig_strbuf_ensure_capacity(strbuf_t *buf, size_t len)
{
  static const size_t mask = ~(STRING_BLOCK_SIZE - 1);

  size_t newlen = buf->length + len + 1; /* add 1 for NUL */
  if(newlen > buf->capacity)
  {
    buf->capacity = (newlen + (STRING_BLOCK_SIZE - 1)) & mask;
    buf->string = (char *)realloc(buf->string, buf->capacity);
  }
}

/* ------------------------------------------------------------------------- */

char *libconfig_strbuf_release(strbuf_t *buf)
{
  char *r = buf->string;
  __zero(buf);
  return(r);
}

/* ------------------------------------------------------------------------- */

void libconfig_strbuf_append_string(strbuf_t *buf, const char *s)
{
  size_t len = strlen(s);
  libconfig_strbuf_ensure_capacity(buf, len);
  strcpy(buf->string + buf->length, s);
  buf->length += len;
}

/* ------------------------------------------------------------------------- */

void libconfig_strbuf_append_char(strbuf_t *buf, char c)
{
  libconfig_strbuf_ensure_capacity(buf, 1);
  *(buf->string + buf->length) = c;
  ++(buf->length);
  *(buf->string + buf->length) = '\0';
}

/* ------------------------------------------------------------------------- */
/* end libconfig/lib/strbuf.c */
/* begin libconfig/lib/strvec.c */
/* ----------------------------------------------------------------------------
   libconfig - A library for processing structured configuration files
   Copyright (C) 2005-2020  Mark A Lindner

   This file is part of libconfig.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License
   as published by the Free Software Foundation; either version 2.1 of
   the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Library General Public
   License along with this library; if not, see
   <http://www.gnu.org/licenses/>.
   ----------------------------------------------------------------------------
*/


#include <stdlib.h>

#define CHUNK_SIZE 32

/* ------------------------------------------------------------------------- */

void libconfig_strvec_append(strvec_t *vec, const char *s)
{
  if(vec->length == vec->capacity)
  {
    vec->capacity += CHUNK_SIZE;
    vec->strings = (const char **)realloc(
        (void *)vec->strings,
        (vec->capacity + 1) * sizeof(const char *));
    vec->end = vec->strings + vec->length;
  }

  *(vec->end) = s;
  ++(vec->end);
  ++(vec->length);
}

/* ------------------------------------------------------------------------- */

const char **libconfig_strvec_release(strvec_t *vec)
{
  const char **r = vec->strings;
  if(r)
    *(vec->end) = NULL;

  __zero(vec);
  return(r);
}

/* ------------------------------------------------------------------------- */

void libconfig_strvec_delete(const char *const *vec)
{
  const char *const *p;

  if(!vec) return;

  for(p = vec; *p; ++p)
    __delete(*p);

  __delete(vec);
}

/* ------------------------------------------------------------------------- */
/* end libconfig/lib/strvec.c */
/* begin libconfig/lib/util.c */
/* ----------------------------------------------------------------------------
   libconfig - A library for processing structured configuration files
   Copyright (C) 2005-2018  Mark A Lindner

   This file is part of libconfig.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License
   as published by the Free Software Foundation; either version 2.1 of
   the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Library General Public
   License along with this library; if not, see
   <http://www.gnu.org/licenses/>.
   ----------------------------------------------------------------------------
*/


#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------- */

long long libconfig_parse_integer(const char *s, int *ok)
{
  long long llval;
  char *endptr;
  int errsave = errno;
  errno = 0;
  llval = strtoll(s, &endptr, 0);	/* base 10 or base 8 */
  if(*endptr || errno)
  {
    errno = 0;
    *ok = 0;
    return(0);	/* parse error */
  }
  errno = errsave;

  *ok = 1;
  return(llval);
}

/* ------------------------------------------------------------------------- */

unsigned long long libconfig_parse_hex64(const char *s)
{
#ifdef __MINGW32__

  /* MinGW's strtoull() seems to be broken; it only returns the lower
   * 32 bits...
   */

  const char *p = s;
  unsigned long long val = 0;

  if(*p != '0')
    return(0);

  ++p;

  if(*p != 'x' && *p != 'X')
    return(0);

  for(++p; isxdigit(*p); ++p)
  {
    val <<= 4;
    val |= ((*p < 'A') ? (*p & 0xF) : (9 + (*p & 0x7)));
  }

  return(val);

#else /* ! __MINGW32__ */

  return(strtoull(s, NULL, 16));

#endif /* __MINGW32__ */
}

/* ------------------------------------------------------------------------- */

void libconfig_format_double(double val, int precision, int sci_ok, char *buf,
                             size_t buflen)
{
  const char *fmt = sci_ok ? "%.*g" : "%.*f";
  char *p, *q;

  snprintf(buf, buflen - 3, fmt, precision, val);

  /* Check for exponent. */
  p = strchr(buf, 'e');
  if(p) return;

  /* Check for decimal point. */
  p = strchr(buf, '.');
  if(!p)
  {
    /* No decimal point. Add trailing ".0". */
    strcat(buf, ".0");
  }
  else
  {
    /* Remove any excess trailing 0's after decimal point. */
    for(q = buf + strlen(buf) - 1; q > p + 1; --q)
    {
      if(*q == '0')
        *q = '\0';
      else
        break;
    }
  }
}

/* ------------------------------------------------------------------------- */
/* end libconfig/lib/util.c */
/* begin libconfig/lib/wincompat.c */
/* ----------------------------------------------------------------------------
   libconfig - A library for processing structured configuration files
   Copyright (C) 2005-2018  Mark A Lindner

   This file is part of libconfig.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License
   as published by the Free Software Foundation; either version 2.1 of
   the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Library General Public
   License along with this library; if not, see
   <http://www.gnu.org/licenses/>.
   ----------------------------------------------------------------------------
*/


#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) \
  || defined(WIN64) || defined(_WIN64) || defined(__WIN64__)

#include <errno.h>
#include <io.h>

int fsync(int fd)
{
  HANDLE h = (HANDLE)_get_osfhandle(fd);
  if(h == INVALID_HANDLE_VALUE)
  {
    errno = EBADF;
    return(-1);
  }

  if(!FlushFileBuffers(h))
  {
    DWORD err = GetLastError();
    switch(err)
    {
      case ERROR_ACCESS_DENIED:
        return(0);

      case ERROR_INVALID_HANDLE:
        errno = EINVAL;
        break;

      default:
        errno = EIO;
    }
    return(-1);
  }

  return(0);
}

#endif // WIN32 || WIN64
/* end libconfig/lib/wincompat.c */
/* begin libconfig/lib/scanctx.c */
/* ----------------------------------------------------------------------------
   libconfig - A library for processing structured configuration files
   Copyright (C) 2005-2020  Mark A Lindner

   This file is part of libconfig.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License
   as published by the Free Software Foundation; either version 2.1 of
   the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Library General Public
   License along with this library; if not, see
   <http://www.gnu.org/licenses/>.
   ----------------------------------------------------------------------------
*/


#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------- */

static const char *err_bad_include = "cannot open include file";
static const char *err_include_too_deep = "include file nesting too deep";

/* ------------------------------------------------------------------------- */

void libconfig_scanctx_init(struct scan_context *ctx, const char *top_filename)
{
  __zero(ctx);
  if(top_filename)
  {
    ctx->top_filename = strdup(top_filename);
    libconfig_strvec_append(&(ctx->filenames), ctx->top_filename);
  }
}

/* ------------------------------------------------------------------------- */

const char **libconfig_scanctx_cleanup(struct scan_context *ctx)
{
  int i;

  for(i = 0; i < ctx->stack_depth; ++i)
  {
    struct include_stack_frame *frame = &(ctx->include_stack[i]);

    if(frame->current_stream)
      fclose(frame->current_stream);

    __delete(frame->files);
  }

  __delete(libconfig_strbuf_release(&(ctx->string)));

  return(libconfig_strvec_release(&(ctx->filenames)));
}

/* ------------------------------------------------------------------------- */

FILE *libconfig_scanctx_push_include(struct scan_context *ctx, void *prev_buffer,
                                     const char *path, const char **error)
{
  struct include_stack_frame *frame;
  const char **files = NULL, **f;
  FILE *fp;

  if(ctx->stack_depth == MAX_INCLUDE_DEPTH)
  {
    *error = err_include_too_deep;
    return(NULL);
  }

  *error = NULL;

  if(ctx->config->include_fn)
    files = ctx->config->include_fn(ctx->config, ctx->config->include_dir,
                                    path, error);

  if(*error || !files)
  {
    libconfig_strvec_delete(files);
    return(NULL);
  }

  if(!*files)
  {
    libconfig_strvec_delete(files);
    return(NULL);
  }

  frame = &(ctx->include_stack[ctx->stack_depth]);

  for(f = files; *f; ++f)
    libconfig_strvec_append(&(ctx->filenames), *f);

  frame->files = files;
  frame->current_file = NULL;
  frame->current_stream = NULL;
  frame->parent_buffer = prev_buffer;
  ++(ctx->stack_depth);

  fp = libconfig_scanctx_next_include_file(ctx, error);
  if(!fp)
    (void)libconfig_scanctx_pop_include(ctx);

  return(fp);
}

/* ------------------------------------------------------------------------- */

FILE *libconfig_scanctx_next_include_file(struct scan_context *ctx,
                                          const char **error)
{
  struct include_stack_frame *include_frame;

  *error = NULL;

  if(ctx->stack_depth == 0)
    return(NULL);

  include_frame = &(ctx->include_stack[ctx->stack_depth - 1]);

  if(include_frame->current_file)
    ++(include_frame->current_file);
  else
    include_frame->current_file = include_frame->files;

  if(include_frame->current_stream)
  {
    fclose(include_frame->current_stream);
    include_frame->current_stream = NULL;
  }

  if(!*(include_frame->current_file))
    return(NULL);

  include_frame->current_stream = fopen(*(include_frame->current_file), "rt");
  if(!include_frame->current_stream)
    *error = err_bad_include;

  return(include_frame->current_stream);
}

/* ------------------------------------------------------------------------- */

void *libconfig_scanctx_pop_include(struct scan_context *ctx)
{
  struct include_stack_frame *frame;

  if(ctx->stack_depth == 0)
    return(NULL); /* stack underflow */

  frame = &(ctx->include_stack[--(ctx->stack_depth)]);

  __delete(frame->files);
  frame->files = NULL;

  if(frame->current_stream)
  {
    fclose(frame->current_stream);
    frame->current_stream = NULL;
  }

  return(frame->parent_buffer);
}

/* ------------------------------------------------------------------------- */

char *libconfig_scanctx_take_string(struct scan_context *ctx)
{
  char *r = libconfig_strbuf_release(&(ctx->string));

  return(r ? r : strdup(""));
}

/* ------------------------------------------------------------------------- */

const char *libconfig_scanctx_current_filename(struct scan_context *ctx)
{
  if(ctx->stack_depth > 0)
    return(*(ctx->include_stack[ctx->stack_depth - 1].current_file));

  return(ctx->top_filename);
}

/* ------------------------------------------------------------------------- */
/* end libconfig/lib/scanctx.c */
/* begin libconfig/lib/grammar.c */
/* A Bison parser, made by GNU Bison 3.0.4.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "3.0.4"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 1

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1


/* Substitute the variable and function names.  */
#define yyparse         libconfig_yyparse
#define yylex           libconfig_yylex
#define yyerror         libconfig_yyerror
#define yydebug         libconfig_yydebug
#define yynerrs         libconfig_yynerrs


/* Copy the first part of user declarations.  */
#line 32 "grammar.y" /* yacc.c:339  */

#include <string.h>
#include <stdlib.h>


/* These declarations are provided to suppress compiler warnings. */
extern int libconfig_yylex();
extern int libconfig_yyget_lineno();

static const char *err_array_elem_type = "mismatched element type in array";
static const char *err_duplicate_setting = "duplicate setting name";

#define IN_ARRAY() \
  (ctx->parent && (ctx->parent->type == CONFIG_TYPE_ARRAY))

#define IN_LIST() \
  (ctx->parent && (ctx->parent->type == CONFIG_TYPE_LIST))

static void capture_parse_pos(void *scanner, struct scan_context *scan_ctx,
                              config_setting_t *setting)
{
  setting->line = (unsigned int)libconfig_yyget_lineno(scanner);
  setting->file = libconfig_scanctx_current_filename(scan_ctx);
}

#define CAPTURE_PARSE_POS(S) \
  capture_parse_pos(scanner, scan_ctx, (S))

void libconfig_yyerror(void *scanner, struct parse_context *ctx,
                       struct scan_context *scan_ctx, char const *s)
{
  if(ctx->config->error_text) return;
  ctx->config->error_line = libconfig_yyget_lineno(scanner);
  ctx->config->error_text = s;
}


#line 115 "grammar.c" /* yacc.c:339  */

# ifndef YY_NULLPTR
#  if defined __cplusplus && 201103L <= __cplusplus
#   define YY_NULLPTR nullptr
#  else
#   define YY_NULLPTR 0
#  endif
# endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* In a future release of Bison, this section will be replaced
   by #include "y.tab.h".  */
#ifndef YY_LIBCONFIG_YY_GRAMMAR_H_INCLUDED
# define YY_LIBCONFIG_YY_GRAMMAR_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int libconfig_yydebug;
#endif

/* Token type.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    TOK_BOOLEAN = 258,
    TOK_INTEGER = 259,
    TOK_HEX = 260,
    TOK_INTEGER64 = 261,
    TOK_HEX64 = 262,
    TOK_FLOAT = 263,
    TOK_STRING = 264,
    TOK_NAME = 265,
    TOK_EQUALS = 266,
    TOK_NEWLINE = 267,
    TOK_ARRAY_START = 268,
    TOK_ARRAY_END = 269,
    TOK_LIST_START = 270,
    TOK_LIST_END = 271,
    TOK_COMMA = 272,
    TOK_GROUP_START = 273,
    TOK_GROUP_END = 274,
    TOK_SEMICOLON = 275,
    TOK_GARBAGE = 276,
    TOK_ERROR = 277
  };
#endif
/* Tokens.  */
#define TOK_BOOLEAN 258
#define TOK_INTEGER 259
#define TOK_HEX 260
#define TOK_INTEGER64 261
#define TOK_HEX64 262
#define TOK_FLOAT 263
#define TOK_STRING 264
#define TOK_NAME 265
#define TOK_EQUALS 266
#define TOK_NEWLINE 267
#define TOK_ARRAY_START 268
#define TOK_ARRAY_END 269
#define TOK_LIST_START 270
#define TOK_LIST_END 271
#define TOK_COMMA 272
#define TOK_GROUP_START 273
#define TOK_GROUP_END 274
#define TOK_SEMICOLON 275
#define TOK_GARBAGE 276
#define TOK_ERROR 277

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED

union YYSTYPE
{
#line 76 "grammar.y" /* yacc.c:355  */

  int ival;
  long long llval;
  double fval;
  char *sval;

#line 206 "grammar.c" /* yacc.c:355  */
};

typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif



int libconfig_yyparse (void *scanner, struct parse_context *ctx, struct scan_context *scan_ctx);

#endif /* !YY_LIBCONFIG_YY_GRAMMAR_H_INCLUDED  */

/* Copy the second part of user declarations.  */

#line 222 "grammar.c" /* yacc.c:358  */

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif

#ifndef YY_ATTRIBUTE
# if (defined __GNUC__                                               \
      && (2 < __GNUC__ || (__GNUC__ == 2 && 96 <= __GNUC_MINOR__)))  \
     || defined __SUNPRO_C && 0x5110 <= __SUNPRO_C
#  define YY_ATTRIBUTE(Spec) __attribute__(Spec)
# else
#  define YY_ATTRIBUTE(Spec) /* empty */
# endif
#endif

#ifndef YY_ATTRIBUTE_PURE
# define YY_ATTRIBUTE_PURE   YY_ATTRIBUTE ((__pure__))
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# define YY_ATTRIBUTE_UNUSED YY_ATTRIBUTE ((__unused__))
#endif

#if !defined _Noreturn \
     && (!defined __STDC_VERSION__ || __STDC_VERSION__ < 201112)
# if defined _MSC_VER && 1200 <= _MSC_VER
#  define _Noreturn __declspec (noreturn)
# else
#  define _Noreturn YY_ATTRIBUTE ((__noreturn__))
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif

#if defined __GNUC__ && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN \
    _Pragma ("GCC diagnostic push") \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")\
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define YY_IGNORE_MAYBE_UNINITIALIZED_END \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif


#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYSIZE_T yynewbytes;                                            \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / sizeof (*yyptr);                          \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, (Count) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYSIZE_T yyi;                         \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  6
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   35

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  23
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  20
/* YYNRULES -- Number of rules.  */
#define YYNRULES  41
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  47

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   277

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,    92,    92,    94,    98,    99,   102,   104,   107,   109,
     110,   115,   114,   134,   133,   157,   156,   179,   180,   181,
     182,   186,   187,   191,   211,   233,   255,   277,   299,   317,
     345,   346,   347,   350,   352,   356,   357,   358,   361,   363,
     368,   367
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || 0
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "TOK_BOOLEAN", "TOK_INTEGER", "TOK_HEX",
  "TOK_INTEGER64", "TOK_HEX64", "TOK_FLOAT", "TOK_STRING", "TOK_NAME",
  "TOK_EQUALS", "TOK_NEWLINE", "TOK_ARRAY_START", "TOK_ARRAY_END",
  "TOK_LIST_START", "TOK_LIST_END", "TOK_COMMA", "TOK_GROUP_START",
  "TOK_GROUP_END", "TOK_SEMICOLON", "TOK_GARBAGE", "TOK_ERROR", "$accept",
  "configuration", "setting_list", "setting_list_optional",
  "setting_terminator", "setting", "$@1", "array", "$@2", "list", "$@3",
  "value", "string", "simple_value", "value_list", "value_list_optional",
  "simple_value_list", "simple_value_list_optional", "group", "$@4", YY_NULLPTR
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277
};
# endif

#define YYPACT_NINF -26

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-26)))

#define YYTABLE_NINF -1

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int8 yypact[] =
{
      -8,   -26,    12,    -8,   -26,     5,   -26,   -26,     0,   -26,
     -26,   -26,   -26,   -26,   -26,   -26,   -26,   -26,   -26,   -26,
     -26,    -6,    10,   -26,   -26,    23,     0,    -8,   -26,   -26,
     -26,   -26,   -26,     3,     7,   -26,     6,     8,    -8,    14,
      23,   -26,     0,   -26,   -26,   -26,   -26
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       2,    11,     0,     3,     4,     0,     1,     5,     0,    23,
      24,    26,    25,    27,    28,    21,    13,    15,    40,    18,
      19,     8,    29,    17,    20,    38,    33,     6,    10,     9,
      12,    22,    35,    39,     0,    30,    34,     0,     7,     0,
      37,    14,    32,    16,    41,    36,    31
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int8 yypgoto[] =
{
     -26,   -26,    -5,   -26,   -26,    -3,   -26,   -26,   -26,   -26,
     -26,   -25,   -26,   -15,   -26,   -26,   -26,   -26,   -26,   -26
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int8 yydefgoto[] =
{
      -1,     2,     3,    39,    30,     4,     5,    19,    25,    20,
      26,    21,    22,    23,    36,    37,    33,    34,    24,    27
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_uint8 yytable[] =
{
       7,    35,     1,     9,    10,    11,    12,    13,    14,    15,
      32,    28,     6,    16,    29,    17,     8,    46,    18,    31,
      40,    41,    38,    42,    43,    45,     9,    10,    11,    12,
      13,    14,    15,    44,     0,     7
};

static const yytype_int8 yycheck[] =
{
       3,    26,    10,     3,     4,     5,     6,     7,     8,     9,
      25,    17,     0,    13,    20,    15,    11,    42,    18,     9,
      17,    14,    27,    17,    16,    40,     3,     4,     5,     6,
       7,     8,     9,    19,    -1,    38
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    10,    24,    25,    28,    29,     0,    28,    11,     3,
       4,     5,     6,     7,     8,     9,    13,    15,    18,    30,
      32,    34,    35,    36,    41,    31,    33,    42,    17,    20,
      27,     9,    36,    39,    40,    34,    37,    38,    25,    26,
      17,    14,    17,    16,    19,    36,    34
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    23,    24,    24,    25,    25,    26,    26,    27,    27,
      27,    29,    28,    31,    30,    33,    32,    34,    34,    34,
      34,    35,    35,    36,    36,    36,    36,    36,    36,    36,
      37,    37,    37,    38,    38,    39,    39,    39,    40,    40,
      42,    41
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     1,     1,     2,     0,     1,     0,     1,
       1,     0,     5,     0,     4,     0,     4,     1,     1,     1,
       1,     1,     2,     1,     1,     1,     1,     1,     1,     1,
       1,     3,     2,     0,     1,     1,     3,     2,     0,     1,
       0,     4
};


#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)
#define YYEMPTY         (-2)
#define YYEOF           0

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                  \
do                                                              \
  if (yychar == YYEMPTY)                                        \
    {                                                           \
      yychar = (Token);                                         \
      yylval = (Value);                                         \
      YYPOPSTACK (yylen);                                       \
      yystate = *yyssp;                                         \
      goto yybackup;                                            \
    }                                                           \
  else                                                          \
    {                                                           \
      yyerror (scanner, ctx, scan_ctx, YY_("syntax error: cannot back up")); \
      YYERROR;                                                  \
    }                                                           \
while (0)

/* Error token number */
#define YYTERROR        1
#define YYERRCODE       256



/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)

/* This macro is provided for backward compatibility. */
#ifndef YY_LOCATION_PRINT
# define YY_LOCATION_PRINT(File, Loc) ((void) 0)
#endif


# define YY_SYMBOL_PRINT(Title, Type, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Type, Value, scanner, ctx, scan_ctx); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*----------------------------------------.
| Print this symbol's value on YYOUTPUT.  |
`----------------------------------------*/

static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep, void *scanner, struct parse_context *ctx, struct scan_context *scan_ctx)
{
  FILE *yyo = yyoutput;
  YYUSE (yyo);
  YYUSE (scanner);
  YYUSE (ctx);
  YYUSE (scan_ctx);
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# endif
  YYUSE (yytype);
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep, void *scanner, struct parse_context *ctx, struct scan_context *scan_ctx)
{
  YYFPRINTF (yyoutput, "%s %s (",
             yytype < YYNTOKENS ? "token" : "nterm", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep, scanner, ctx, scan_ctx);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yytype_int16 *yyssp, YYSTYPE *yyvsp, int yyrule, void *scanner, struct parse_context *ctx, struct scan_context *scan_ctx)
{
  unsigned long int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       yystos[yyssp[yyi + 1 - yynrhs]],
                       &(yyvsp[(yyi + 1) - (yynrhs)])
                                              , scanner, ctx, scan_ctx);
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule, scanner, ctx, scan_ctx); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif


#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
static YYSIZE_T
yystrlen (const char *yystr)
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
yystpcpy (char *yydest, const char *yysrc)
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
        switch (*++yyp)
          {
          case '\'':
          case ',':
            goto do_not_strip_quotes;

          case '\\':
            if (*++yyp != '\\')
              goto do_not_strip_quotes;
            /* Fall through.  */
          default:
            if (yyres)
              yyres[yyn] = *yyp;
            yyn++;
            break;

          case '"':
            if (yyres)
              yyres[yyn] = '\0';
            return yyn;
          }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return 1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store.  */
static int
yysyntax_error (YYSIZE_T *yymsg_alloc, char **yymsg,
                yytype_int16 *yyssp, int yytoken)
{
  YYSIZE_T yysize0 = yytnamerr (YY_NULLPTR, yytname[yytoken]);
  YYSIZE_T yysize = yysize0;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULLPTR;
  /* Arguments of yyformat. */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Number of reported tokens (one for the "unexpected", one per
     "expected"). */
  int yycount = 0;

  /* There are many possibilities here to consider:
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yytoken != YYEMPTY)
    {
      int yyn = yypact[*yyssp];
      yyarg[yycount++] = yytname[yytoken];
      if (!yypact_value_is_default (yyn))
        {
          /* Start YYX at -YYN if negative to avoid negative indexes in
             YYCHECK.  In other words, skip the first -YYN actions for
             this state because they are default actions.  */
          int yyxbegin = yyn < 0 ? -yyn : 0;
          /* Stay within bounds of both yycheck and yytname.  */
          int yychecklim = YYLAST - yyn + 1;
          int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
          int yyx;

          for (yyx = yyxbegin; yyx < yyxend; ++yyx)
            if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR
                && !yytable_value_is_error (yytable[yyx + yyn]))
              {
                if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
                  {
                    yycount = 1;
                    yysize = yysize0;
                    break;
                  }
                yyarg[yycount++] = yytname[yyx];
                {
                  YYSIZE_T yysize1 = yysize + yytnamerr (YY_NULLPTR, yytname[yyx]);
                  if (! (yysize <= yysize1
                         && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
                    return 2;
                  yysize = yysize1;
                }
              }
        }
    }

  switch (yycount)
    {
# define YYCASE_(N, S)                      \
      case N:                               \
        yyformat = S;                       \
      break
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
# undef YYCASE_
    }

  {
    YYSIZE_T yysize1 = yysize + yystrlen (yyformat);
    if (! (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
      return 2;
    yysize = yysize1;
  }

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return 1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
        {
          yyp += yytnamerr (yyp, yyarg[yyi++]);
          yyformat += 2;
        }
      else
        {
          yyp++;
          yyformat++;
        }
  }
  return 0;
}
#endif /* YYERROR_VERBOSE */

/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep, void *scanner, struct parse_context *ctx, struct scan_context *scan_ctx)
{
  YYUSE (yyvaluep);
  YYUSE (scanner);
  YYUSE (ctx);
  YYUSE (scan_ctx);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  switch (yytype)
    {
          case 9: /* TOK_STRING  */
#line 88 "grammar.y" /* yacc.c:1257  */
      { free(((*yyvaluep).sval)); }
#line 1083 "grammar.c" /* yacc.c:1257  */
        break;


      default:
        break;
    }
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}




/*----------.
| yyparse.  |
`----------*/

int
yyparse (void *scanner, struct parse_context *ctx, struct scan_context *scan_ctx)
{
/* The lookahead symbol.  */
int yychar;


/* The semantic value of the lookahead symbol.  */
/* Default value used for initialization, for pacifying older GCCs
   or non-GCC compilers.  */
YY_INITIAL_VALUE (static YYSTYPE yyval_default;)
YYSTYPE yylval YY_INITIAL_VALUE (= yyval_default);

    /* Number of syntax errors so far.  */
    int yynerrs;

    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       'yyss': related to states.
       'yyvs': related to semantic values.

       Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken = 0;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yyssp = yyss = yyssa;
  yyvsp = yyvs = yyvsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */
  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        YYSTYPE *yyvs1 = yyvs;
        yytype_int16 *yyss1 = yyss;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * sizeof (*yyssp),
                    &yyvs1, yysize * sizeof (*yyvsp),
                    &yystacksize);

        yyss = yyss1;
        yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yytype_int16 *yyss1 = yyss;
        union yyalloc *yyptr =
          (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
        if (! yyptr)
          goto yyexhaustedlab;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
                  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = yylex (&yylval, scanner);
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
  yychar = YYEMPTY;

  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 11:
#line 115 "grammar.y" /* yacc.c:1646  */
    {
    ctx->setting = config_setting_add(ctx->parent, (yyvsp[0].sval), CONFIG_TYPE_NONE);

    if(ctx->setting == NULL)
    {
      libconfig_yyerror(scanner, ctx, scan_ctx, err_duplicate_setting);
      YYABORT;
    }
    else
    {
      CAPTURE_PARSE_POS(ctx->setting);
    }
  }
#line 1363 "grammar.c" /* yacc.c:1646  */
    break;

  case 13:
#line 134 "grammar.y" /* yacc.c:1646  */
    {
    if(IN_LIST())
    {
      ctx->parent = config_setting_add(ctx->parent, NULL, CONFIG_TYPE_ARRAY);
      CAPTURE_PARSE_POS(ctx->parent);
    }
    else
    {
      ctx->setting->type = CONFIG_TYPE_ARRAY;
      ctx->parent = ctx->setting;
      ctx->setting = NULL;
    }
  }
#line 1381 "grammar.c" /* yacc.c:1646  */
    break;

  case 14:
#line 149 "grammar.y" /* yacc.c:1646  */
    {
    if(ctx->parent)
      ctx->parent = ctx->parent->parent;
  }
#line 1390 "grammar.c" /* yacc.c:1646  */
    break;

  case 15:
#line 157 "grammar.y" /* yacc.c:1646  */
    {
    if(IN_LIST())
    {
      ctx->parent = config_setting_add(ctx->parent, NULL, CONFIG_TYPE_LIST);
      CAPTURE_PARSE_POS(ctx->parent);
    }
    else
    {
      ctx->setting->type = CONFIG_TYPE_LIST;
      ctx->parent = ctx->setting;
      ctx->setting = NULL;
    }
  }
#line 1408 "grammar.c" /* yacc.c:1646  */
    break;

  case 16:
#line 172 "grammar.y" /* yacc.c:1646  */
    {
    if(ctx->parent)
      ctx->parent = ctx->parent->parent;
  }
#line 1417 "grammar.c" /* yacc.c:1646  */
    break;

  case 21:
#line 186 "grammar.y" /* yacc.c:1646  */
    { libconfig_parsectx_append_string(ctx, (yyvsp[0].sval)); free((yyvsp[0].sval)); }
#line 1423 "grammar.c" /* yacc.c:1646  */
    break;

  case 22:
#line 187 "grammar.y" /* yacc.c:1646  */
    { libconfig_parsectx_append_string(ctx, (yyvsp[0].sval)); free((yyvsp[0].sval)); }
#line 1429 "grammar.c" /* yacc.c:1646  */
    break;

  case 23:
#line 192 "grammar.y" /* yacc.c:1646  */
    {
    if(IN_ARRAY() || IN_LIST())
    {
      config_setting_t *e = config_setting_set_bool_elem(ctx->parent, -1,
                                                         (int)(yyvsp[0].ival));

      if(! e)
      {
        libconfig_yyerror(scanner, ctx, scan_ctx, err_array_elem_type);
        YYABORT;
      }
      else
      {
        CAPTURE_PARSE_POS(e);
      }
    }
    else
      config_setting_set_bool(ctx->setting, (int)(yyvsp[0].ival));
  }
#line 1453 "grammar.c" /* yacc.c:1646  */
    break;

  case 24:
#line 212 "grammar.y" /* yacc.c:1646  */
    {
    if(IN_ARRAY() || IN_LIST())
    {
      config_setting_t *e = config_setting_set_int_elem(ctx->parent, -1, (yyvsp[0].ival));
      if(! e)
      {
        libconfig_yyerror(scanner, ctx, scan_ctx, err_array_elem_type);
        YYABORT;
      }
      else
      {
        config_setting_set_format(e, CONFIG_FORMAT_DEFAULT);
        CAPTURE_PARSE_POS(e);
      }
    }
    else
    {
      config_setting_set_int(ctx->setting, (yyvsp[0].ival));
      config_setting_set_format(ctx->setting, CONFIG_FORMAT_DEFAULT);
    }
  }
#line 1479 "grammar.c" /* yacc.c:1646  */
    break;

  case 25:
#line 234 "grammar.y" /* yacc.c:1646  */
    {
    if(IN_ARRAY() || IN_LIST())
    {
      config_setting_t *e = config_setting_set_int64_elem(ctx->parent, -1, (yyvsp[0].llval));
      if(! e)
      {
        libconfig_yyerror(scanner, ctx, scan_ctx, err_array_elem_type);
        YYABORT;
      }
      else
      {
        config_setting_set_format(e, CONFIG_FORMAT_DEFAULT);
        CAPTURE_PARSE_POS(e);
      }
    }
    else
    {
      config_setting_set_int64(ctx->setting, (yyvsp[0].llval));
      config_setting_set_format(ctx->setting, CONFIG_FORMAT_DEFAULT);
    }
  }
#line 1505 "grammar.c" /* yacc.c:1646  */
    break;

  case 26:
#line 256 "grammar.y" /* yacc.c:1646  */
    {
    if(IN_ARRAY() || IN_LIST())
    {
      config_setting_t *e = config_setting_set_int_elem(ctx->parent, -1, (yyvsp[0].ival));
      if(! e)
      {
        libconfig_yyerror(scanner, ctx, scan_ctx, err_array_elem_type);
        YYABORT;
      }
      else
      {
        config_setting_set_format(e, CONFIG_FORMAT_HEX);
        CAPTURE_PARSE_POS(e);
      }
    }
    else
    {
      config_setting_set_int(ctx->setting, (yyvsp[0].ival));
      config_setting_set_format(ctx->setting, CONFIG_FORMAT_HEX);
    }
  }
#line 1531 "grammar.c" /* yacc.c:1646  */
    break;

  case 27:
#line 278 "grammar.y" /* yacc.c:1646  */
    {
    if(IN_ARRAY() || IN_LIST())
    {
      config_setting_t *e = config_setting_set_int64_elem(ctx->parent, -1, (yyvsp[0].llval));
      if(! e)
      {
        libconfig_yyerror(scanner, ctx, scan_ctx, err_array_elem_type);
        YYABORT;
      }
      else
      {
        config_setting_set_format(e, CONFIG_FORMAT_HEX);
        CAPTURE_PARSE_POS(e);
      }
    }
    else
    {
      config_setting_set_int64(ctx->setting, (yyvsp[0].llval));
      config_setting_set_format(ctx->setting, CONFIG_FORMAT_HEX);
    }
  }
#line 1557 "grammar.c" /* yacc.c:1646  */
    break;

  case 28:
#line 300 "grammar.y" /* yacc.c:1646  */
    {
    if(IN_ARRAY() || IN_LIST())
    {
      config_setting_t *e = config_setting_set_float_elem(ctx->parent, -1, (yyvsp[0].fval));
      if(! e)
      {
        libconfig_yyerror(scanner, ctx, scan_ctx, err_array_elem_type);
        YYABORT;
      }
      else
      {
        CAPTURE_PARSE_POS(e);
      }
    }
    else
      config_setting_set_float(ctx->setting, (yyvsp[0].fval));
  }
#line 1579 "grammar.c" /* yacc.c:1646  */
    break;

  case 29:
#line 318 "grammar.y" /* yacc.c:1646  */
    {
    if(IN_ARRAY() || IN_LIST())
    {
      const char *s = libconfig_parsectx_take_string(ctx);
      config_setting_t *e = config_setting_set_string_elem(ctx->parent, -1, s);
      __delete(s);

      if(! e)
      {
        libconfig_yyerror(scanner, ctx, scan_ctx, err_array_elem_type);
        YYABORT;
      }
      else
      {
        CAPTURE_PARSE_POS(e);
      }
    }
    else
    {
      const char *s = libconfig_parsectx_take_string(ctx);
      config_setting_set_string(ctx->setting, s);
      __delete(s);
    }
  }
#line 1608 "grammar.c" /* yacc.c:1646  */
    break;

  case 40:
#line 368 "grammar.y" /* yacc.c:1646  */
    {
    if(IN_LIST())
    {
      ctx->parent = config_setting_add(ctx->parent, NULL, CONFIG_TYPE_GROUP);
      CAPTURE_PARSE_POS(ctx->parent);
    }
    else
    {
      ctx->setting->type = CONFIG_TYPE_GROUP;
      ctx->parent = ctx->setting;
      ctx->setting = NULL;
    }
  }
#line 1626 "grammar.c" /* yacc.c:1646  */
    break;

  case 41:
#line 383 "grammar.y" /* yacc.c:1646  */
    {
    if(ctx->parent)
      ctx->parent = ctx->parent->parent;
  }
#line 1635 "grammar.c" /* yacc.c:1646  */
    break;


#line 1639 "grammar.c" /* yacc.c:1646  */
      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYEMPTY : YYTRANSLATE (yychar);

  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (scanner, ctx, scan_ctx, YY_("syntax error"));
#else
# define YYSYNTAX_ERROR yysyntax_error (&yymsg_alloc, &yymsg, \
                                        yyssp, yytoken)
      {
        char const *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        yysyntax_error_status = YYSYNTAX_ERROR;
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == 1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = (char *) YYSTACK_ALLOC (yymsg_alloc);
            if (!yymsg)
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = 2;
              }
            else
              {
                yysyntax_error_status = YYSYNTAX_ERROR;
                yymsgp = yymsg;
              }
          }
        yyerror (scanner, ctx, scan_ctx, yymsgp);
        if (yysyntax_error_status == 2)
          goto yyexhaustedlab;
      }
# undef YYSYNTAX_ERROR
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval, scanner, ctx, scan_ctx);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYTERROR;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;


      yydestruct ("Error: popping",
                  yystos[yystate], yyvsp, scanner, ctx, scan_ctx);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#if !defined yyoverflow || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (scanner, ctx, scan_ctx, YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval, scanner, ctx, scan_ctx);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  yystos[*yyssp], yyvsp, scanner, ctx, scan_ctx);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  return yyresult;
}
#line 389 "grammar.y" /* yacc.c:1906  */

/* end libconfig/lib/grammar.c */
/* begin libconfig/lib/scanner.c */
#line 1 "scanner.c"

#line 3 "scanner.c"

#define  YY_INT_ALIGNED short int

/* A lexical scanner generated by flex */

#define FLEX_SCANNER
#define YY_FLEX_MAJOR_VERSION 2
#define YY_FLEX_MINOR_VERSION 6
#define YY_FLEX_SUBMINOR_VERSION 4
#if YY_FLEX_SUBMINOR_VERSION > 0
#define FLEX_BETA
#endif

#ifdef yy_create_buffer
#define libconfig_yy_create_buffer_ALREADY_DEFINED
#else
#define yy_create_buffer libconfig_yy_create_buffer
#endif

#ifdef yy_delete_buffer
#define libconfig_yy_delete_buffer_ALREADY_DEFINED
#else
#define yy_delete_buffer libconfig_yy_delete_buffer
#endif

#ifdef yy_scan_buffer
#define libconfig_yy_scan_buffer_ALREADY_DEFINED
#else
#define yy_scan_buffer libconfig_yy_scan_buffer
#endif

#ifdef yy_scan_string
#define libconfig_yy_scan_string_ALREADY_DEFINED
#else
#define yy_scan_string libconfig_yy_scan_string
#endif

#ifdef yy_scan_bytes
#define libconfig_yy_scan_bytes_ALREADY_DEFINED
#else
#define yy_scan_bytes libconfig_yy_scan_bytes
#endif

#ifdef yy_init_buffer
#define libconfig_yy_init_buffer_ALREADY_DEFINED
#else
#define yy_init_buffer libconfig_yy_init_buffer
#endif

#ifdef yy_flush_buffer
#define libconfig_yy_flush_buffer_ALREADY_DEFINED
#else
#define yy_flush_buffer libconfig_yy_flush_buffer
#endif

#ifdef yy_load_buffer_state
#define libconfig_yy_load_buffer_state_ALREADY_DEFINED
#else
#define yy_load_buffer_state libconfig_yy_load_buffer_state
#endif

#ifdef yy_switch_to_buffer
#define libconfig_yy_switch_to_buffer_ALREADY_DEFINED
#else
#define yy_switch_to_buffer libconfig_yy_switch_to_buffer
#endif

#ifdef yypush_buffer_state
#define libconfig_yypush_buffer_state_ALREADY_DEFINED
#else
#define yypush_buffer_state libconfig_yypush_buffer_state
#endif

#ifdef yypop_buffer_state
#define libconfig_yypop_buffer_state_ALREADY_DEFINED
#else
#define yypop_buffer_state libconfig_yypop_buffer_state
#endif

#ifdef yyensure_buffer_stack
#define libconfig_yyensure_buffer_stack_ALREADY_DEFINED
#else
#define yyensure_buffer_stack libconfig_yyensure_buffer_stack
#endif

#ifdef yylex
#define libconfig_yylex_ALREADY_DEFINED
#else
#define yylex libconfig_yylex
#endif

#ifdef yyrestart
#define libconfig_yyrestart_ALREADY_DEFINED
#else
#define yyrestart libconfig_yyrestart
#endif

#ifdef yylex_init
#define libconfig_yylex_init_ALREADY_DEFINED
#else
#define yylex_init libconfig_yylex_init
#endif

#ifdef yylex_init_extra
#define libconfig_yylex_init_extra_ALREADY_DEFINED
#else
#define yylex_init_extra libconfig_yylex_init_extra
#endif

#ifdef yylex_destroy
#define libconfig_yylex_destroy_ALREADY_DEFINED
#else
#define yylex_destroy libconfig_yylex_destroy
#endif

#ifdef yyget_debug
#define libconfig_yyget_debug_ALREADY_DEFINED
#else
#define yyget_debug libconfig_yyget_debug
#endif

#ifdef yyset_debug
#define libconfig_yyset_debug_ALREADY_DEFINED
#else
#define yyset_debug libconfig_yyset_debug
#endif

#ifdef yyget_extra
#define libconfig_yyget_extra_ALREADY_DEFINED
#else
#define yyget_extra libconfig_yyget_extra
#endif

#ifdef yyset_extra
#define libconfig_yyset_extra_ALREADY_DEFINED
#else
#define yyset_extra libconfig_yyset_extra
#endif

#ifdef yyget_in
#define libconfig_yyget_in_ALREADY_DEFINED
#else
#define yyget_in libconfig_yyget_in
#endif

#ifdef yyset_in
#define libconfig_yyset_in_ALREADY_DEFINED
#else
#define yyset_in libconfig_yyset_in
#endif

#ifdef yyget_out
#define libconfig_yyget_out_ALREADY_DEFINED
#else
#define yyget_out libconfig_yyget_out
#endif

#ifdef yyset_out
#define libconfig_yyset_out_ALREADY_DEFINED
#else
#define yyset_out libconfig_yyset_out
#endif

#ifdef yyget_leng
#define libconfig_yyget_leng_ALREADY_DEFINED
#else
#define yyget_leng libconfig_yyget_leng
#endif

#ifdef yyget_text
#define libconfig_yyget_text_ALREADY_DEFINED
#else
#define yyget_text libconfig_yyget_text
#endif

#ifdef yyget_lineno
#define libconfig_yyget_lineno_ALREADY_DEFINED
#else
#define yyget_lineno libconfig_yyget_lineno
#endif

#ifdef yyset_lineno
#define libconfig_yyset_lineno_ALREADY_DEFINED
#else
#define yyset_lineno libconfig_yyset_lineno
#endif

#ifdef yyget_column
#define libconfig_yyget_column_ALREADY_DEFINED
#else
#define yyget_column libconfig_yyget_column
#endif

#ifdef yyset_column
#define libconfig_yyset_column_ALREADY_DEFINED
#else
#define yyset_column libconfig_yyset_column
#endif

#ifdef yywrap
#define libconfig_yywrap_ALREADY_DEFINED
#else
#define yywrap libconfig_yywrap
#endif

#ifdef yyget_lval
#define libconfig_yyget_lval_ALREADY_DEFINED
#else
#define yyget_lval libconfig_yyget_lval
#endif

#ifdef yyset_lval
#define libconfig_yyset_lval_ALREADY_DEFINED
#else
#define yyset_lval libconfig_yyset_lval
#endif

#ifdef yyalloc
#define libconfig_yyalloc_ALREADY_DEFINED
#else
#define yyalloc libconfig_yyalloc
#endif

#ifdef yyrealloc
#define libconfig_yyrealloc_ALREADY_DEFINED
#else
#define yyrealloc libconfig_yyrealloc
#endif

#ifdef yyfree
#define libconfig_yyfree_ALREADY_DEFINED
#else
#define yyfree libconfig_yyfree
#endif

/* First, we deal with  platform-specific or compiler-specific issues. */

/* begin standard C headers. */
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>

/* end standard C headers. */

/* flex integer type definitions */

#ifndef FLEXINT_H
#define FLEXINT_H

/* C99 systems have <inttypes.h>. Non-C99 systems may or may not. */

#if defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L

/* C99 says to define __STDC_LIMIT_MACROS before including stdint.h,
 * if you want the limit (max/min) macros for int types.
 */
#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS 1
#endif

#include <inttypes.h>
typedef int8_t flex_int8_t;
typedef uint8_t flex_uint8_t;
typedef int16_t flex_int16_t;
typedef uint16_t flex_uint16_t;
typedef int32_t flex_int32_t;
typedef uint32_t flex_uint32_t;
#else
typedef signed char flex_int8_t;
typedef short int flex_int16_t;
typedef int flex_int32_t;
typedef unsigned char flex_uint8_t;
typedef unsigned short int flex_uint16_t;
typedef unsigned int flex_uint32_t;

/* Limits of integral types. */
#ifndef INT8_MIN
#define INT8_MIN               (-128)
#endif
#ifndef INT16_MIN
#define INT16_MIN              (-32767-1)
#endif
#ifndef INT32_MIN
#define INT32_MIN              (-2147483647-1)
#endif
#ifndef INT8_MAX
#define INT8_MAX               (127)
#endif
#ifndef INT16_MAX
#define INT16_MAX              (32767)
#endif
#ifndef INT32_MAX
#define INT32_MAX              (2147483647)
#endif
#ifndef UINT8_MAX
#define UINT8_MAX              (255U)
#endif
#ifndef UINT16_MAX
#define UINT16_MAX             (65535U)
#endif
#ifndef UINT32_MAX
#define UINT32_MAX             (4294967295U)
#endif

#ifndef SIZE_MAX
#define SIZE_MAX               (~(size_t)0)
#endif

#endif /* ! C99 */

#endif /* ! FLEXINT_H */

/* begin standard C++ headers. */

/* TODO: this is always defined, so inline it */
#define yyconst const

#if defined(__GNUC__) && __GNUC__ >= 3
#define yynoreturn __attribute__((__noreturn__))
#else
#define yynoreturn
#endif

/* Returned upon end-of-file. */
#define YY_NULL 0

/* Promotes a possibly negative, possibly signed char to an
 *   integer in range [0..255] for use as an array index.
 */
#define YY_SC_TO_UI(c) ((YY_CHAR) (c))

/* An opaque pointer. */
#ifndef YY_TYPEDEF_YY_SCANNER_T
#define YY_TYPEDEF_YY_SCANNER_T
typedef void* yyscan_t;
#endif

/* For convenience, these vars (plus the bison vars far below)
   are macros in the reentrant scanner. */
#define yyin yyg->yyin_r
#define yyout yyg->yyout_r
#define yyextra yyg->yyextra_r
#define yyleng yyg->yyleng_r
#define yytext yyg->yytext_r
#define yylineno (YY_CURRENT_BUFFER_LVALUE->yy_bs_lineno)
#define yycolumn (YY_CURRENT_BUFFER_LVALUE->yy_bs_column)
#define yy_flex_debug yyg->yy_flex_debug_r

/* Enter a start condition.  This macro really ought to take a parameter,
 * but we do it the disgusting crufty way forced on us by the ()-less
 * definition of BEGIN.
 */
#define BEGIN yyg->yy_start = 1 + 2 *
/* Translate the current start state into a value that can be later handed
 * to BEGIN to return to the state.  The YYSTATE alias is for lex
 * compatibility.
 */
#define YY_START ((yyg->yy_start - 1) / 2)
#define YYSTATE YY_START
/* Action number for EOF rule of a given start state. */
#define YY_STATE_EOF(state) (YY_END_OF_BUFFER + state + 1)
/* Special action meaning "start processing a new file". */
#define YY_NEW_FILE yyrestart( yyin , yyscanner )
#define YY_END_OF_BUFFER_CHAR 0

/* Size of default input buffer. */
#ifndef YY_BUF_SIZE
#ifdef __ia64__
/* On IA-64, the buffer size is 16k, not 8k.
 * Moreover, YY_BUF_SIZE is 2*YY_READ_BUF_SIZE in the general case.
 * Ditto for the __ia64__ case accordingly.
 */
#define YY_BUF_SIZE 32768
#else
#define YY_BUF_SIZE 16384
#endif /* __ia64__ */
#endif

/* The state buf must be large enough to hold one state per character in the main buffer.
 */
#define YY_STATE_BUF_SIZE   ((YY_BUF_SIZE + 2) * sizeof(yy_state_type))

#ifndef YY_TYPEDEF_YY_BUFFER_STATE
#define YY_TYPEDEF_YY_BUFFER_STATE
typedef struct yy_buffer_state *YY_BUFFER_STATE;
#endif

#ifndef YY_TYPEDEF_YY_SIZE_T
#define YY_TYPEDEF_YY_SIZE_T
typedef size_t yy_size_t;
#endif

#define EOB_ACT_CONTINUE_SCAN 0
#define EOB_ACT_END_OF_FILE 1
#define EOB_ACT_LAST_MATCH 2

    /* Note: We specifically omit the test for yy_rule_can_match_eol because it requires
     *       access to the local variable yy_act. Since yyless() is a macro, it would break
     *       existing scanners that call yyless() from OUTSIDE yylex.
     *       One obvious solution it to make yy_act a global. I tried that, and saw
     *       a 5% performance hit in a non-yylineno scanner, because yy_act is
     *       normally declared as a register variable-- so it is not worth it.
     */
    #define  YY_LESS_LINENO(n) \
            do { \
                int yyl;\
                for ( yyl = n; yyl < yyleng; ++yyl )\
                    if ( yytext[yyl] == '\n' )\
                        --yylineno;\
            }while(0)
    #define YY_LINENO_REWIND_TO(dst) \
            do {\
                const char *p;\
                for ( p = yy_cp-1; p >= (dst); --p)\
                    if ( *p == '\n' )\
                        --yylineno;\
            }while(0)

/* Return all but the first "n" matched characters back to the input stream. */
#define yyless(n) \
	do \
		{ \
		/* Undo effects of setting up yytext. */ \
        int yyless_macro_arg = (n); \
        YY_LESS_LINENO(yyless_macro_arg);\
		*yy_cp = yyg->yy_hold_char; \
		YY_RESTORE_YY_MORE_OFFSET \
		yyg->yy_c_buf_p = yy_cp = yy_bp + yyless_macro_arg - YY_MORE_ADJ; \
		YY_DO_BEFORE_ACTION; /* set up yytext again */ \
		} \
	while ( 0 )
#define unput(c) yyunput( c, yyg->yytext_ptr , yyscanner )

#ifndef YY_STRUCT_YY_BUFFER_STATE
#define YY_STRUCT_YY_BUFFER_STATE
struct yy_buffer_state
	{
	FILE *yy_input_file;

	char *yy_ch_buf;		/* input buffer */
	char *yy_buf_pos;		/* current position in input buffer */

	/* Size of input buffer in bytes, not including room for EOB
	 * characters.
	 */
	int yy_buf_size;

	/* Number of characters read into yy_ch_buf, not including EOB
	 * characters.
	 */
	int yy_n_chars;

	/* Whether we "own" the buffer - i.e., we know we created it,
	 * and can realloc() it to grow it, and should free() it to
	 * delete it.
	 */
	int yy_is_our_buffer;

	/* Whether this is an "interactive" input source; if so, and
	 * if we're using stdio for input, then we want to use getc()
	 * instead of fread(), to make sure we stop fetching input after
	 * each newline.
	 */
	int yy_is_interactive;

	/* Whether we're considered to be at the beginning of a line.
	 * If so, '^' rules will be active on the next match, otherwise
	 * not.
	 */
	int yy_at_bol;

    int yy_bs_lineno; /**< The line count. */
    int yy_bs_column; /**< The column count. */

	/* Whether to try to fill the input buffer when we reach the
	 * end of it.
	 */
	int yy_fill_buffer;

	int yy_buffer_status;

#define YY_BUFFER_NEW 0
#define YY_BUFFER_NORMAL 1
	/* When an EOF's been seen but there's still some text to process
	 * then we mark the buffer as YY_EOF_PENDING, to indicate that we
	 * shouldn't try reading from the input source any more.  We might
	 * still have a bunch of tokens to match, though, because of
	 * possible backing-up.
	 *
	 * When we actually see the EOF, we change the status to "new"
	 * (via yyrestart()), so that the user can continue scanning by
	 * just pointing yyin at a new input file.
	 */
#define YY_BUFFER_EOF_PENDING 2

	};
#endif /* !YY_STRUCT_YY_BUFFER_STATE */

/* We provide macros for accessing buffer states in case in the
 * future we want to put the buffer states in a more general
 * "scanner state".
 *
 * Returns the top of the stack, or NULL.
 */
#define YY_CURRENT_BUFFER ( yyg->yy_buffer_stack \
                          ? yyg->yy_buffer_stack[yyg->yy_buffer_stack_top] \
                          : NULL)
/* Same as previous macro, but useful when we know that the buffer stack is not
 * NULL or when we need an lvalue. For internal use only.
 */
#define YY_CURRENT_BUFFER_LVALUE yyg->yy_buffer_stack[yyg->yy_buffer_stack_top]

void yyrestart ( FILE *input_file , yyscan_t yyscanner );
void yy_switch_to_buffer ( YY_BUFFER_STATE new_buffer , yyscan_t yyscanner );
YY_BUFFER_STATE yy_create_buffer ( FILE *file, int size , yyscan_t yyscanner );
void yy_delete_buffer ( YY_BUFFER_STATE b , yyscan_t yyscanner );
void yy_flush_buffer ( YY_BUFFER_STATE b , yyscan_t yyscanner );
void yypush_buffer_state ( YY_BUFFER_STATE new_buffer , yyscan_t yyscanner );
void yypop_buffer_state ( yyscan_t yyscanner );

static void yyensure_buffer_stack ( yyscan_t yyscanner );
static void yy_load_buffer_state ( yyscan_t yyscanner );
static void yy_init_buffer ( YY_BUFFER_STATE b, FILE *file , yyscan_t yyscanner );
#define YY_FLUSH_BUFFER yy_flush_buffer( YY_CURRENT_BUFFER , yyscanner)

YY_BUFFER_STATE yy_scan_buffer ( char *base, yy_size_t size , yyscan_t yyscanner );
YY_BUFFER_STATE yy_scan_string ( const char *yy_str , yyscan_t yyscanner );
YY_BUFFER_STATE yy_scan_bytes ( const char *bytes, int len , yyscan_t yyscanner );

void *yyalloc ( yy_size_t , yyscan_t yyscanner );
void *yyrealloc ( void *, yy_size_t , yyscan_t yyscanner );
void yyfree ( void * , yyscan_t yyscanner );

#define yy_new_buffer yy_create_buffer
#define yy_set_interactive(is_interactive) \
	{ \
	if ( ! YY_CURRENT_BUFFER ){ \
        yyensure_buffer_stack (yyscanner); \
		YY_CURRENT_BUFFER_LVALUE =    \
            yy_create_buffer( yyin, YY_BUF_SIZE , yyscanner); \
	} \
	YY_CURRENT_BUFFER_LVALUE->yy_is_interactive = is_interactive; \
	}
#define yy_set_bol(at_bol) \
	{ \
	if ( ! YY_CURRENT_BUFFER ){\
        yyensure_buffer_stack (yyscanner); \
		YY_CURRENT_BUFFER_LVALUE =    \
            yy_create_buffer( yyin, YY_BUF_SIZE , yyscanner); \
	} \
	YY_CURRENT_BUFFER_LVALUE->yy_at_bol = at_bol; \
	}
#define YY_AT_BOL() (YY_CURRENT_BUFFER_LVALUE->yy_at_bol)

#define libconfig_yywrap(yyscanner) (/*CONSTCOND*/1)
#define YY_SKIP_YYWRAP
typedef flex_uint8_t YY_CHAR;

typedef int yy_state_type;

#define yytext_ptr yytext_r

static yy_state_type yy_get_previous_state ( yyscan_t yyscanner );
static yy_state_type yy_try_NUL_trans ( yy_state_type current_state  , yyscan_t yyscanner);
static int yy_get_next_buffer ( yyscan_t yyscanner );
static void yynoreturn yy_fatal_error ( const char* msg , yyscan_t yyscanner );

/* Done after the current pattern has been matched and before the
 * corresponding action - sets up yytext.
 */
#define YY_DO_BEFORE_ACTION \
	yyg->yytext_ptr = yy_bp; \
	yyleng = (int) (yy_cp - yy_bp); \
	yyg->yy_hold_char = *yy_cp; \
	*yy_cp = '\0'; \
	yyg->yy_c_buf_p = yy_cp;
#define YY_NUM_RULES 44
#define YY_END_OF_BUFFER 45
/* This struct is not used in this scanner,
   but its presence is necessary. */
struct yy_trans_info
	{
	flex_int32_t yy_verify;
	flex_int32_t yy_nxt;
	};
static const flex_int16_t yy_accept[105] =
    {   0,
        0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
       45,   43,   25,   24,   24,    8,    1,   40,   41,   32,
       43,   27,   33,   43,   34,   34,   26,   42,   32,   32,
       38,   39,   28,   29,   25,   43,    3,    2,    6,    7,
        6,    9,   18,   17,   20,   23,   44,   25,   32,   33,
       34,   33,    0,    4,    1,   33,    0,   35,    0,   32,
       32,   25,    0,    0,    5,    9,   15,    0,   14,   13,
       10,   11,   12,   20,   22,   21,    0,   33,   33,    0,
        0,   33,   35,   36,   32,   32,    0,    0,    0,   33,
       37,   32,   30,    0,   16,   37,   31,    0,    0,    0,

        0,    0,   19,    0
    } ;

static const YY_CHAR yy_ec[256] =
    {   0,
        1,    1,    1,    1,    1,    1,    1,    1,    2,    3,
        1,    4,    5,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    2,    1,    6,    7,    1,    1,    1,    1,    8,
        9,   10,   11,   12,   13,   14,   15,   16,   17,   17,
       17,   17,   17,   17,   17,   17,   17,   18,   19,    1,
       20,    1,    1,   21,   22,   23,   23,   23,   24,   25,
       26,   26,   26,   26,   26,   27,   26,   26,   26,   26,
       26,   28,   29,   30,   31,   26,   26,   32,   26,   26,
       33,   34,   35,    1,   36,    1,   22,   23,   37,   38,

       39,   40,   26,   26,   41,   26,   26,   42,   26,   43,
       26,   26,   26,   44,   29,   45,   46,   26,   26,   32,
       26,   26,   47,    1,   48,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,

        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1
    } ;

static const YY_CHAR yy_meta[49] =
    {   0,
        1,    1,    1,    1,    1,    2,    1,    1,    1,    3,
        1,    1,    3,    1,    1,    4,    4,    1,    1,    1,
        1,    4,    4,    4,    4,    3,    3,    3,    3,    3,
        3,    3,    1,    2,    1,    3,    4,    4,    4,    4,
        3,    3,    3,    3,    3,    3,    1,    1
    } ;

static const flex_int16_t yy_base[115] =
    {   0,
        0,   47,  199,  198,   47,   48,   46,   47,   48,   49,
      200,  203,  197,  203,  203,  203,  203,  203,  203,    0,
       45,  203,   47,   50,   60,   74,  203,  203,  176,   28,
      203,  203,  203,  203,   64,  156,  203,  203,  203,  203,
      181,    0,  203,   63,    0,  203,   83,  191,    0,   88,
      102,   98,   62,  203,  203,  104,  119,  162,    0,   67,
       65,  121,  127,  124,  203,    0,  203,    0,  203,  203,
      203,  203,  203,    0,  203,  203,  108,  117,  122,  134,
      132,  136,  203,  139,  136,  116,  126,    0,  140,  142,
      135,  130,    0,  102,  203,  203,    0,   85,   72,   63,

       98,  158,  203,  203,  169,  173,  177,  181,  183,  187,
      191,   89,   66,   63
    } ;

static const flex_int16_t yy_def[115] =
    {   0,
      104,    1,  105,  105,  106,  106,  107,  107,  108,  108,
      104,  104,  104,  104,  104,  104,  104,  104,  104,  109,
      104,  104,  104,  104,  104,  104,  104,  104,  109,  109,
      104,  104,  104,  104,  104,  104,  104,  104,  104,  104,
      104,  110,  104,  104,  111,  104,  104,  104,  109,  104,
      104,  104,  104,  104,  104,  104,  104,  104,  112,  109,
      109,  104,  104,  104,  104,  110,  104,  113,  104,  104,
      104,  104,  104,  111,  104,  104,  104,  104,  104,  104,
      104,  104,  104,  112,  109,  109,  104,  114,  104,  104,
      104,  109,  109,  104,  104,  104,  109,  104,  104,  104,

      104,  104,  104,    0,  104,  104,  104,  104,  104,  104,
      104,  104,  104,  104
    } ;

static const flex_int16_t yy_nxt[252] =
    {   0,
       12,   13,   14,   15,   15,   16,   17,   18,   19,   20,
       21,   22,   21,   23,   24,   25,   26,   27,   28,   27,
       12,   20,   20,   20,   29,   20,   20,   20,   20,   30,
       20,   20,   31,   12,   32,   12,   20,   20,   20,   29,
       20,   20,   20,   20,   30,   20,   33,   34,   35,   40,
       40,   43,   43,   46,   46,   61,   41,   41,   50,   54,
       51,   51,   52,   52,   55,   62,   95,   36,   67,   88,
       53,   61,   77,   56,   77,   51,   51,   78,   78,   44,
       44,   47,   47,   57,   63,   53,   58,   56,   75,   51,
       51,   59,   84,   85,   68,   86,   69,   57,   57,  102,

       58,  101,   70,   52,   52,   71,   72,   73,   85,  100,
       86,   53,   57,   52,   52,   56,   76,   51,   51,   79,
       79,   53,   62,   78,   78,   57,   53,   80,   58,   81,
       99,   81,   78,   78,   82,   82,   53,   79,   79,   93,
       57,   63,   80,   98,   89,   80,   89,   82,   82,   90,
       90,   82,   82,   97,   93,   90,   90,   90,   90,  102,
       80,   96,   94,  103,   92,   91,   87,   64,   97,   37,
       37,   37,   37,   39,   39,   39,   39,   42,   42,   42,
       42,   45,   45,   45,   45,   49,   49,   66,   83,   66,
       66,   74,   48,   74,   74,   65,   64,   60,   48,  104,

       38,   38,   11,  104,  104,  104,  104,  104,  104,  104,
      104,  104,  104,  104,  104,  104,  104,  104,  104,  104,
      104,  104,  104,  104,  104,  104,  104,  104,  104,  104,
      104,  104,  104,  104,  104,  104,  104,  104,  104,  104,
      104,  104,  104,  104,  104,  104,  104,  104,  104,  104,
      104
    } ;

static const flex_int16_t yy_chk[252] =
    {   0,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    2,    5,
        6,    7,    8,    9,   10,   30,    5,    6,   21,   24,
       21,   21,   23,   23,   24,   35,  114,    2,   44,  113,
       23,   30,   53,   25,   53,   25,   25,   53,   53,    7,
        8,    9,   10,   25,   35,   23,   25,   26,   47,   26,
       26,   25,  112,   60,   44,   61,   44,   26,   25,  101,

       26,  100,   44,   50,   50,   44,   44,   44,   60,   99,
       61,   50,   26,   52,   52,   51,   47,   51,   51,   56,
       56,   52,   62,   77,   77,   51,   50,   56,   51,   57,
       98,   57,   78,   78,   57,   57,   52,   79,   79,   86,
       51,   62,   56,   94,   80,   79,   80,   81,   81,   80,
       80,   82,   82,   92,   86,   89,   89,   90,   90,  102,
       79,   91,   87,  102,   85,   84,   64,   63,   92,  105,
      105,  105,  105,  106,  106,  106,  106,  107,  107,  107,
      107,  108,  108,  108,  108,  109,  109,  110,   58,  110,
      110,  111,   48,  111,  111,   41,   36,   29,   13,   11,

        4,    3,  104,  104,  104,  104,  104,  104,  104,  104,
      104,  104,  104,  104,  104,  104,  104,  104,  104,  104,
      104,  104,  104,  104,  104,  104,  104,  104,  104,  104,
      104,  104,  104,  104,  104,  104,  104,  104,  104,  104,
      104,  104,  104,  104,  104,  104,  104,  104,  104,  104,
      104
    } ;

/* Table of booleans, true if rule could match eol. */
static const flex_int32_t yy_rule_can_match_eol[45] =
    {   0,
0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,     };

/* The intent behind this definition is that it'll catch
 * any uses of REJECT which flex missed.
 */
#define REJECT reject_used_but_not_detected
#define yymore() yymore_used_but_not_detected
#define YY_MORE_ADJ 0
#define YY_RESTORE_YY_MORE_OFFSET
#line 1 "scanner.l"
/* -*- mode: C -*- */
/* --------------------------------------------------------------------------
   libconfig - A library for processing structured configuration files
   Copyright (C) 2005-2020  Mark A Lindner

   This file is part of libconfig.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License
   as published by the Free Software Foundation; either version 2.1 of
   the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Library General Public
   License along with this library; if not, see
   <http://www.gnu.org/licenses/>.
   ----------------------------------------------------------------------------
*/
#define YY_NO_UNISTD_H 1
#line 36 "scanner.l"

#ifdef _MSC_VER
#pragma warning (disable: 4996)
#endif

#include <stdlib.h>
#include <errno.h>
#include <ctype.h>
#include <string.h>
#include <limits.h>


#define YY_NO_INPUT // Suppress generation of useless input() function

#line 803 "scanner.c"

#line 805 "scanner.c"

#define INITIAL 0
#define SINGLE_LINE_COMMENT 1
#define MULTI_LINE_COMMENT 2
#define STRING 3
#define INCLUDE 4

#ifndef YY_NO_UNISTD_H
/* Special case for "unistd.h", since it is non-ANSI. We include it way
 * down here because we want the user's section 1 to have been scanned first.
 * The user has a chance to override it with an option.
 */
#include <unistd.h>
#endif

#define YY_EXTRA_TYPE struct scan_context *

/* Holds the entire state of the reentrant scanner. */
struct yyguts_t
    {

    /* User-defined. Not touched by flex. */
    YY_EXTRA_TYPE yyextra_r;

    /* The rest are the same as the globals declared in the non-reentrant scanner. */
    FILE *yyin_r, *yyout_r;
    size_t yy_buffer_stack_top; /**< index of top of stack. */
    size_t yy_buffer_stack_max; /**< capacity of stack. */
    YY_BUFFER_STATE * yy_buffer_stack; /**< Stack as an array. */
    char yy_hold_char;
    int yy_n_chars;
    int yyleng_r;
    char *yy_c_buf_p;
    int yy_init;
    int yy_start;
    int yy_did_buffer_switch_on_eof;
    int yy_start_stack_ptr;
    int yy_start_stack_depth;
    int *yy_start_stack;
    yy_state_type yy_last_accepting_state;
    char* yy_last_accepting_cpos;

    int yylineno_r;
    int yy_flex_debug_r;

    char *yytext_r;
    int yy_more_flag;
    int yy_more_len;

    YYSTYPE * yylval_r;

    }; /* end struct yyguts_t */

static int yy_init_globals ( yyscan_t yyscanner );

    /* This must go here because YYSTYPE and YYLTYPE are included
     * from bison output in section 1.*/
    #    define yylval yyg->yylval_r

int yylex_init (yyscan_t* scanner);

int yylex_init_extra ( YY_EXTRA_TYPE user_defined, yyscan_t* scanner);

/* Accessor methods to globals.
   These are made visible to non-reentrant scanners for convenience. */

int yylex_destroy ( yyscan_t yyscanner );

int yyget_debug ( yyscan_t yyscanner );

void yyset_debug ( int debug_flag , yyscan_t yyscanner );

YY_EXTRA_TYPE yyget_extra ( yyscan_t yyscanner );

void yyset_extra ( YY_EXTRA_TYPE user_defined , yyscan_t yyscanner );

FILE *yyget_in ( yyscan_t yyscanner );

void yyset_in  ( FILE * _in_str , yyscan_t yyscanner );

FILE *yyget_out ( yyscan_t yyscanner );

void yyset_out  ( FILE * _out_str , yyscan_t yyscanner );

			int yyget_leng ( yyscan_t yyscanner );

char *yyget_text ( yyscan_t yyscanner );

int yyget_lineno ( yyscan_t yyscanner );

void yyset_lineno ( int _line_number , yyscan_t yyscanner );

int yyget_column  ( yyscan_t yyscanner );

void yyset_column ( int _column_no , yyscan_t yyscanner );

YYSTYPE * yyget_lval ( yyscan_t yyscanner );

void yyset_lval ( YYSTYPE * yylval_param , yyscan_t yyscanner );

/* Macros after this point can all be overridden by user definitions in
 * section 1.
 */

#ifndef YY_SKIP_YYWRAP
#ifdef __cplusplus
extern "C" int yywrap ( yyscan_t yyscanner );
#else
extern int yywrap ( yyscan_t yyscanner );
#endif
#endif

#ifndef YY_NO_UNPUT

#endif

#ifndef yytext_ptr
static void yy_flex_strncpy ( char *, const char *, int , yyscan_t yyscanner);
#endif

#ifdef YY_NEED_STRLEN
static int yy_flex_strlen ( const char * , yyscan_t yyscanner);
#endif

#ifndef YY_NO_INPUT
#ifdef __cplusplus
static int yyinput ( yyscan_t yyscanner );
#else
static int input ( yyscan_t yyscanner );
#endif

#endif

/* Amount of stuff to slurp up with each read. */
#ifndef YY_READ_BUF_SIZE
#ifdef __ia64__
/* On IA-64, the buffer size is 16k, not 8k */
#define YY_READ_BUF_SIZE 16384
#else
#define YY_READ_BUF_SIZE 8192
#endif /* __ia64__ */
#endif

/* Copy whatever the last rule matched to the standard output. */
#ifndef ECHO
/* This used to be an fputs(), but since the string might contain NUL's,
 * we now use fwrite().
 */
#define ECHO do { if (fwrite( yytext, (size_t) yyleng, 1, yyout )) {} } while (0)
#endif

/* Gets input and stuffs it into "buf".  number of characters read, or YY_NULL,
 * is returned in "result".
 */
#ifndef YY_INPUT
#define YY_INPUT(buf,result,max_size) \
	if ( YY_CURRENT_BUFFER_LVALUE->yy_is_interactive ) \
		{ \
		int c = '*'; \
		int n; \
		for ( n = 0; n < max_size && \
			     (c = getc( yyin )) != EOF && c != '\n'; ++n ) \
			buf[n] = (char) c; \
		if ( c == '\n' ) \
			buf[n++] = (char) c; \
		if ( c == EOF && ferror( yyin ) ) \
			YY_FATAL_ERROR( "input in flex scanner failed" ); \
		result = n; \
		} \
	else \
		{ \
		errno=0; \
		while ( (result = (int) fread(buf, 1, (yy_size_t) max_size, yyin)) == 0 && ferror(yyin)) \
			{ \
			if( errno != EINTR) \
				{ \
				YY_FATAL_ERROR( "input in flex scanner failed" ); \
				break; \
				} \
			errno=0; \
			clearerr(yyin); \
			} \
		}\
\

#endif

/* No semi-colon after return; correct usage is to write "yyterminate();" -
 * we don't want an extra ';' after the "return" because that will cause
 * some compilers to complain about unreachable statements.
 */
#ifndef yyterminate
#define yyterminate() return YY_NULL
#endif

/* Number of entries by which start-condition stack grows. */
#ifndef YY_START_STACK_INCR
#define YY_START_STACK_INCR 25
#endif

/* Report a fatal error. */
#ifndef YY_FATAL_ERROR
#define YY_FATAL_ERROR(msg) yy_fatal_error( msg , yyscanner)
#endif

/* end tables serialization structures and prototypes */

/* Default declaration of generated scanner - a define so the user can
 * easily add parameters.
 */
#ifndef YY_DECL
#define YY_DECL_IS_OURS 1

extern int yylex \
               (YYSTYPE * yylval_param , yyscan_t yyscanner);

#define YY_DECL int yylex \
               (YYSTYPE * yylval_param , yyscan_t yyscanner)
#endif /* !YY_DECL */

/* Code executed at the beginning of each rule, after yytext and yyleng
 * have been set up.
 */
#ifndef YY_USER_ACTION
#define YY_USER_ACTION
#endif

/* Code executed at the end of each rule. */
#ifndef YY_BREAK
#define YY_BREAK /*LINTED*/break;
#endif

#define YY_RULE_SETUP \
	if ( yyleng > 0 ) \
		YY_CURRENT_BUFFER_LVALUE->yy_at_bol = \
				(yytext[yyleng - 1] == '\n'); \
	YY_USER_ACTION

/** The main scanner function which does all the work.
 */
YY_DECL
{
	yy_state_type yy_current_state;
	char *yy_cp, *yy_bp;
	int yy_act;
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;

    yylval = yylval_param;

	if ( !yyg->yy_init )
		{
		yyg->yy_init = 1;

#ifdef YY_USER_INIT
		YY_USER_INIT;
#endif

		if ( ! yyg->yy_start )
			yyg->yy_start = 1;	/* first start state */

		if ( ! yyin )
			yyin = stdin;

		if ( ! yyout )
			yyout = stdout;

		if ( ! YY_CURRENT_BUFFER ) {
			yyensure_buffer_stack (yyscanner);
			YY_CURRENT_BUFFER_LVALUE =
				yy_create_buffer( yyin, YY_BUF_SIZE , yyscanner);
		}

		yy_load_buffer_state( yyscanner );
		}

	{
#line 70 "scanner.l"


#line 1085 "scanner.c"

	while ( /*CONSTCOND*/1 )		/* loops until end-of-file is reached */
		{
		yy_cp = yyg->yy_c_buf_p;

		/* Support of yytext. */
		*yy_cp = yyg->yy_hold_char;

		/* yy_bp points to the position in yy_ch_buf of the start of
		 * the current run.
		 */
		yy_bp = yy_cp;

		yy_current_state = yyg->yy_start;
		yy_current_state += YY_AT_BOL();
yy_match:
		do
			{
			YY_CHAR yy_c = yy_ec[YY_SC_TO_UI(*yy_cp)] ;
			if ( yy_accept[yy_current_state] )
				{
				yyg->yy_last_accepting_state = yy_current_state;
				yyg->yy_last_accepting_cpos = yy_cp;
				}
			while ( yy_chk[yy_base[yy_current_state] + yy_c] != yy_current_state )
				{
				yy_current_state = (int) yy_def[yy_current_state];
				if ( yy_current_state >= 105 )
					yy_c = yy_meta[yy_c];
				}
			yy_current_state = yy_nxt[yy_base[yy_current_state] + yy_c];
			++yy_cp;
			}
		while ( yy_current_state != 104 );
		yy_cp = yyg->yy_last_accepting_cpos;
		yy_current_state = yyg->yy_last_accepting_state;

yy_find_action:
		yy_act = yy_accept[yy_current_state];

		YY_DO_BEFORE_ACTION;

		if ( yy_act != YY_END_OF_BUFFER && yy_rule_can_match_eol[yy_act] )
			{
			int yyl;
			for ( yyl = 0; yyl < yyleng; ++yyl )
				if ( yytext[yyl] == '\n' )

    do{ yylineno++;
        yycolumn=0;
    }while(0)
;
			}

do_action:	/* This label is used only to access EOF actions. */

		switch ( yy_act )
	{ /* beginning of action switch */
			case 0: /* must back up */
			/* undo the effects of YY_DO_BEFORE_ACTION */
			*yy_cp = yyg->yy_hold_char;
			yy_cp = yyg->yy_last_accepting_cpos;
			yy_current_state = yyg->yy_last_accepting_state;
			goto yy_find_action;

case 1:
YY_RULE_SETUP
#line 72 "scanner.l"
{ BEGIN SINGLE_LINE_COMMENT; }
	YY_BREAK
case 2:
/* rule 2 can match eol */
YY_RULE_SETUP
#line 73 "scanner.l"
{ BEGIN INITIAL; }
	YY_BREAK
case 3:
YY_RULE_SETUP
#line 74 "scanner.l"
{ /* ignore */ }
	YY_BREAK
case 4:
YY_RULE_SETUP
#line 76 "scanner.l"
{ BEGIN MULTI_LINE_COMMENT; }
	YY_BREAK
case 5:
YY_RULE_SETUP
#line 77 "scanner.l"
{ BEGIN INITIAL; }
	YY_BREAK
case 6:
YY_RULE_SETUP
#line 78 "scanner.l"
{ /* ignore */ }
	YY_BREAK
case 7:
/* rule 7 can match eol */
YY_RULE_SETUP
#line 79 "scanner.l"
{ /* ignore */ }
	YY_BREAK
case 8:
YY_RULE_SETUP
#line 81 "scanner.l"
{ BEGIN STRING; }
	YY_BREAK
case 9:
/* rule 9 can match eol */
YY_RULE_SETUP
#line 82 "scanner.l"
{ libconfig_scanctx_append_string(yyextra, yytext); }
	YY_BREAK
case 10:
YY_RULE_SETUP
#line 83 "scanner.l"
{ libconfig_scanctx_append_char(yyextra, '\n'); }
	YY_BREAK
case 11:
YY_RULE_SETUP
#line 84 "scanner.l"
{ libconfig_scanctx_append_char(yyextra, '\r'); }
	YY_BREAK
case 12:
YY_RULE_SETUP
#line 85 "scanner.l"
{ libconfig_scanctx_append_char(yyextra, '\t'); }
	YY_BREAK
case 13:
YY_RULE_SETUP
#line 86 "scanner.l"
{ libconfig_scanctx_append_char(yyextra, '\f'); }
	YY_BREAK
case 14:
YY_RULE_SETUP
#line 87 "scanner.l"
{ libconfig_scanctx_append_char(yyextra, '\\'); }
	YY_BREAK
case 15:
YY_RULE_SETUP
#line 88 "scanner.l"
{ libconfig_scanctx_append_char(yyextra, '\"'); }
	YY_BREAK
case 16:
YY_RULE_SETUP
#line 89 "scanner.l"
{
                    char c = (char)(strtol(yytext + 2, NULL, 16) & 0xFF);
                    libconfig_scanctx_append_char(yyextra, c);
                  }
	YY_BREAK
case 17:
YY_RULE_SETUP
#line 93 "scanner.l"
{ libconfig_scanctx_append_char(yyextra, '\\'); }
	YY_BREAK
case 18:
YY_RULE_SETUP
#line 94 "scanner.l"
{
                    yylval->sval = libconfig_scanctx_take_string(yyextra);
                    BEGIN INITIAL;
                    return(TOK_STRING);
                  }
	YY_BREAK
case 19:
YY_RULE_SETUP
#line 100 "scanner.l"
{ BEGIN INCLUDE; }
	YY_BREAK
case 20:
/* rule 20 can match eol */
YY_RULE_SETUP
#line 101 "scanner.l"
{ libconfig_scanctx_append_string(yyextra, yytext); }
	YY_BREAK
case 21:
YY_RULE_SETUP
#line 102 "scanner.l"
{ libconfig_scanctx_append_char(yyextra, '\\'); }
	YY_BREAK
case 22:
YY_RULE_SETUP
#line 103 "scanner.l"
{ libconfig_scanctx_append_char(yyextra, '\"'); }
	YY_BREAK
case 23:
YY_RULE_SETUP
#line 104 "scanner.l"
{
  const char *error = NULL;
  const char *path = libconfig_scanctx_take_string(yyextra);
  FILE *fp = libconfig_scanctx_push_include(yyextra, (void *)YY_CURRENT_BUFFER,
                                            path, &error);
  __delete(path);

  if(fp)
  {
    yyin = fp;
    yy_switch_to_buffer(yy_create_buffer(yyin, YY_BUF_SIZE, yyscanner),
                        yyscanner);
  }
  else if(error)
  {
    yyextra->config->error_text = error;
    yyextra->config->error_file = libconfig_scanctx_current_filename(yyextra);
    yyextra->config->error_line = libconfig_yyget_lineno(yyscanner);
    return TOK_ERROR;
  }
  BEGIN INITIAL;
}
	YY_BREAK
case 24:
/* rule 24 can match eol */
YY_RULE_SETUP
#line 127 "scanner.l"
{ /* ignore */ }
	YY_BREAK
case 25:
YY_RULE_SETUP
#line 128 "scanner.l"
{ /* ignore */ }
	YY_BREAK
case 26:
YY_RULE_SETUP
#line 130 "scanner.l"
{ return(TOK_EQUALS); }
	YY_BREAK
case 27:
YY_RULE_SETUP
#line 131 "scanner.l"
{ return(TOK_COMMA); }
	YY_BREAK
case 28:
YY_RULE_SETUP
#line 132 "scanner.l"
{ return(TOK_GROUP_START); }
	YY_BREAK
case 29:
YY_RULE_SETUP
#line 133 "scanner.l"
{ return(TOK_GROUP_END); }
	YY_BREAK
case 30:
YY_RULE_SETUP
#line 134 "scanner.l"
{ yylval->ival = 1; return(TOK_BOOLEAN); }
	YY_BREAK
case 31:
YY_RULE_SETUP
#line 135 "scanner.l"
{ yylval->ival = 0; return(TOK_BOOLEAN); }
	YY_BREAK
case 32:
YY_RULE_SETUP
#line 136 "scanner.l"
{ yylval->sval = yytext; return(TOK_NAME); }
	YY_BREAK
case 33:
YY_RULE_SETUP
#line 137 "scanner.l"
{ yylval->fval = atof(yytext); return(TOK_FLOAT); }
	YY_BREAK
case 34:
YY_RULE_SETUP
#line 138 "scanner.l"
{
                    int ok;
                    long long llval = libconfig_parse_integer(yytext, &ok);
                    if(!ok)
                      return(TOK_ERROR);

                    if((llval < INT_MIN) || (llval > INT_MAX))
                    {
                      yylval->llval = llval;
                      return(TOK_INTEGER64);
                    }
                    else
                    {
                      yylval->ival = (int)llval;
                      return(TOK_INTEGER);
                    }
                  }
	YY_BREAK
case 35:
YY_RULE_SETUP
#line 155 "scanner.l"
{ yylval->llval = atoll(yytext); return(TOK_INTEGER64); }
	YY_BREAK
case 36:
YY_RULE_SETUP
#line 156 "scanner.l"
{
                    yylval->ival = strtoul(yytext, NULL, 16);
                    return(TOK_HEX);
                  }
	YY_BREAK
case 37:
YY_RULE_SETUP
#line 160 "scanner.l"
{
                    yylval->llval = libconfig_parse_hex64(yytext);
                    return(TOK_HEX64);
                  }
	YY_BREAK
case 38:
YY_RULE_SETUP
#line 164 "scanner.l"
{ return(TOK_ARRAY_START); }
	YY_BREAK
case 39:
YY_RULE_SETUP
#line 165 "scanner.l"
{ return(TOK_ARRAY_END); }
	YY_BREAK
case 40:
YY_RULE_SETUP
#line 166 "scanner.l"
{ return(TOK_LIST_START); }
	YY_BREAK
case 41:
YY_RULE_SETUP
#line 167 "scanner.l"
{ return(TOK_LIST_END); }
	YY_BREAK
case 42:
YY_RULE_SETUP
#line 168 "scanner.l"
{ return(TOK_SEMICOLON); }
	YY_BREAK
case 43:
YY_RULE_SETUP
#line 169 "scanner.l"
{ return(TOK_GARBAGE); }
	YY_BREAK
case YY_STATE_EOF(INITIAL):
case YY_STATE_EOF(SINGLE_LINE_COMMENT):
case YY_STATE_EOF(MULTI_LINE_COMMENT):
case YY_STATE_EOF(STRING):
case YY_STATE_EOF(INCLUDE):
#line 171 "scanner.l"
{
  const char *error = NULL;
  FILE *fp;

  fp = libconfig_scanctx_next_include_file(yyextra, &error);
  if(fp)
  {
    yyin = fp;
    yy_delete_buffer(YY_CURRENT_BUFFER, yyscanner);
    yy_switch_to_buffer(yy_create_buffer(yyin, YY_BUF_SIZE, yyscanner),
                        yyscanner);
  }
  else if(error)
  {
    yyextra->config->error_text = error;
    yyextra->config->error_file = libconfig_scanctx_current_filename(yyextra);
    yyextra->config->error_line = libconfig_yyget_lineno(yyscanner);
    return TOK_ERROR;
  }
  else
  {
    /* No more files in the current include list. */
    YY_BUFFER_STATE buf = (YY_BUFFER_STATE)libconfig_scanctx_pop_include(yyextra);
    if(buf)
    {
      yy_delete_buffer(YY_CURRENT_BUFFER, yyscanner);
      yy_switch_to_buffer(buf, yyscanner);
    }
    else
      yyterminate();
  }
}
	YY_BREAK
case 44:
YY_RULE_SETUP
#line 203 "scanner.l"
ECHO;
	YY_BREAK
#line 1465 "scanner.c"

	case YY_END_OF_BUFFER:
		{
		/* Amount of text matched not including the EOB char. */
		int yy_amount_of_matched_text = (int) (yy_cp - yyg->yytext_ptr) - 1;

		/* Undo the effects of YY_DO_BEFORE_ACTION. */
		*yy_cp = yyg->yy_hold_char;
		YY_RESTORE_YY_MORE_OFFSET

		if ( YY_CURRENT_BUFFER_LVALUE->yy_buffer_status == YY_BUFFER_NEW )
			{
			/* We're scanning a new file or input source.  It's
			 * possible that this happened because the user
			 * just pointed yyin at a new source and called
			 * yylex().  If so, then we have to assure
			 * consistency between YY_CURRENT_BUFFER and our
			 * globals.  Here is the right place to do so, because
			 * this is the first action (other than possibly a
			 * back-up) that will match for the new input source.
			 */
			yyg->yy_n_chars = YY_CURRENT_BUFFER_LVALUE->yy_n_chars;
			YY_CURRENT_BUFFER_LVALUE->yy_input_file = yyin;
			YY_CURRENT_BUFFER_LVALUE->yy_buffer_status = YY_BUFFER_NORMAL;
			}

		/* Note that here we test for yy_c_buf_p "<=" to the position
		 * of the first EOB in the buffer, since yy_c_buf_p will
		 * already have been incremented past the NUL character
		 * (since all states make transitions on EOB to the
		 * end-of-buffer state).  Contrast this with the test
		 * in input().
		 */
		if ( yyg->yy_c_buf_p <= &YY_CURRENT_BUFFER_LVALUE->yy_ch_buf[yyg->yy_n_chars] )
			{ /* This was really a NUL. */
			yy_state_type yy_next_state;

			yyg->yy_c_buf_p = yyg->yytext_ptr + yy_amount_of_matched_text;

			yy_current_state = yy_get_previous_state( yyscanner );

			/* Okay, we're now positioned to make the NUL
			 * transition.  We couldn't have
			 * yy_get_previous_state() go ahead and do it
			 * for us because it doesn't know how to deal
			 * with the possibility of jamming (and we don't
			 * want to build jamming into it because then it
			 * will run more slowly).
			 */

			yy_next_state = yy_try_NUL_trans( yy_current_state , yyscanner);

			yy_bp = yyg->yytext_ptr + YY_MORE_ADJ;

			if ( yy_next_state )
				{
				/* Consume the NUL. */
				yy_cp = ++yyg->yy_c_buf_p;
				yy_current_state = yy_next_state;
				goto yy_match;
				}

			else
				{
				yy_cp = yyg->yy_last_accepting_cpos;
				yy_current_state = yyg->yy_last_accepting_state;
				goto yy_find_action;
				}
			}

		else switch ( yy_get_next_buffer( yyscanner ) )
			{
			case EOB_ACT_END_OF_FILE:
				{
				yyg->yy_did_buffer_switch_on_eof = 0;

				if ( yywrap( yyscanner ) )
					{
					/* Note: because we've taken care in
					 * yy_get_next_buffer() to have set up
					 * yytext, we can now set up
					 * yy_c_buf_p so that if some total
					 * hoser (like flex itself) wants to
					 * call the scanner after we return the
					 * YY_NULL, it'll still work - another
					 * YY_NULL will get returned.
					 */
					yyg->yy_c_buf_p = yyg->yytext_ptr + YY_MORE_ADJ;

					yy_act = YY_STATE_EOF(YY_START);
					goto do_action;
					}

				else
					{
					if ( ! yyg->yy_did_buffer_switch_on_eof )
						YY_NEW_FILE;
					}
				break;
				}

			case EOB_ACT_CONTINUE_SCAN:
				yyg->yy_c_buf_p =
					yyg->yytext_ptr + yy_amount_of_matched_text;

				yy_current_state = yy_get_previous_state( yyscanner );

				yy_cp = yyg->yy_c_buf_p;
				yy_bp = yyg->yytext_ptr + YY_MORE_ADJ;
				goto yy_match;

			case EOB_ACT_LAST_MATCH:
				yyg->yy_c_buf_p =
				&YY_CURRENT_BUFFER_LVALUE->yy_ch_buf[yyg->yy_n_chars];

				yy_current_state = yy_get_previous_state( yyscanner );

				yy_cp = yyg->yy_c_buf_p;
				yy_bp = yyg->yytext_ptr + YY_MORE_ADJ;
				goto yy_find_action;
			}
		break;
		}

	default:
		YY_FATAL_ERROR(
			"fatal flex scanner internal error--no action found" );
	} /* end of action switch */
		} /* end of scanning one token */
	} /* end of user's declarations */
} /* end of yylex */

/* yy_get_next_buffer - try to read in a new buffer
 *
 * Returns a code representing an action:
 *	EOB_ACT_LAST_MATCH -
 *	EOB_ACT_CONTINUE_SCAN - continue scanning from current position
 *	EOB_ACT_END_OF_FILE - end of file
 */
static int yy_get_next_buffer (yyscan_t yyscanner)
{
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;
	char *dest = YY_CURRENT_BUFFER_LVALUE->yy_ch_buf;
	char *source = yyg->yytext_ptr;
	int number_to_move, i;
	int ret_val;

	if ( yyg->yy_c_buf_p > &YY_CURRENT_BUFFER_LVALUE->yy_ch_buf[yyg->yy_n_chars + 1] )
		YY_FATAL_ERROR(
		"fatal flex scanner internal error--end of buffer missed" );

	if ( YY_CURRENT_BUFFER_LVALUE->yy_fill_buffer == 0 )
		{ /* Don't try to fill the buffer, so this is an EOF. */
		if ( yyg->yy_c_buf_p - yyg->yytext_ptr - YY_MORE_ADJ == 1 )
			{
			/* We matched a single character, the EOB, so
			 * treat this as a final EOF.
			 */
			return EOB_ACT_END_OF_FILE;
			}

		else
			{
			/* We matched some text prior to the EOB, first
			 * process it.
			 */
			return EOB_ACT_LAST_MATCH;
			}
		}

	/* Try to read more data. */

	/* First move last chars to start of buffer. */
	number_to_move = (int) (yyg->yy_c_buf_p - yyg->yytext_ptr - 1);

	for ( i = 0; i < number_to_move; ++i )
		*(dest++) = *(source++);

	if ( YY_CURRENT_BUFFER_LVALUE->yy_buffer_status == YY_BUFFER_EOF_PENDING )
		/* don't do the read, it's not guaranteed to return an EOF,
		 * just force an EOF
		 */
		YY_CURRENT_BUFFER_LVALUE->yy_n_chars = yyg->yy_n_chars = 0;

	else
		{
			int num_to_read =
			YY_CURRENT_BUFFER_LVALUE->yy_buf_size - number_to_move - 1;

		while ( num_to_read <= 0 )
			{ /* Not enough room in the buffer - grow it. */

			/* just a shorter name for the current buffer */
			YY_BUFFER_STATE b = YY_CURRENT_BUFFER_LVALUE;

			int yy_c_buf_p_offset =
				(int) (yyg->yy_c_buf_p - b->yy_ch_buf);

			if ( b->yy_is_our_buffer )
				{
				int new_size = b->yy_buf_size * 2;

				if ( new_size <= 0 )
					b->yy_buf_size += b->yy_buf_size / 8;
				else
					b->yy_buf_size *= 2;

				b->yy_ch_buf = (char *)
					/* Include room in for 2 EOB chars. */
					yyrealloc( (void *) b->yy_ch_buf,
							 (yy_size_t) (b->yy_buf_size + 2) , yyscanner );
				}
			else
				/* Can't grow it, we don't own it. */
				b->yy_ch_buf = NULL;

			if ( ! b->yy_ch_buf )
				YY_FATAL_ERROR(
				"fatal error - scanner input buffer overflow" );

			yyg->yy_c_buf_p = &b->yy_ch_buf[yy_c_buf_p_offset];

			num_to_read = YY_CURRENT_BUFFER_LVALUE->yy_buf_size -
						number_to_move - 1;

			}

		if ( num_to_read > YY_READ_BUF_SIZE )
			num_to_read = YY_READ_BUF_SIZE;

		/* Read in more data. */
		YY_INPUT( (&YY_CURRENT_BUFFER_LVALUE->yy_ch_buf[number_to_move]),
			yyg->yy_n_chars, num_to_read );

		YY_CURRENT_BUFFER_LVALUE->yy_n_chars = yyg->yy_n_chars;
		}

	if ( yyg->yy_n_chars == 0 )
		{
		if ( number_to_move == YY_MORE_ADJ )
			{
			ret_val = EOB_ACT_END_OF_FILE;
			yyrestart( yyin  , yyscanner);
			}

		else
			{
			ret_val = EOB_ACT_LAST_MATCH;
			YY_CURRENT_BUFFER_LVALUE->yy_buffer_status =
				YY_BUFFER_EOF_PENDING;
			}
		}

	else
		ret_val = EOB_ACT_CONTINUE_SCAN;

	if ((yyg->yy_n_chars + number_to_move) > YY_CURRENT_BUFFER_LVALUE->yy_buf_size) {
		/* Extend the array by 50%, plus the number we really need. */
		int new_size = yyg->yy_n_chars + number_to_move + (yyg->yy_n_chars >> 1);
		YY_CURRENT_BUFFER_LVALUE->yy_ch_buf = (char *) yyrealloc(
			(void *) YY_CURRENT_BUFFER_LVALUE->yy_ch_buf, (yy_size_t) new_size , yyscanner );
		if ( ! YY_CURRENT_BUFFER_LVALUE->yy_ch_buf )
			YY_FATAL_ERROR( "out of dynamic memory in yy_get_next_buffer()" );
		/* "- 2" to take care of EOB's */
		YY_CURRENT_BUFFER_LVALUE->yy_buf_size = (int) (new_size - 2);
	}

	yyg->yy_n_chars += number_to_move;
	YY_CURRENT_BUFFER_LVALUE->yy_ch_buf[yyg->yy_n_chars] = YY_END_OF_BUFFER_CHAR;
	YY_CURRENT_BUFFER_LVALUE->yy_ch_buf[yyg->yy_n_chars + 1] = YY_END_OF_BUFFER_CHAR;

	yyg->yytext_ptr = &YY_CURRENT_BUFFER_LVALUE->yy_ch_buf[0];

	return ret_val;
}

/* yy_get_previous_state - get the state just before the EOB char was reached */

    static yy_state_type yy_get_previous_state (yyscan_t yyscanner)
{
	yy_state_type yy_current_state;
	char *yy_cp;
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;

	yy_current_state = yyg->yy_start;
	yy_current_state += YY_AT_BOL();

	for ( yy_cp = yyg->yytext_ptr + YY_MORE_ADJ; yy_cp < yyg->yy_c_buf_p; ++yy_cp )
		{
		YY_CHAR yy_c = (*yy_cp ? yy_ec[YY_SC_TO_UI(*yy_cp)] : 1);
		if ( yy_accept[yy_current_state] )
			{
			yyg->yy_last_accepting_state = yy_current_state;
			yyg->yy_last_accepting_cpos = yy_cp;
			}
		while ( yy_chk[yy_base[yy_current_state] + yy_c] != yy_current_state )
			{
			yy_current_state = (int) yy_def[yy_current_state];
			if ( yy_current_state >= 105 )
				yy_c = yy_meta[yy_c];
			}
		yy_current_state = yy_nxt[yy_base[yy_current_state] + yy_c];
		}

	return yy_current_state;
}

/* yy_try_NUL_trans - try to make a transition on the NUL character
 *
 * synopsis
 *	next_state = yy_try_NUL_trans( current_state );
 */
    static yy_state_type yy_try_NUL_trans  (yy_state_type yy_current_state , yyscan_t yyscanner)
{
	int yy_is_jam;
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner; /* This var may be unused depending upon options. */
	char *yy_cp = yyg->yy_c_buf_p;

	YY_CHAR yy_c = 1;
	if ( yy_accept[yy_current_state] )
		{
		yyg->yy_last_accepting_state = yy_current_state;
		yyg->yy_last_accepting_cpos = yy_cp;
		}
	while ( yy_chk[yy_base[yy_current_state] + yy_c] != yy_current_state )
		{
		yy_current_state = (int) yy_def[yy_current_state];
		if ( yy_current_state >= 105 )
			yy_c = yy_meta[yy_c];
		}
	yy_current_state = yy_nxt[yy_base[yy_current_state] + yy_c];
	yy_is_jam = (yy_current_state == 104);

	(void)yyg;
	return yy_is_jam ? 0 : yy_current_state;
}

#ifndef YY_NO_UNPUT

#endif

#ifndef YY_NO_INPUT
#ifdef __cplusplus
    static int yyinput (yyscan_t yyscanner)
#else
    static int input  (yyscan_t yyscanner)
#endif

{
	int c;
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;

	*yyg->yy_c_buf_p = yyg->yy_hold_char;

	if ( *yyg->yy_c_buf_p == YY_END_OF_BUFFER_CHAR )
		{
		/* yy_c_buf_p now points to the character we want to return.
		 * If this occurs *before* the EOB characters, then it's a
		 * valid NUL; if not, then we've hit the end of the buffer.
		 */
		if ( yyg->yy_c_buf_p < &YY_CURRENT_BUFFER_LVALUE->yy_ch_buf[yyg->yy_n_chars] )
			/* This was really a NUL. */
			*yyg->yy_c_buf_p = '\0';

		else
			{ /* need more input */
			int offset = (int) (yyg->yy_c_buf_p - yyg->yytext_ptr);
			++yyg->yy_c_buf_p;

			switch ( yy_get_next_buffer( yyscanner ) )
				{
				case EOB_ACT_LAST_MATCH:
					/* This happens because yy_g_n_b()
					 * sees that we've accumulated a
					 * token and flags that we need to
					 * try matching the token before
					 * proceeding.  But for input(),
					 * there's no matching to consider.
					 * So convert the EOB_ACT_LAST_MATCH
					 * to EOB_ACT_END_OF_FILE.
					 */

					/* Reset buffer status. */
					yyrestart( yyin , yyscanner);

					/*FALLTHROUGH*/

				case EOB_ACT_END_OF_FILE:
					{
					if ( yywrap( yyscanner ) )
						return 0;

					if ( ! yyg->yy_did_buffer_switch_on_eof )
						YY_NEW_FILE;
#ifdef __cplusplus
					return yyinput(yyscanner);
#else
					return input(yyscanner);
#endif
					}

				case EOB_ACT_CONTINUE_SCAN:
					yyg->yy_c_buf_p = yyg->yytext_ptr + offset;
					break;
				}
			}
		}

	c = *(unsigned char *) yyg->yy_c_buf_p;	/* cast for 8-bit char's */
	*yyg->yy_c_buf_p = '\0';	/* preserve yytext */
	yyg->yy_hold_char = *++yyg->yy_c_buf_p;

	YY_CURRENT_BUFFER_LVALUE->yy_at_bol = (c == '\n');
	if ( YY_CURRENT_BUFFER_LVALUE->yy_at_bol )

    do{ yylineno++;
        yycolumn=0;
    }while(0)
;

	return c;
}
#endif	/* ifndef YY_NO_INPUT */

/** Immediately switch to a different input stream.
 * @param input_file A readable stream.
 * @param yyscanner The scanner object.
 * @note This function does not reset the start condition to @c INITIAL .
 */
    void yyrestart  (FILE * input_file , yyscan_t yyscanner)
{
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;

	if ( ! YY_CURRENT_BUFFER ){
        yyensure_buffer_stack (yyscanner);
		YY_CURRENT_BUFFER_LVALUE =
            yy_create_buffer( yyin, YY_BUF_SIZE , yyscanner);
	}

	yy_init_buffer( YY_CURRENT_BUFFER, input_file , yyscanner);
	yy_load_buffer_state( yyscanner );
}

/** Switch to a different input buffer.
 * @param new_buffer The new input buffer.
 * @param yyscanner The scanner object.
 */
    void yy_switch_to_buffer  (YY_BUFFER_STATE  new_buffer , yyscan_t yyscanner)
{
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;

	/* TODO. We should be able to replace this entire function body
	 * with
	 *		yypop_buffer_state();
	 *		yypush_buffer_state(new_buffer);
     */
	yyensure_buffer_stack (yyscanner);
	if ( YY_CURRENT_BUFFER == new_buffer )
		return;

	if ( YY_CURRENT_BUFFER )
		{
		/* Flush out information for old buffer. */
		*yyg->yy_c_buf_p = yyg->yy_hold_char;
		YY_CURRENT_BUFFER_LVALUE->yy_buf_pos = yyg->yy_c_buf_p;
		YY_CURRENT_BUFFER_LVALUE->yy_n_chars = yyg->yy_n_chars;
		}

	YY_CURRENT_BUFFER_LVALUE = new_buffer;
	yy_load_buffer_state( yyscanner );

	/* We don't actually know whether we did this switch during
	 * EOF (yywrap()) processing, but the only time this flag
	 * is looked at is after yywrap() is called, so it's safe
	 * to go ahead and always set it.
	 */
	yyg->yy_did_buffer_switch_on_eof = 1;
}

static void yy_load_buffer_state  (yyscan_t yyscanner)
{
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;
	yyg->yy_n_chars = YY_CURRENT_BUFFER_LVALUE->yy_n_chars;
	yyg->yytext_ptr = yyg->yy_c_buf_p = YY_CURRENT_BUFFER_LVALUE->yy_buf_pos;
	yyin = YY_CURRENT_BUFFER_LVALUE->yy_input_file;
	yyg->yy_hold_char = *yyg->yy_c_buf_p;
}

/** Allocate and initialize an input buffer state.
 * @param file A readable stream.
 * @param size The character buffer size in bytes. When in doubt, use @c YY_BUF_SIZE.
 * @param yyscanner The scanner object.
 * @return the allocated buffer state.
 */
    YY_BUFFER_STATE yy_create_buffer  (FILE * file, int  size , yyscan_t yyscanner)
{
	YY_BUFFER_STATE b;

	b = (YY_BUFFER_STATE) yyalloc( sizeof( struct yy_buffer_state ) , yyscanner );
	if ( ! b )
		YY_FATAL_ERROR( "out of dynamic memory in yy_create_buffer()" );

	b->yy_buf_size = size;

	/* yy_ch_buf has to be 2 characters longer than the size given because
	 * we need to put in 2 end-of-buffer characters.
	 */
	b->yy_ch_buf = (char *) yyalloc( (yy_size_t) (b->yy_buf_size + 2) , yyscanner );
	if ( ! b->yy_ch_buf )
		YY_FATAL_ERROR( "out of dynamic memory in yy_create_buffer()" );

	b->yy_is_our_buffer = 1;

	yy_init_buffer( b, file , yyscanner);

	return b;
}

/** Destroy the buffer.
 * @param b a buffer created with yy_create_buffer()
 * @param yyscanner The scanner object.
 */
    void yy_delete_buffer (YY_BUFFER_STATE  b , yyscan_t yyscanner)
{
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;

	if ( ! b )
		return;

	if ( b == YY_CURRENT_BUFFER ) /* Not sure if we should pop here. */
		YY_CURRENT_BUFFER_LVALUE = (YY_BUFFER_STATE) 0;

	if ( b->yy_is_our_buffer )
		yyfree( (void *) b->yy_ch_buf , yyscanner );

	yyfree( (void *) b , yyscanner );
}

/* Initializes or reinitializes a buffer.
 * This function is sometimes called more than once on the same buffer,
 * such as during a yyrestart() or at EOF.
 */
    static void yy_init_buffer  (YY_BUFFER_STATE  b, FILE * file , yyscan_t yyscanner)

{
	int oerrno = errno;
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;

	yy_flush_buffer( b , yyscanner);

	b->yy_input_file = file;
	b->yy_fill_buffer = 1;

    /* If b is the current buffer, then yy_init_buffer was _probably_
     * called from yyrestart() or through yy_get_next_buffer.
     * In that case, we don't want to reset the lineno or column.
     */
    if (b != YY_CURRENT_BUFFER){
        b->yy_bs_lineno = 1;
        b->yy_bs_column = 0;
    }

        b->yy_is_interactive = 0;

	errno = oerrno;
}

/** Discard all buffered characters. On the next scan, YY_INPUT will be called.
 * @param b the buffer state to be flushed, usually @c YY_CURRENT_BUFFER.
 * @param yyscanner The scanner object.
 */
    void yy_flush_buffer (YY_BUFFER_STATE  b , yyscan_t yyscanner)
{
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;
	if ( ! b )
		return;

	b->yy_n_chars = 0;

	/* We always need two end-of-buffer characters.  The first causes
	 * a transition to the end-of-buffer state.  The second causes
	 * a jam in that state.
	 */
	b->yy_ch_buf[0] = YY_END_OF_BUFFER_CHAR;
	b->yy_ch_buf[1] = YY_END_OF_BUFFER_CHAR;

	b->yy_buf_pos = &b->yy_ch_buf[0];

	b->yy_at_bol = 1;
	b->yy_buffer_status = YY_BUFFER_NEW;

	if ( b == YY_CURRENT_BUFFER )
		yy_load_buffer_state( yyscanner );
}

/** Pushes the new state onto the stack. The new state becomes
 *  the current state. This function will allocate the stack
 *  if necessary.
 *  @param new_buffer The new state.
 *  @param yyscanner The scanner object.
 */
void yypush_buffer_state (YY_BUFFER_STATE new_buffer , yyscan_t yyscanner)
{
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;
	if (new_buffer == NULL)
		return;

	yyensure_buffer_stack(yyscanner);

	/* This block is copied from yy_switch_to_buffer. */
	if ( YY_CURRENT_BUFFER )
		{
		/* Flush out information for old buffer. */
		*yyg->yy_c_buf_p = yyg->yy_hold_char;
		YY_CURRENT_BUFFER_LVALUE->yy_buf_pos = yyg->yy_c_buf_p;
		YY_CURRENT_BUFFER_LVALUE->yy_n_chars = yyg->yy_n_chars;
		}

	/* Only push if top exists. Otherwise, replace top. */
	if (YY_CURRENT_BUFFER)
		yyg->yy_buffer_stack_top++;
	YY_CURRENT_BUFFER_LVALUE = new_buffer;

	/* copied from yy_switch_to_buffer. */
	yy_load_buffer_state( yyscanner );
	yyg->yy_did_buffer_switch_on_eof = 1;
}

/** Removes and deletes the top of the stack, if present.
 *  The next element becomes the new top.
 *  @param yyscanner The scanner object.
 */
void yypop_buffer_state (yyscan_t yyscanner)
{
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;
	if (!YY_CURRENT_BUFFER)
		return;

	yy_delete_buffer(YY_CURRENT_BUFFER , yyscanner);
	YY_CURRENT_BUFFER_LVALUE = NULL;
	if (yyg->yy_buffer_stack_top > 0)
		--yyg->yy_buffer_stack_top;

	if (YY_CURRENT_BUFFER) {
		yy_load_buffer_state( yyscanner );
		yyg->yy_did_buffer_switch_on_eof = 1;
	}
}

/* Allocates the stack if it does not exist.
 *  Guarantees space for at least one push.
 */
static void yyensure_buffer_stack (yyscan_t yyscanner)
{
	yy_size_t num_to_alloc;
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;

	if (!yyg->yy_buffer_stack) {

		/* First allocation is just for 2 elements, since we don't know if this
		 * scanner will even need a stack. We use 2 instead of 1 to avoid an
		 * immediate realloc on the next call.
         */
      num_to_alloc = 1; /* After all that talk, this was set to 1 anyways... */
		yyg->yy_buffer_stack = (struct yy_buffer_state**)yyalloc
								(num_to_alloc * sizeof(struct yy_buffer_state*)
								, yyscanner);
		if ( ! yyg->yy_buffer_stack )
			YY_FATAL_ERROR( "out of dynamic memory in yyensure_buffer_stack()" );

		memset(yyg->yy_buffer_stack, 0, num_to_alloc * sizeof(struct yy_buffer_state*));

		yyg->yy_buffer_stack_max = num_to_alloc;
		yyg->yy_buffer_stack_top = 0;
		return;
	}

	if (yyg->yy_buffer_stack_top >= (yyg->yy_buffer_stack_max) - 1){

		/* Increase the buffer to prepare for a possible push. */
		yy_size_t grow_size = 8 /* arbitrary grow size */;

		num_to_alloc = yyg->yy_buffer_stack_max + grow_size;
		yyg->yy_buffer_stack = (struct yy_buffer_state**)yyrealloc
								(yyg->yy_buffer_stack,
								num_to_alloc * sizeof(struct yy_buffer_state*)
								, yyscanner);
		if ( ! yyg->yy_buffer_stack )
			YY_FATAL_ERROR( "out of dynamic memory in yyensure_buffer_stack()" );

		/* zero only the new slots.*/
		memset(yyg->yy_buffer_stack + yyg->yy_buffer_stack_max, 0, grow_size * sizeof(struct yy_buffer_state*));
		yyg->yy_buffer_stack_max = num_to_alloc;
	}
}

/** Setup the input buffer state to scan directly from a user-specified character buffer.
 * @param base the character buffer
 * @param size the size in bytes of the character buffer
 * @param yyscanner The scanner object.
 * @return the newly allocated buffer state object.
 */
YY_BUFFER_STATE yy_scan_buffer  (char * base, yy_size_t  size , yyscan_t yyscanner)
{
	YY_BUFFER_STATE b;

	if ( size < 2 ||
	     base[size-2] != YY_END_OF_BUFFER_CHAR ||
	     base[size-1] != YY_END_OF_BUFFER_CHAR )
		/* They forgot to leave room for the EOB's. */
		return NULL;

	b = (YY_BUFFER_STATE) yyalloc( sizeof( struct yy_buffer_state ) , yyscanner );
	if ( ! b )
		YY_FATAL_ERROR( "out of dynamic memory in yy_scan_buffer()" );

	b->yy_buf_size = (int) (size - 2);	/* "- 2" to take care of EOB's */
	b->yy_buf_pos = b->yy_ch_buf = base;
	b->yy_is_our_buffer = 0;
	b->yy_input_file = NULL;
	b->yy_n_chars = b->yy_buf_size;
	b->yy_is_interactive = 0;
	b->yy_at_bol = 1;
	b->yy_fill_buffer = 0;
	b->yy_buffer_status = YY_BUFFER_NEW;

	yy_switch_to_buffer( b , yyscanner );

	return b;
}

/** Setup the input buffer state to scan a string. The next call to yylex() will
 * scan from a @e copy of @a str.
 * @param yystr a NUL-terminated string to scan
 * @param yyscanner The scanner object.
 * @return the newly allocated buffer state object.
 * @note If you want to scan bytes that may contain NUL values, then use
 *       yy_scan_bytes() instead.
 */
YY_BUFFER_STATE yy_scan_string (const char * yystr , yyscan_t yyscanner)
{

	return yy_scan_bytes( yystr, (int) strlen(yystr) , yyscanner);
}

/** Setup the input buffer state to scan the given bytes. The next call to yylex() will
 * scan from a @e copy of @a bytes.
 * @param yybytes the byte buffer to scan
 * @param _yybytes_len the number of bytes in the buffer pointed to by @a bytes.
 * @param yyscanner The scanner object.
 * @return the newly allocated buffer state object.
 */
YY_BUFFER_STATE yy_scan_bytes  (const char * yybytes, int  _yybytes_len , yyscan_t yyscanner)
{
	YY_BUFFER_STATE b;
	char *buf;
	yy_size_t n;
	int i;

	/* Get memory for full buffer, including space for trailing EOB's. */
	n = (yy_size_t) (_yybytes_len + 2);
	buf = (char *) yyalloc( n , yyscanner );
	if ( ! buf )
		YY_FATAL_ERROR( "out of dynamic memory in yy_scan_bytes()" );

	for ( i = 0; i < _yybytes_len; ++i )
		buf[i] = yybytes[i];

	buf[_yybytes_len] = buf[_yybytes_len+1] = YY_END_OF_BUFFER_CHAR;

	b = yy_scan_buffer( buf, n , yyscanner);
	if ( ! b )
		YY_FATAL_ERROR( "bad buffer in yy_scan_bytes()" );

	/* It's okay to grow etc. this buffer, and we should throw it
	 * away when we're done.
	 */
	b->yy_is_our_buffer = 1;

	return b;
}

#ifndef YY_EXIT_FAILURE
#define YY_EXIT_FAILURE 2
#endif

static void yynoreturn yy_fatal_error (const char* msg , yyscan_t yyscanner)
{
	struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;
	(void)yyg;
	fprintf( stderr, "%s\n", msg );
	exit( YY_EXIT_FAILURE );
}

/* Redefine yyless() so it works in section 3 code. */

#undef yyless
#define yyless(n) \
	do \
		{ \
		/* Undo effects of setting up yytext. */ \
        int yyless_macro_arg = (n); \
        YY_LESS_LINENO(yyless_macro_arg);\
		yytext[yyleng] = yyg->yy_hold_char; \
		yyg->yy_c_buf_p = yytext + yyless_macro_arg; \
		yyg->yy_hold_char = *yyg->yy_c_buf_p; \
		*yyg->yy_c_buf_p = '\0'; \
		yyleng = yyless_macro_arg; \
		} \
	while ( 0 )

/* Accessor  methods (get/set functions) to struct members. */

/** Get the user-defined data for this scanner.
 * @param yyscanner The scanner object.
 */
YY_EXTRA_TYPE yyget_extra  (yyscan_t yyscanner)
{
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;
    return yyextra;
}

/** Get the current line number.
 * @param yyscanner The scanner object.
 */
int yyget_lineno  (yyscan_t yyscanner)
{
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;

        if (! YY_CURRENT_BUFFER)
            return 0;

    return yylineno;
}

/** Get the current column number.
 * @param yyscanner The scanner object.
 */
int yyget_column  (yyscan_t yyscanner)
{
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;

        if (! YY_CURRENT_BUFFER)
            return 0;

    return yycolumn;
}

/** Get the input stream.
 * @param yyscanner The scanner object.
 */
FILE *yyget_in  (yyscan_t yyscanner)
{
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;
    return yyin;
}

/** Get the output stream.
 * @param yyscanner The scanner object.
 */
FILE *yyget_out  (yyscan_t yyscanner)
{
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;
    return yyout;
}

/** Get the length of the current token.
 * @param yyscanner The scanner object.
 */
int yyget_leng  (yyscan_t yyscanner)
{
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;
    return yyleng;
}

/** Get the current token.
 * @param yyscanner The scanner object.
 */

char *yyget_text  (yyscan_t yyscanner)
{
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;
    return yytext;
}

/** Set the user-defined data. This data is never touched by the scanner.
 * @param user_defined The data to be associated with this scanner.
 * @param yyscanner The scanner object.
 */
void yyset_extra (YY_EXTRA_TYPE  user_defined , yyscan_t yyscanner)
{
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;
    yyextra = user_defined ;
}

/** Set the current line number.
 * @param _line_number line number
 * @param yyscanner The scanner object.
 */
void yyset_lineno (int  _line_number , yyscan_t yyscanner)
{
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;

        /* lineno is only valid if an input buffer exists. */
        if (! YY_CURRENT_BUFFER )
           YY_FATAL_ERROR( "yyset_lineno called with no buffer" );

    yylineno = _line_number;
}

/** Set the current column.
 * @param _column_no column number
 * @param yyscanner The scanner object.
 */
void yyset_column (int  _column_no , yyscan_t yyscanner)
{
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;

        /* column is only valid if an input buffer exists. */
        if (! YY_CURRENT_BUFFER )
           YY_FATAL_ERROR( "yyset_column called with no buffer" );

    yycolumn = _column_no;
}

/** Set the input stream. This does not discard the current
 * input buffer.
 * @param _in_str A readable stream.
 * @param yyscanner The scanner object.
 * @see yy_switch_to_buffer
 */
void yyset_in (FILE *  _in_str , yyscan_t yyscanner)
{
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;
    yyin = _in_str ;
}

void yyset_out (FILE *  _out_str , yyscan_t yyscanner)
{
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;
    yyout = _out_str ;
}

int yyget_debug  (yyscan_t yyscanner)
{
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;
    return yy_flex_debug;
}

void yyset_debug (int  _bdebug , yyscan_t yyscanner)
{
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;
    yy_flex_debug = _bdebug ;
}

/* Accessor methods for yylval and yylloc */

YYSTYPE * yyget_lval  (yyscan_t yyscanner)
{
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;
    return yylval;
}

void yyset_lval (YYSTYPE *  yylval_param , yyscan_t yyscanner)
{
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;
    yylval = yylval_param;
}

/* User-visible API */

/* yylex_init is special because it creates the scanner itself, so it is
 * the ONLY reentrant function that doesn't take the scanner as the last argument.
 * That's why we explicitly handle the declaration, instead of using our macros.
 */
int yylex_init(yyscan_t* ptr_yy_globals)
{
    if (ptr_yy_globals == NULL){
        errno = EINVAL;
        return 1;
    }

    *ptr_yy_globals = (yyscan_t) yyalloc ( sizeof( struct yyguts_t ), NULL );

    if (*ptr_yy_globals == NULL){
        errno = ENOMEM;
        return 1;
    }

    /* By setting to 0xAA, we expose bugs in yy_init_globals. Leave at 0x00 for releases. */
    memset(*ptr_yy_globals,0x00,sizeof(struct yyguts_t));

    return yy_init_globals ( *ptr_yy_globals );
}

/* yylex_init_extra has the same functionality as yylex_init, but follows the
 * convention of taking the scanner as the last argument. Note however, that
 * this is a *pointer* to a scanner, as it will be allocated by this call (and
 * is the reason, too, why this function also must handle its own declaration).
 * The user defined value in the first argument will be available to yyalloc in
 * the yyextra field.
 */
int yylex_init_extra( YY_EXTRA_TYPE yy_user_defined, yyscan_t* ptr_yy_globals )
{
    struct yyguts_t dummy_yyguts;

    yyset_extra (yy_user_defined, &dummy_yyguts);

    if (ptr_yy_globals == NULL){
        errno = EINVAL;
        return 1;
    }

    *ptr_yy_globals = (yyscan_t) yyalloc ( sizeof( struct yyguts_t ), &dummy_yyguts );

    if (*ptr_yy_globals == NULL){
        errno = ENOMEM;
        return 1;
    }

    /* By setting to 0xAA, we expose bugs in
    yy_init_globals. Leave at 0x00 for releases. */
    memset(*ptr_yy_globals,0x00,sizeof(struct yyguts_t));

    yyset_extra (yy_user_defined, *ptr_yy_globals);

    return yy_init_globals ( *ptr_yy_globals );
}

static int yy_init_globals (yyscan_t yyscanner)
{
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;
    /* Initialization is the same as for the non-reentrant scanner.
     * This function is called from yylex_destroy(), so don't allocate here.
     */

    yyg->yy_buffer_stack = NULL;
    yyg->yy_buffer_stack_top = 0;
    yyg->yy_buffer_stack_max = 0;
    yyg->yy_c_buf_p = NULL;
    yyg->yy_init = 0;
    yyg->yy_start = 0;

    yyg->yy_start_stack_ptr = 0;
    yyg->yy_start_stack_depth = 0;
    yyg->yy_start_stack =  NULL;

/* Defined in main.c */
#ifdef YY_STDINIT
    yyin = stdin;
    yyout = stdout;
#else
    yyin = NULL;
    yyout = NULL;
#endif

    /* For future reference: Set errno on error, since we are called by
     * yylex_init()
     */
    return 0;
}

/* yylex_destroy is for both reentrant and non-reentrant scanners. */
int yylex_destroy  (yyscan_t yyscanner)
{
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;

    /* Pop the buffer stack, destroying each element. */
	while(YY_CURRENT_BUFFER){
		yy_delete_buffer( YY_CURRENT_BUFFER , yyscanner );
		YY_CURRENT_BUFFER_LVALUE = NULL;
		yypop_buffer_state(yyscanner);
	}

	/* Destroy the stack itself. */
	yyfree(yyg->yy_buffer_stack , yyscanner);
	yyg->yy_buffer_stack = NULL;

    /* Destroy the start condition stack. */
        yyfree( yyg->yy_start_stack , yyscanner );
        yyg->yy_start_stack = NULL;

    /* Reset the globals. This is important in a non-reentrant scanner so the next time
     * yylex() is called, initialization will occur. */
    yy_init_globals( yyscanner);

    /* Destroy the main struct (reentrant only). */
    yyfree ( yyscanner , yyscanner );
    yyscanner = NULL;
    return 0;
}

/*
 * Internal utility routines.
 */

#ifndef yytext_ptr
static void yy_flex_strncpy (char* s1, const char * s2, int n , yyscan_t yyscanner)
{
	struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;
	(void)yyg;

	int i;
	for ( i = 0; i < n; ++i )
		s1[i] = s2[i];
}
#endif

#ifdef YY_NEED_STRLEN
static int yy_flex_strlen (const char * s , yyscan_t yyscanner)
{
	int n;
	for ( n = 0; s[n]; ++n )
		;

	return n;
}
#endif

void *yyalloc (yy_size_t  size , yyscan_t yyscanner)
{
	struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;
	(void)yyg;
	return malloc(size);
}

void *yyrealloc  (void * ptr, yy_size_t  size , yyscan_t yyscanner)
{
	struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;
	(void)yyg;

	/* The cast to (char *) in the following accommodates both
	 * implementations that use char* generic pointers, and those
	 * that use void* generic pointers.  It works with the latter
	 * because both ANSI C and C++ allow castless assignment from
	 * any pointer type to void*, and deal with argument conversions
	 * as though doing an assignment.
	 */
	return realloc(ptr, size);
}

void yyfree (void * ptr , yyscan_t yyscanner)
{
	struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;
	(void)yyg;
	free( (char *) ptr );	/* see yyrealloc() for (char *) cast */
}

#define YYTABLES_NAME "yytables"

#line 203 "scanner.l"
/* end libconfig/lib/scanner.c */
/* begin libconfig/lib/libconfig.c */
/* ----------------------------------------------------------------------------
   libconfig - A library for processing structured configuration files
   Copyright (C) 2005-2020  Mark A Lindner

   This file is part of libconfig.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License
   as published by the Free Software Foundation; either version 2.1 of
   the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Library General Public
   License along with this library; if not, see
   <http://www.gnu.org/licenses/>.
   ----------------------------------------------------------------------------
*/

#ifdef HAVE_CONFIG_H
#endif

#include <locale.h>

#if defined(HAVE_XLOCALE_H) || defined(__APPLE__)
#include <xlocale.h>
#endif

#include <ctype.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>


#define PATH_TOKENS ":./"
#define CHUNK_SIZE 16
#define DEFAULT_TAB_WIDTH 2
#define DEFAULT_FLOAT_PRECISION 6

/* ------------------------------------------------------------------------- */

#ifndef LIBCONFIG_STATIC
#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__) \
  || defined(WIN64) || defined(_WIN64) || defined(__WIN64__))

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved)
{
  return(TRUE);
}

#endif /* WIN32 || WIN64 */
#endif /* LIBCONFIG_STATIC */

/* ------------------------------------------------------------------------- */

static const char *__io_error = "file I/O error";

static void __config_list_destroy(config_list_t *list);
static void __config_write_setting(const config_t *config,
                                   const config_setting_t *setting,
                                   FILE *stream, int depth);

/* ------------------------------------------------------------------------- */

static void __config_locale_override(void)
{
#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__)) \
  && ! defined(__MINGW32__)

  _configthreadlocale(_ENABLE_PER_THREAD_LOCALE);
  setlocale(LC_NUMERIC, "C");

#elif defined(__APPLE__)

  locale_t loc = newlocale(LC_NUMERIC_MASK, "C", NULL);
  uselocale(loc);

#elif ((defined HAVE_NEWLOCALE) && (defined HAVE_USELOCALE))

  locale_t loc = newlocale(LC_NUMERIC, "C", NULL);
  uselocale(loc);

#else

#warning "No way to modify calling thread's locale!"

#endif
}

/* ------------------------------------------------------------------------- */

static void __config_locale_restore(void)
{
#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__)) \
  && ! defined(__MINGW32__)

    _configthreadlocale(_DISABLE_PER_THREAD_LOCALE);

#elif ((defined HAVE_USELOCALE) && (defined HAVE_FREELOCALE))

  locale_t loc = uselocale(LC_GLOBAL_LOCALE);
  freelocale(loc);

#else

#warning "No way to modify calling thread's locale!"

#endif
}

/* ------------------------------------------------------------------------- */

static int __config_name_compare(const char *a, const char *b)
{
  const char *p, *q;

  for(p = a, q = b; ; p++, q++)
  {
    int pd = ((! *p) || strchr(PATH_TOKENS, *p));
    int qd = ((! *q) || strchr(PATH_TOKENS, *q));

    if(pd && qd)
      break;
    else if(pd)
      return(-1);
    else if(qd)
      return(1);
    else if(*p < *q)
      return(-1);
    else if(*p > *q)
      return(1);
  }

  return(0);
}

/* ------------------------------------------------------------------------- */

static void __config_indent(FILE *stream, int depth, unsigned short w)
{
  if(w)
    fprintf(stream, "%*s", (depth - 1) * w, " ");
  else
  {
    int i;
    for(i = 0; i < (depth - 1); ++i)
      fputc('\t', stream);
  }
}

/* ------------------------------------------------------------------------- */

static void __config_write_value(const config_t *config,
                                 const config_value_t *value, int type,
                                 int format, int depth, FILE *stream)
{
  char fbuf[64];

  switch(type)
  {
    /* boolean */
    case CONFIG_TYPE_BOOL:
      fputs(value->ival ? "true" : "false", stream);
      break;

    /* int */
    case CONFIG_TYPE_INT:
      switch(format)
      {
        case CONFIG_FORMAT_HEX:
          fprintf(stream, "0x%X", value->ival);
          break;

        case CONFIG_FORMAT_DEFAULT:
        default:
          fprintf(stream, "%d", value->ival);
          break;
      }
      break;

    /* 64-bit int */
    case CONFIG_TYPE_INT64:
      switch(format)
      {
        case CONFIG_FORMAT_HEX:
          fprintf(stream, "0x" INT64_HEX_FMT "L", value->llval);
          break;

        case CONFIG_FORMAT_DEFAULT:
        default:
          fprintf(stream, INT64_FMT "L", value->llval);
          break;
      }
      break;

    /* float */
    case CONFIG_TYPE_FLOAT:
    {
      const int sci_ok = config_get_option(
            config, CONFIG_OPTION_ALLOW_SCIENTIFIC_NOTATION);
      libconfig_format_double(value->fval, config->float_precision, sci_ok,
                              fbuf, sizeof(fbuf));
      fputs(fbuf, stream);
      break;
    }

    /* string */
    case CONFIG_TYPE_STRING:
    {
      char *p;

      fputc('\"', stream);

      if(value->sval)
      {
        for(p = value->sval; *p; p++)
        {
          int c = (int)*p & 0xFF;
          switch(c)
          {
            case '\"':
            case '\\':
              fputc('\\', stream);
              fputc(c, stream);
              break;

            case '\n':
              fputs("\\n", stream);
              break;

            case '\r':
              fputs("\\r", stream);
              break;

            case '\f':
              fputs("\\f", stream);
              break;

            case '\t':
              fputs("\\t", stream);
              break;

            default:
              if(c >= ' ')
                fputc(c, stream);
              else
                fprintf(stream, "\\x%02X", c);
          }
        }
      }
      fputc('\"', stream);
      break;
    }

    /* list */
    case CONFIG_TYPE_LIST:
    {
      config_list_t *list = value->list;

      fputs("( ", stream);

      if(list)
      {
        int len = list->length;
        config_setting_t **s;

        for(s = list->elements; len--; s++)
        {
          __config_write_value(config, &((*s)->value), (*s)->type,
                               config_setting_get_format(*s), depth + 1,
                               stream);

          if(len)
            fputc(',', stream);

          fputc(' ', stream);
        }
      }

      fputc(')', stream);
      break;
    }

    /* array */
    case CONFIG_TYPE_ARRAY:
    {
      config_list_t *list = value->list;

      fputs("[ ", stream);

      if(list)
      {
        int len = list->length;
        config_setting_t **s;

        for(s = list->elements; len--; s++)
        {
          __config_write_value(config, &((*s)->value), (*s)->type,
                               config_setting_get_format(*s), depth + 1,
                               stream);

          if(len)
            fputc(',', stream);

          fputc(' ', stream);
        }
      }

      fputc(']', stream);
      break;
    }

    /* group */
    case CONFIG_TYPE_GROUP:
    {
      config_list_t *list = value->list;

      if(depth > 0)
      {
        if(config_get_option(config, CONFIG_OPTION_OPEN_BRACE_ON_SEPARATE_LINE))
        {
          fputc('\n', stream);

          if(depth > 1)
            __config_indent(stream, depth, config->tab_width);
        }

        fputs("{\n", stream);
      }

      if(list)
      {
        int len = list->length;
        config_setting_t **s;

        for(s = list->elements; len--; s++)
          __config_write_setting(config, *s, stream, depth + 1);
      }

      if(depth > 1)
        __config_indent(stream, depth, config->tab_width);

      if(depth > 0)
        fputc('}', stream);

      break;
    }

    default:
      /* this shouldn't happen, but handle it gracefully... */
      fputs("???", stream);
      break;
  }
}

/* ------------------------------------------------------------------------- */

static void __config_list_add(config_list_t *list, config_setting_t *setting)
{
  if((list->length % CHUNK_SIZE) == 0)
  {
    list->elements = (config_setting_t **)realloc(
      list->elements,
      (list->length + CHUNK_SIZE) * sizeof(config_setting_t *));
  }

  list->elements[list->length] = setting;
  list->length++;
}

/* ------------------------------------------------------------------------- */

static config_setting_t *__config_list_search(config_list_t *list,
                                              const char *name,
                                              unsigned int *idx)
{
  config_setting_t **found = NULL;
  unsigned int i;

  if(! list)
    return(NULL);

  for(i = 0, found = list->elements; i < list->length; i++, found++)
  {
    if(! (*found)->name)
      continue;

    if(! __config_name_compare(name, (*found)->name))
    {
      if(idx)
        *idx = i;

      return(*found);
    }
  }

  return(NULL);
}

/* ------------------------------------------------------------------------- */

static config_setting_t *__config_list_remove(config_list_t *list, int idx)
{
  config_setting_t *removed = *(list->elements + idx);
  int offset = (idx * sizeof(config_setting_t *));
  int len = list->length - 1 - idx;
  char *base = (char *)list->elements + offset;

  memmove(base, base + sizeof(config_setting_t *),
          len * sizeof(config_setting_t *));

  list->length--;

  /* possibly realloc smaller? */

  return(removed);
}

/* ------------------------------------------------------------------------- */

static void __config_setting_destroy(config_setting_t *setting)
{
  if(setting)
  {
    if(setting->name)
      __delete(setting->name);

    if(setting->type == CONFIG_TYPE_STRING)
      __delete(setting->value.sval);

    else if(config_setting_is_aggregate(setting))
    {
      if(setting->value.list)
        __config_list_destroy(setting->value.list);
    }

    if(setting->hook && setting->config->destructor)
      setting->config->destructor(setting->hook);

    __delete(setting);
  }
}

/* ------------------------------------------------------------------------- */

static void __config_list_destroy(config_list_t *list)
{
  config_setting_t **p;
  unsigned int i;

  if(! list)
    return;

  if(list->elements)
  {
    for(p = list->elements, i = 0; i < list->length; p++, i++)
      __config_setting_destroy(*p);

    __delete(list->elements);
  }

  __delete(list);
}

/* ------------------------------------------------------------------------- */

static int __config_list_checktype(const config_setting_t *setting, int type)
{
  /* if the array is empty, then it has no type yet */

  if(! setting->value.list)
    return(CONFIG_TRUE);

  if(setting->value.list->length == 0)
    return(CONFIG_TRUE);

  /* if it's a list, any type is allowed */

  if(setting->type == CONFIG_TYPE_LIST)
    return(CONFIG_TRUE);

  /* otherwise the first element added determines the type of the array */

  return((setting->value.list->elements[0]->type == type)
         ? CONFIG_TRUE : CONFIG_FALSE);
}

/* ------------------------------------------------------------------------- */

static int __config_type_is_scalar(int type)
{
  return((type >= CONFIG_TYPE_INT) && (type <= CONFIG_TYPE_BOOL));
}

/* ------------------------------------------------------------------------- */

static int __config_validate_name(const char *name)
{
  const char *p = name;

  if(*p == '\0')
    return(CONFIG_FALSE);

  if(! isalpha((int)*p) && (*p != '*'))
    return(CONFIG_FALSE);

  for(++p; *p; ++p)
  {
    if(! (isalpha((int)*p) || isdigit((int)*p) || strchr("*_-", (int)*p)))
      return(CONFIG_FALSE);
  }

  return(CONFIG_TRUE);
}

/* ------------------------------------------------------------------------- */

static int __config_read(config_t *config, FILE *stream, const char *filename,
                         const char *str)
{
  yyscan_t scanner;
  struct scan_context scan_ctx;
  struct parse_context parse_ctx;
  int r;

  config_clear(config);

  libconfig_parsectx_init(&parse_ctx);
  parse_ctx.config = config;
  parse_ctx.parent = config->root;
  parse_ctx.setting = config->root;

  __config_locale_override();

  libconfig_scanctx_init(&scan_ctx, filename);
  config->root->file = libconfig_scanctx_current_filename(&scan_ctx);
  scan_ctx.config = config;
  libconfig_yylex_init_extra(&scan_ctx, &scanner);

  if(stream)
    libconfig_yyrestart(stream, scanner);
  else /* read from string */
    (void)libconfig_yy_scan_string(str, scanner);

  libconfig_yyset_lineno(1, scanner);
  r = libconfig_yyparse(scanner, &parse_ctx, &scan_ctx);

  if(r != 0)
  {
    YY_BUFFER_STATE buf;

    config->error_file = libconfig_scanctx_current_filename(&scan_ctx);
    config->error_type = CONFIG_ERR_PARSE;

    /* Unwind the include stack, freeing the buffers and closing the files. */
    while((buf = (YY_BUFFER_STATE)libconfig_scanctx_pop_include(&scan_ctx))
          != NULL)
      libconfig_yy_delete_buffer(buf, scanner);
  }

  libconfig_yylex_destroy(scanner);
  config->filenames = libconfig_scanctx_cleanup(&scan_ctx);
  libconfig_parsectx_cleanup(&parse_ctx);

  __config_locale_restore();

  return(r == 0 ? CONFIG_TRUE : CONFIG_FALSE);
}

/* ------------------------------------------------------------------------- */

int config_read(config_t *config, FILE *stream)
{
  return(__config_read(config, stream, NULL, NULL));
}

/* ------------------------------------------------------------------------- */

int config_read_string(config_t *config, const char *str)
{
  return(__config_read(config, NULL, NULL, str));
}

/* ------------------------------------------------------------------------- */

static void __config_write_setting(const config_t *config,
                                   const config_setting_t *setting,
                                   FILE *stream, int depth)
{
  char group_assign_char = config_get_option(
    config, CONFIG_OPTION_COLON_ASSIGNMENT_FOR_GROUPS) ? ':' : '=';

  char nongroup_assign_char = config_get_option(
    config, CONFIG_OPTION_COLON_ASSIGNMENT_FOR_NON_GROUPS) ? ':' : '=';

  if(depth > 1)
    __config_indent(stream, depth, config->tab_width);


  if(setting->name)
  {
    fputs(setting->name, stream);
    fprintf(stream, " %c ", ((setting->type == CONFIG_TYPE_GROUP)
                             ? group_assign_char
                             : nongroup_assign_char));
  }

  __config_write_value(config, &(setting->value), setting->type,
                       config_setting_get_format(setting), depth, stream);

  if(depth > 0)
  {
    if(config_get_option(config, CONFIG_OPTION_SEMICOLON_SEPARATORS))
      fputc(';', stream);

    fputc('\n', stream);
  }
}

/* ------------------------------------------------------------------------- */

void config_write(const config_t *config, FILE *stream)
{
  __config_locale_override();

  __config_write_setting(config, config->root, stream, 0);

  __config_locale_restore();
}

/* ------------------------------------------------------------------------- */

int config_read_file(config_t *config, const char *filename)
{
  int ret, ok = 0;

  FILE *stream = fopen(filename, "rt");
  if(stream != NULL)
  {
    // On some operating systems, fopen() succeeds on a directory.
    int fd = fileno(stream);
    struct stat statbuf;

    if(fstat(fd, &statbuf) == 0)
    {
      // Only proceed if this is not a directory.
      if(!S_ISDIR(statbuf.st_mode))
        ok = 1;
    }
  }

  if(!ok)
  {
    if(stream != NULL)
      fclose(stream);

    config->error_text = __io_error;
    config->error_type = CONFIG_ERR_FILE_IO;
    return(CONFIG_FALSE);
  }

  ret = __config_read(config, stream, filename, NULL);
  fclose(stream);

  return(ret);
}

/* ------------------------------------------------------------------------- */

int config_write_file(config_t *config, const char *filename)
{
  FILE *stream = fopen(filename, "wt");
  if(stream == NULL)
  {
    config->error_text = __io_error;
    config->error_type = CONFIG_ERR_FILE_IO;
    return(CONFIG_FALSE);
  }

  config_write(config, stream);

  if(config_get_option(config, CONFIG_OPTION_FSYNC))
  {
    int fd = fileno(stream);

    if(fd >= 0)
    {
      if(fsync(fd) != 0)
      {
        fclose(stream);
        config->error_text = __io_error;
        config->error_type = CONFIG_ERR_FILE_IO;
        return(CONFIG_FALSE);
      }
    }
  }

  fclose(stream);
  config->error_type = CONFIG_ERR_NONE;
  return(CONFIG_TRUE);
}

/* ------------------------------------------------------------------------- */

void config_destroy(config_t *config)
{
  __config_setting_destroy(config->root);
  libconfig_strvec_delete(config->filenames);
  __delete(config->include_dir);
  __zero(config);
}

/* ------------------------------------------------------------------------- */

void config_clear(config_t *config)
{
  /* Destroy the root setting (recursively) and then create a new one. */
  __config_setting_destroy(config->root);

  libconfig_strvec_delete(config->filenames);
  config->filenames = NULL;

  config->root = __new(config_setting_t);
  config->root->type = CONFIG_TYPE_GROUP;
  config->root->config = config;
}

/* ------------------------------------------------------------------------- */

void config_set_tab_width(config_t *config, unsigned short width)
{
  /* As per documentation: valid range is 0 - 15. */
  config->tab_width = (width <= 15) ? width : 15;
}

/* ------------------------------------------------------------------------- */

unsigned short config_get_tab_width(const config_t *config)
{
  return config->tab_width;
}

/* ------------------------------------------------------------------------- */

void config_set_float_precision(config_t *config, unsigned short digits)
{
  config->float_precision = digits;
}

/* ------------------------------------------------------------------------- */

unsigned short config_get_float_precision(const config_t *config)
{
  return config->float_precision;
}

/* ------------------------------------------------------------------------- */

void config_init(config_t *config)
{
  __zero(config);
  config_clear(config);

  /* Set default options. */
  config->options = (CONFIG_OPTION_SEMICOLON_SEPARATORS
                     | CONFIG_OPTION_COLON_ASSIGNMENT_FOR_GROUPS
                     | CONFIG_OPTION_OPEN_BRACE_ON_SEPARATE_LINE);
  config->tab_width = DEFAULT_TAB_WIDTH;
  config->float_precision = DEFAULT_FLOAT_PRECISION;
  config->include_fn = config_default_include_func;
}

/* ------------------------------------------------------------------------- */

void config_set_options(config_t *config, int options)
{
  config->options = options;
}

/* ------------------------------------------------------------------------- */

int config_get_options(const config_t *config)
{
  return(config->options);
}

/* ------------------------------------------------------------------------- */

void config_set_option(config_t *config, int option, int flag)
{
  if(flag)
    config->options |= option;
  else
    config->options &= ~option;
}

/* ------------------------------------------------------------------------- */

int config_get_option(const config_t *config, int option)
{
  return((config->options & option) == option);
}

/* ------------------------------------------------------------------------- */

void config_set_hook(config_t *config, void *hook)
{
  config->hook = hook;
}

/* ------------------------------------------------------------------------- */

static config_setting_t *config_setting_create(config_setting_t *parent,
                                               const char *name, int type)
{
  config_setting_t *setting;
  config_list_t *list;

  if(!config_setting_is_aggregate(parent))
    return(NULL);

  setting = __new(config_setting_t);
  setting->parent = parent;
  setting->name = (name == NULL) ? NULL : strdup(name);
  setting->type = type;
  setting->config = parent->config;
  setting->hook = NULL;
  setting->line = 0;

  list = parent->value.list;

  if(! list)
    list = parent->value.list = __new(config_list_t);

  __config_list_add(list, setting);

  return(setting);
}

/* ------------------------------------------------------------------------- */

static int __config_setting_get_int(const config_setting_t *setting,
                                    int *value)
{
  switch(setting->type)
  {
    case CONFIG_TYPE_INT:
      *value = setting->value.ival;
      return(CONFIG_TRUE);

    case CONFIG_TYPE_INT64:
      if((setting->value.llval >= INT_MIN)
         && (setting->value.llval <= INT_MAX))
      {
        *value = (int)(setting->value.llval);
        return(CONFIG_TRUE);
      }
      else
        return(CONFIG_FALSE);

    case CONFIG_TYPE_FLOAT:
      if(config_get_option(setting->config, CONFIG_OPTION_AUTOCONVERT))
      {
        *value = (int)(setting->value.fval);
        return(CONFIG_TRUE);
      }
      else
        return(CONFIG_FALSE);

    default:
      return(CONFIG_FALSE);
  }
}

/* ------------------------------------------------------------------------- */

int config_setting_get_int(const config_setting_t *setting)
{
  int value = 0;
  __config_setting_get_int(setting, &value);
  return(value);
}

/* ------------------------------------------------------------------------- */

static int __config_setting_get_int64(const config_setting_t *setting,
                                      long long *value)
{
  switch(setting->type)
  {
    case CONFIG_TYPE_INT64:
      *value = setting->value.llval;
      return(CONFIG_TRUE);

    case CONFIG_TYPE_INT:
      *value = (long long)(setting->value.ival);
      return(CONFIG_TRUE);

    case CONFIG_TYPE_FLOAT:
      if(config_get_option(setting->config, CONFIG_OPTION_AUTOCONVERT))
      {
        *value = (long long)(setting->value.fval);
        return(CONFIG_TRUE);
      }
      else
        return(CONFIG_FALSE);

    default:
      return(CONFIG_FALSE);
  }
}

/* ------------------------------------------------------------------------- */

long long config_setting_get_int64(const config_setting_t *setting)
{
  long long value = 0;
  __config_setting_get_int64(setting, &value);
  return(value);
}

/* ------------------------------------------------------------------------- */

int config_setting_lookup_int(const config_setting_t *setting,
                              const char *name, int *value)
{
  config_setting_t *member = config_setting_get_member(setting, name);
  if(! member)
    return(CONFIG_FALSE);

  return(__config_setting_get_int(member, value));
}

/* ------------------------------------------------------------------------- */

int config_setting_lookup_int64(const config_setting_t *setting,
                                const char *name, long long *value)
{
  config_setting_t *member = config_setting_get_member(setting, name);
  if(! member)
    return(CONFIG_FALSE);

  return(__config_setting_get_int64(member, value));
}

/* ------------------------------------------------------------------------- */

static int __config_setting_get_float(const config_setting_t *setting,
                                      double *value)
{
  switch(setting->type)
  {
    case CONFIG_TYPE_FLOAT:
      *value = setting->value.fval;
      return(CONFIG_TRUE);

    case CONFIG_TYPE_INT:
      if(config_get_auto_convert(setting->config))
      {
        *value = (double)(setting->value.ival);
        return(CONFIG_TRUE);
      }
      else
        return(CONFIG_FALSE);

    case CONFIG_TYPE_INT64:
      if(config_get_auto_convert(setting->config))
      {
        *value = (double)(setting->value.llval);
        return(CONFIG_TRUE);
      }
      else
        return(CONFIG_FALSE);

    default:
      return(CONFIG_FALSE);
  }
}

/* ------------------------------------------------------------------------- */

double config_setting_get_float(const config_setting_t *setting)
{
  double value = 0.0;
  __config_setting_get_float(setting, &value);
  return(value);
}

/* ------------------------------------------------------------------------- */

int config_setting_lookup_float(const config_setting_t *setting,
                                const char *name, double *value)
{
  config_setting_t *member = config_setting_get_member(setting, name);
  if(! member)
    return(CONFIG_FALSE);

  return(__config_setting_get_float(member, value));
}

/* ------------------------------------------------------------------------- */

int config_setting_lookup_string(const config_setting_t *setting,
                                 const char *name, const char **value)
{
  config_setting_t *member = config_setting_get_member(setting, name);
  if(! member)
    return(CONFIG_FALSE);

  if(config_setting_type(member) != CONFIG_TYPE_STRING)
    return(CONFIG_FALSE);

  *value = config_setting_get_string(member);
  return(CONFIG_TRUE);
}

/* ------------------------------------------------------------------------- */

int config_setting_lookup_bool(const config_setting_t *setting,
                               const char *name, int *value)
{
  config_setting_t *member = config_setting_get_member(setting, name);
  if(! member)
    return(CONFIG_FALSE);

  if(config_setting_type(member) != CONFIG_TYPE_BOOL)
    return(CONFIG_FALSE);

  *value = config_setting_get_bool(member);
  return(CONFIG_TRUE);
}

/* ------------------------------------------------------------------------- */

int config_setting_set_int(config_setting_t *setting, int value)
{
  switch(setting->type)
  {
    case CONFIG_TYPE_NONE:
      setting->type = CONFIG_TYPE_INT;
      /* fall through */

    case CONFIG_TYPE_INT:
      setting->value.ival = value;
      return(CONFIG_TRUE);

    case CONFIG_TYPE_FLOAT:
      if(config_get_auto_convert(setting->config))
      {
        setting->value.fval = (float)value;
        return(CONFIG_TRUE);
      }
      else
        return(CONFIG_FALSE);

    default:
      return(CONFIG_FALSE);
  }
}

/* ------------------------------------------------------------------------- */

int config_setting_set_int64(config_setting_t *setting, long long value)
{
  switch(setting->type)
  {
    case CONFIG_TYPE_NONE:
      setting->type = CONFIG_TYPE_INT64;
      /* fall through */

    case CONFIG_TYPE_INT64:
      setting->value.llval = value;
      return(CONFIG_TRUE);

    case CONFIG_TYPE_INT:
      if((value >= INT_MIN) && (value <= INT_MAX))
      {
        setting->value.ival = (int)value;
        return(CONFIG_TRUE);
      }
      else
        return(CONFIG_FALSE);

    case CONFIG_TYPE_FLOAT:
      if(config_get_auto_convert(setting->config))
      {
        setting->value.fval = (float)value;
        return(CONFIG_TRUE);
      }
      else
        return(CONFIG_FALSE);

    default:
      return(CONFIG_FALSE);
  }
}

/* ------------------------------------------------------------------------- */

int config_setting_set_float(config_setting_t *setting, double value)
{
  switch(setting->type)
  {
    case CONFIG_TYPE_NONE:
      setting->type = CONFIG_TYPE_FLOAT;
      /* fall through */

    case CONFIG_TYPE_FLOAT:
      setting->value.fval = value;
      return(CONFIG_TRUE);

    case CONFIG_TYPE_INT:
      if(config_get_option(setting->config, CONFIG_OPTION_AUTOCONVERT))
      {
        setting->value.ival = (int)value;
        return(CONFIG_TRUE);
      }
      else
        return(CONFIG_FALSE);

    case CONFIG_TYPE_INT64:
      if(config_get_option(setting->config, CONFIG_OPTION_AUTOCONVERT))
      {
        setting->value.llval = (long long)value;
        return(CONFIG_TRUE);
      }
      else
        return(CONFIG_FALSE);

    default:
      return(CONFIG_FALSE);
  }
}

/* ------------------------------------------------------------------------- */

int config_setting_get_bool(const config_setting_t *setting)
{
  return((setting->type == CONFIG_TYPE_BOOL) ? setting->value.ival : 0);
}

/* ------------------------------------------------------------------------- */

int config_setting_set_bool(config_setting_t *setting, int value)
{
  if(setting->type == CONFIG_TYPE_NONE)
    setting->type = CONFIG_TYPE_BOOL;
  else if(setting->type != CONFIG_TYPE_BOOL)
    return(CONFIG_FALSE);

  setting->value.ival = value;
  return(CONFIG_TRUE);
}

/* ------------------------------------------------------------------------- */

const char *config_setting_get_string(const config_setting_t *setting)
{
  return((setting->type == CONFIG_TYPE_STRING) ? setting->value.sval : NULL);
}

/* ------------------------------------------------------------------------- */

int config_setting_set_string(config_setting_t *setting, const char *value)
{
  if(setting->type == CONFIG_TYPE_NONE)
    setting->type = CONFIG_TYPE_STRING;
  else if(setting->type != CONFIG_TYPE_STRING)
    return(CONFIG_FALSE);

  if(setting->value.sval)
    __delete(setting->value.sval);

  setting->value.sval = (value == NULL) ? NULL : strdup(value);
  return(CONFIG_TRUE);
}

/* ------------------------------------------------------------------------- */

int config_setting_set_format(config_setting_t *setting, short format)
{
  if(((setting->type != CONFIG_TYPE_INT)
      && (setting->type != CONFIG_TYPE_INT64))
     || ((format != CONFIG_FORMAT_DEFAULT) && (format != CONFIG_FORMAT_HEX)))
    return(CONFIG_FALSE);

  setting->format = format;

  return(CONFIG_TRUE);
}

/* ------------------------------------------------------------------------- */

short config_setting_get_format(const config_setting_t *setting)
{
  return(setting->format != 0 ? setting->format
         : setting->config->default_format);
}

/* ------------------------------------------------------------------------- */

config_setting_t *config_setting_lookup(config_setting_t *setting,
                                        const char *path)
{
  const char *p = path;
  config_setting_t *found = setting;

  for(;;)
  {
    while(*p && strchr(PATH_TOKENS, *p))
      p++;

    if(! *p)
      break;

    if(*p == '[')
      found = config_setting_get_elem(found, atoi(++p));
    else
      found = config_setting_get_member(found, p);

    if(! found)
      break;

    while(! strchr(PATH_TOKENS, *p))
      p++;
  }

  return(*p || (found == setting) ? NULL : found);
}

/* ------------------------------------------------------------------------- */

config_setting_t *config_lookup(const config_t *config, const char *path)
{
  return(config_setting_lookup(config->root, path));
}

/* ------------------------------------------------------------------------- */

int config_lookup_string(const config_t *config, const char *path,
                         const char **value)
{
  const config_setting_t *s = config_lookup(config, path);
  if(! s)
    return(CONFIG_FALSE);

  if(config_setting_type(s) != CONFIG_TYPE_STRING)
    return(CONFIG_FALSE);

  *value = config_setting_get_string(s);

  return(CONFIG_TRUE);
}

/* ------------------------------------------------------------------------- */

int config_lookup_int(const config_t *config, const char *path,
                      int *value)
{
  const config_setting_t *s = config_lookup(config, path);
  if(! s)
    return(CONFIG_FALSE);

  return(__config_setting_get_int(s, value));
}

/* ------------------------------------------------------------------------- */

int config_lookup_int64(const config_t *config, const char *path,
                        long long *value)
{
  const config_setting_t *s = config_lookup(config, path);
  if(! s)
    return(CONFIG_FALSE);

  return(__config_setting_get_int64(s, value));
}

/* ------------------------------------------------------------------------- */

int config_lookup_float(const config_t *config, const char *path,
                        double *value)
{
  const config_setting_t *s = config_lookup(config, path);
  if(! s)
    return(CONFIG_FALSE);

  return(__config_setting_get_float(s, value));
}

/* ------------------------------------------------------------------------- */

int config_lookup_bool(const config_t *config, const char *path, int *value)
{
  const config_setting_t *s = config_lookup(config, path);
  if(! s)
    return(CONFIG_FALSE);

  if(config_setting_type(s) != CONFIG_TYPE_BOOL)
    return(CONFIG_FALSE);

  *value = config_setting_get_bool(s);
  return(CONFIG_TRUE);
}

/* ------------------------------------------------------------------------- */

int config_setting_get_int_elem(const config_setting_t *setting, int idx)
{
  const config_setting_t *element = config_setting_get_elem(setting, idx);

  return(element ? config_setting_get_int(element) : 0);
}

/* ------------------------------------------------------------------------- */

config_setting_t *config_setting_set_int_elem(config_setting_t *setting,
                                              int idx, int value)
{
  config_setting_t *element = NULL;

  if((setting->type != CONFIG_TYPE_ARRAY)
     && (setting->type != CONFIG_TYPE_LIST))
    return(NULL);

  if(idx < 0)
  {
    if(! __config_list_checktype(setting, CONFIG_TYPE_INT))
      return(NULL);

    element = config_setting_create(setting, NULL, CONFIG_TYPE_INT);
  }
  else
  {
    element = config_setting_get_elem(setting, idx);

    if(! element)
      return(NULL);
  }

  if(! config_setting_set_int(element, value))
    return(NULL);

  return(element);
}

/* ------------------------------------------------------------------------- */

long long config_setting_get_int64_elem(const config_setting_t *setting,
                                        int idx)
{
  const config_setting_t *element = config_setting_get_elem(setting, idx);

  return(element ? config_setting_get_int64(element) : 0);
}

/* ------------------------------------------------------------------------- */

config_setting_t *config_setting_set_int64_elem(config_setting_t *setting,
                                                int idx, long long value)
{
  config_setting_t *element = NULL;

  if((setting->type != CONFIG_TYPE_ARRAY)
     && (setting->type != CONFIG_TYPE_LIST))
    return(NULL);

  if(idx < 0)
  {
    if(! __config_list_checktype(setting, CONFIG_TYPE_INT64))
      return(NULL);

    element = config_setting_create(setting, NULL, CONFIG_TYPE_INT64);
  }
  else
  {
    element = config_setting_get_elem(setting, idx);

    if(! element)
      return(NULL);
  }

  if(! config_setting_set_int64(element, value))
    return(NULL);

  return(element);
}

/* ------------------------------------------------------------------------- */

double config_setting_get_float_elem(const config_setting_t *setting, int idx)
{
  config_setting_t *element = config_setting_get_elem(setting, idx);

  return(element ? config_setting_get_float(element) : 0.0);
}

/* ------------------------------------------------------------------------- */

config_setting_t *config_setting_set_float_elem(config_setting_t *setting,
                                                int idx, double value)
{
  config_setting_t *element = NULL;

  if((setting->type != CONFIG_TYPE_ARRAY)
     && (setting->type != CONFIG_TYPE_LIST))
    return(NULL);

  if(idx < 0)
  {
    if(! __config_list_checktype(setting, CONFIG_TYPE_FLOAT))
      return(NULL);

    element = config_setting_create(setting, NULL, CONFIG_TYPE_FLOAT);
  }
  else
    element = config_setting_get_elem(setting, idx);

  if(! element)
    return(NULL);

  if(! config_setting_set_float(element, value))
    return(NULL);

  return(element);
}

/* ------------------------------------------------------------------------- */

int config_setting_get_bool_elem(const config_setting_t *setting, int idx)
{
  config_setting_t *element = config_setting_get_elem(setting, idx);

  if(! element)
    return(CONFIG_FALSE);

  if(element->type != CONFIG_TYPE_BOOL)
    return(CONFIG_FALSE);

  return(element->value.ival);
}

/* ------------------------------------------------------------------------- */

config_setting_t *config_setting_set_bool_elem(config_setting_t *setting,
                                               int idx, int value)
{
  config_setting_t *element = NULL;

  if((setting->type != CONFIG_TYPE_ARRAY)
     && (setting->type != CONFIG_TYPE_LIST))
    return(NULL);

  if(idx < 0)
  {
    if(! __config_list_checktype(setting, CONFIG_TYPE_BOOL))
      return(NULL);

    element = config_setting_create(setting, NULL, CONFIG_TYPE_BOOL);
  }
  else
    element = config_setting_get_elem(setting, idx);

  if(! element)
    return(NULL);

  if(! config_setting_set_bool(element, value))
    return(NULL);

  return(element);
}

/* ------------------------------------------------------------------------- */

const char *config_setting_get_string_elem(const config_setting_t *setting,
                                           int idx)
{
  config_setting_t *element = config_setting_get_elem(setting, idx);

  if(! element)
    return(NULL);

  if(element->type != CONFIG_TYPE_STRING)
    return(NULL);

  return(element->value.sval);
}

/* ------------------------------------------------------------------------- */

config_setting_t *config_setting_set_string_elem(config_setting_t *setting,
                                                 int idx, const char *value)
{
  config_setting_t *element = NULL;

  if((setting->type != CONFIG_TYPE_ARRAY)
     && (setting->type != CONFIG_TYPE_LIST))
    return(NULL);

  if(idx < 0)
  {
    if(! __config_list_checktype(setting, CONFIG_TYPE_STRING))
      return(NULL);

    element = config_setting_create(setting, NULL, CONFIG_TYPE_STRING);
  }
  else
    element = config_setting_get_elem(setting, idx);

  if(! element)
    return(NULL);

  if(! config_setting_set_string(element, value))
    return(NULL);

  return(element);
}

/* ------------------------------------------------------------------------- */

config_setting_t *config_setting_get_elem(const config_setting_t *setting,
                                          unsigned int idx)
{
  config_list_t *list;

  if(! config_setting_is_aggregate(setting))
    return(NULL);

  list = setting->value.list;
  if(! list)
    return(NULL);

  if(idx >= list->length)
    return(NULL);

  return(list->elements[idx]);
}

/* ------------------------------------------------------------------------- */

config_setting_t *config_setting_get_member(const config_setting_t *setting,
                                            const char *name)
{
  if(setting->type != CONFIG_TYPE_GROUP)
    return(NULL);

  return(__config_list_search(setting->value.list, name, NULL));
}

/* ------------------------------------------------------------------------- */

void config_set_destructor(config_t *config, void (*destructor)(void *))
{
  config->destructor = destructor;
}

/* ------------------------------------------------------------------------- */

void config_set_include_dir(config_t *config, const char *include_dir)
{
  __delete(config->include_dir);
  config->include_dir = strdup(include_dir);
}

/* ------------------------------------------------------------------------- */

void config_set_include_func(config_t *config, config_include_fn_t func)
{
  config->include_fn = func ? func : config_default_include_func;
}

/* ------------------------------------------------------------------------- */

int config_setting_length(const config_setting_t *setting)
{
  if(! config_setting_is_aggregate(setting))
    return(0);

  if(! setting->value.list)
    return(0);

  return(setting->value.list->length);
}

/* ------------------------------------------------------------------------- */

void config_setting_set_hook(config_setting_t *setting, void *hook)
{
  setting->hook = hook;
}

/* ------------------------------------------------------------------------- */

config_setting_t *config_setting_add(config_setting_t *parent,
                                     const char *name, int type)
{
  if((type < CONFIG_TYPE_NONE) || (type > CONFIG_TYPE_LIST))
    return(NULL);

  if(! parent)
    return(NULL);

  if((parent->type == CONFIG_TYPE_ARRAY) && !__config_type_is_scalar(type))
    return(NULL); /* only scalars can be added to arrays */

  if((parent->type == CONFIG_TYPE_ARRAY) || (parent->type == CONFIG_TYPE_LIST))
    name = NULL;

  if(name)
  {
    if(! __config_validate_name(name))
      return(NULL);
  }

  if(config_setting_get_member(parent, name) != NULL)
  {
    if(config_get_option(parent->config, CONFIG_OPTION_ALLOW_OVERRIDES))
      config_setting_remove(parent, name);
    else
      return(NULL); /* already exists */
  }

  return(config_setting_create(parent, name, type));
}

/* ------------------------------------------------------------------------- */

int config_setting_remove(config_setting_t *parent, const char *name)
{
  unsigned int idx;
  config_setting_t *setting;
  const char *settingName;
  const char *lastFound;

  if(! parent)
    return(CONFIG_FALSE);

  if(parent->type != CONFIG_TYPE_GROUP)
    return(CONFIG_FALSE);

  setting = config_setting_lookup(parent, name);
  if(! setting)
    return(CONFIG_FALSE);

  settingName = name;
  do
  {
    lastFound = settingName;
    while(settingName && !strchr(PATH_TOKENS, *settingName))
      ++settingName;

    if(*settingName == '\0')
    {
      settingName = lastFound;
      break;
    }

  }while(*++settingName);

  if(!(setting = __config_list_search(setting->parent->value.list, settingName, &idx)))
    return(CONFIG_FALSE);

  __config_list_remove(setting->parent->value.list, idx);
  __config_setting_destroy(setting);

  return(CONFIG_TRUE);
}

/* ------------------------------------------------------------------------- */

int config_setting_remove_elem(config_setting_t *parent, unsigned int idx)
{
  config_list_t *list;
  config_setting_t *removed = NULL;

  if(! parent)
    return(CONFIG_FALSE);

  if(! config_setting_is_aggregate(parent))
    return(CONFIG_FALSE);

  list = parent->value.list;
  if(! list)
    return(CONFIG_FALSE);

  if(idx >= list->length)
    return(CONFIG_FALSE);

  removed = __config_list_remove(list, idx);
  __config_setting_destroy(removed);

  return(CONFIG_TRUE);
}

/* ------------------------------------------------------------------------- */

int config_setting_index(const config_setting_t *setting)
{
  config_setting_t **found = NULL;
  config_list_t *list;
  int i;

  if(! setting->parent)
    return(-1);

  list = setting->parent->value.list;

  for(i = 0, found = list->elements; i < (int)list->length; ++i, ++found)
  {
    if(*found == setting)
      return(i);
  }

  return(-1);
}

/* ------------------------------------------------------------------------- */

const char **config_default_include_func(config_t *config,
                                         const char *include_dir,
                                         const char *path,
                                         const char **error)
{
  char *file;
  const char **files;

  if(include_dir && IS_RELATIVE_PATH(path))
  {
    file = (char *)malloc(strlen(include_dir) + strlen(path) + 2);
    strcpy(file, include_dir);
    strcat(file, FILE_SEPARATOR);
    strcat(file, path);
  }
  else
    file = strdup(path);

  *error = NULL;

  files = (const char **)malloc(sizeof(char **) * 2);
  files[0] = file;
  files[1] = NULL;

  return(files);
}

/* ------------------------------------------------------------------------- */

int config_setting_is_scalar(const config_setting_t *setting)
{
  return(__config_type_is_scalar(setting->type));
}

/* ------------------------------------------------------------------------- */

int config_setting_is_aggregate(const config_setting_t *setting)
{
  return((setting->type == CONFIG_TYPE_ARRAY)
         || (setting->type == CONFIG_TYPE_LIST)
         || (setting->type == CONFIG_TYPE_GROUP));
}

/* ------------------------------------------------------------------------- */
/* end libconfig/lib/libconfig.c */
#undef BEGIN
#undef CHUNK_SIZE
#undef ECHO
/* end bundled libconfig */