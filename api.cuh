#pragma once
























#ifndef __libconfig_h
#define __libconfig_h

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#if defined(LIBCONFIG_STATIC)
#define LIBCONFIG_API
#elif defined(LIBCONFIG_EXPORTS)
#define LIBCONFIG_API __declspec(dllexport)
#else
#define LIBCONFIG_API __declspec(dllimport)
#endif
#else
#define LIBCONFIG_API
#endif

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

#define                    config_get_include_dir(                       C) \
  ((C)->include_dir)

#define            config_set_auto_convert(                 C, F) \
  config_set_option((C), CONFIG_OPTION_AUTOCONVERT, (F))
#define           config_get_auto_convert(                       C) \
  config_get_option((C), CONFIG_OPTION_AUTOCONVERT)

#define           config_setting_type(                               S) \
  ((S)->type)

#define           config_setting_is_group(                               S) \
  ((S)->type == CONFIG_TYPE_GROUP)
#define           config_setting_is_array(                               S) \
  ((S)->type == CONFIG_TYPE_ARRAY)
#define           config_setting_is_list(                               S) \
  ((S)->type == CONFIG_TYPE_LIST)

#define           config_setting_is_number(                               S) \
  (((S)->type == CONFIG_TYPE_INT)                                       \
   || ((S)->type == CONFIG_TYPE_INT64)                                  \
   || ((S)->type == CONFIG_TYPE_FLOAT))

#define                    config_setting_name( \
                                 S)             \
  ((S)->name)

#define                          config_setting_parent( \
                                 S)                     \
  ((S)->parent)

#define           config_setting_is_root(       \
                                 S)             \
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

#define                          config_root_setting( \
                         C)                           \
  ((C)->root)

#define             config_set_default_format(                 C,       \
                                                          F)            \
  (C)->default_format = (F)

#define             config_get_default_format(                 C)       \
  ((C)->default_format)

#define                      config_setting_source_line(   \
                                 S)                        \
  ((S)->line)

#define                  config_setting_source_file(    \
                                 S)                     \
  ((S)->file)

#define                    config_error_text(                       C)  \
  ((C)->error_text)

#define                    config_error_file(                       C)  \
  ((C)->error_file)

#define           config_error_line(                       C)   \
  ((C)->error_line)

#define                      config_error_type(                       C) \
  ((C)->error_type)

#ifdef __cplusplus
}
#endif

#endif
























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

#endif
























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

#endif
























#include <string.h>
#include <sys/types.h>

#define __new(T) (T *)calloc(1, sizeof(T))
#define __delete(P) free((void *)(P))
#define __zero(P) memset((void *)(P), 0, sizeof(*P))

extern long long libconfig_parse_integer(const char *s, int *ok);
extern unsigned long long libconfig_parse_hex64(const char *s);

extern void libconfig_format_double(double val, int precision, int sci_ok,
                                    char *buf, size_t buflen);

























#ifndef __wincompat_h
#define __wincompat_h

#include <limits.h>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) \
  || defined(WIN64) || defined(_WIN64) || defined(__WIN64__)


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

#else

#define INT64_FMT "%lld"
#define UINT64_FMT "%llu"

#define INT64_HEX_FMT "%llX"

#define FILE_SEPARATOR "/"

#endif

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

#else

#define INT64_CONST(I)  (I ## LL)
#define UINT64_CONST(I) (I ## ULL)

#define IS_RELATIVE_PATH(P) \
  ((P)[0] != '/')

#include <unistd.h>

#endif

#endif
























#ifndef __libconfig_scanctx_h
#define __libconfig_scanctx_h

#include <stdio.h>
#include <string.h>
#include <sys/types.h>


#define MAX_INCLUDE_DEPTH 10

struct include_stack_frame
{




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



















extern FILE *libconfig_scanctx_push_include(struct scan_context *ctx,
                                            void *prev_buffer,
                                            const char *path,
                                            const char **error);







extern FILE *libconfig_scanctx_next_include_file(struct scan_context *ctx,
                                                 const char **error);




extern void *libconfig_scanctx_pop_include(struct scan_context *ctx);

#define libconfig_scanctx_append_string(C, S) \
  libconfig_strbuf_append_string(&((C)->string), (S))

#define libconfig_scanctx_append_char(C, X) \
  libconfig_strbuf_append_char(&((C)->string), (X))

extern char *libconfig_scanctx_take_string(struct scan_context *ctx);

extern const char *libconfig_scanctx_current_filename(struct scan_context *ctx);

#endif
























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

#endif


































#ifndef YY_LIBCONFIG_YY_GRAMMAR_H_INCLUDED
# define YY_LIBCONFIG_YY_GRAMMAR_H_INCLUDED



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


#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED

union YYSTYPE
{
#line 76 "grammar.y"

  int ival;
  long long llval;
  double fval;
  char *sval;

#line 105 "grammar.h"
};

typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif



int libconfig_yyparse (void *scanner, struct parse_context *ctx, struct scan_context *scan_ctx);

#endif


#ifndef libconfig_yyHEADER_H
#define libconfig_yyHEADER_H 1
#define libconfig_yyIN_HEADER 1

#line 5 "scanner.h"

#line 7 "scanner.h"

#define  YY_INT_ALIGNED short int



#define FLEX_SCANNER
#define YY_FLEX_MAJOR_VERSION 2
#define YY_FLEX_MINOR_VERSION 6
#define YY_FLEX_SUBMINOR_VERSION 4
#define FLEX_BETA

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




#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>





#ifndef FLEXINT_H
#define FLEXINT_H



#if defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L




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

#endif

#endif




#define yyconst const

#if defined(__GNUC__) && __GNUC__ >= 3
#define yynoreturn __attribute__((__noreturn__))
#else
#define yynoreturn
#endif


#ifndef YY_TYPEDEF_YY_SCANNER_T
#define YY_TYPEDEF_YY_SCANNER_T
typedef void* yyscan_t;
#endif



#define yyin yyg->yyin_r
#define yyout yyg->yyout_r
#define yyextra yyg->yyextra_r
#define yyleng yyg->yyleng_r
#define yytext yyg->yytext_r
#define yylineno (YY_CURRENT_BUFFER_LVALUE->yy_bs_lineno)
#define yycolumn (YY_CURRENT_BUFFER_LVALUE->yy_bs_column)
#define yy_flex_debug yyg->yy_flex_debug_r


#ifndef YY_BUF_SIZE
#ifdef __ia64__




#define YY_BUF_SIZE 32768
#else
#define YY_BUF_SIZE 16384
#endif
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

	char *yy_ch_buf;
	char *yy_buf_pos;




	int yy_buf_size;




	int yy_n_chars;





	int yy_is_our_buffer;






	int yy_is_interactive;





	int yy_at_bol;

    int yy_bs_lineno;
    int yy_bs_column;




	int yy_fill_buffer;

	int yy_buffer_status;

	};
#endif

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

#define libconfig_yywrap(yyscanner) (             1)
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




#include <unistd.h>
#endif

#define YY_EXTRA_TYPE struct scan_context *

int yylex_init (yyscan_t* scanner);

int yylex_init_extra ( YY_EXTRA_TYPE user_defined, yyscan_t* scanner);




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


#ifndef YY_READ_BUF_SIZE
#ifdef __ia64__

#define YY_READ_BUF_SIZE 16384
#else
#define YY_READ_BUF_SIZE 8192
#endif
#endif


#ifndef YY_START_STACK_INCR
#define YY_START_STACK_INCR 25
#endif




#ifndef YY_DECL
#define YY_DECL_IS_OURS 1

extern int yylex \
               (YYSTYPE * yylval_param , yyscan_t yyscanner);

#define YY_DECL int yylex \
               (YYSTYPE * yylval_param , yyscan_t yyscanner)
#endif



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
#endif




#ifndef UDX_BOP_COMMON
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdarg.h>


struct Cbuf {
    enum {SIZ=256};
    char c[SIZ];
};
struct BopData;
typedef int BopStatus;


enum {BopFLOAT, BopDOUBLE, BopINT, BopFASCII, BopIASCII, _BopNTYPES};
typedef int BopType;



BopStatus bop_ini(BopData **d);
BopStatus bop_alloc(BopData *d);
BopStatus bop_fin(BopData *d);



BopStatus bop_set_n(long n, BopData *d);
BopStatus bop_set_vars(int n, const char *vars, BopData *d);
BopStatus bop_set_type(BopType type, BopData *d);
void* bop_get_data(BopData *d);



BopStatus bop_get_n(const BopData *d, long *n);
BopStatus bop_get_nvars(const BopData *d, int *n);
BopStatus bop_get_vars(const BopData *d, Cbuf *vars);
BopStatus bop_get_vars(const BopData *d, const char **vars);
BopStatus bop_get_type(const BopData *d, BopType *type);
const void* bop_get_data(const BopData *d);



BopStatus bop_summary(const BopData *d);
BopStatus bop_concatenate(const int nd, const BopData **dd, BopData *dall);



bool         bop_success(BopStatus status);
const char * bop_report_error_desc(BopStatus status);
char *       bop_report_error_mesg();






#define CBUFSIZE 256
#define CERRSIZE 1024


struct BopData {
    long n;
    int nvars;
    char vars[CBUFSIZE];
    BopType type;
    void *data;
};


extern char bop_error_msg[CERRSIZE];

enum {
    BOP_SUCCESS,
    BOP_BADALLOC,
    BOP_BADFILE,
    BOP_WRONGVAR,
    BOP_MISMATCH,
    BOP_WFORMAT,
    BOP_NULLPTR,
    BOP_WMPISIZE,
    BOP_OVERFLOW,
    _BOP_NERR
};



#define xstr(s) str(s)
#define str(s) #s



BopStatus safe_malloc(size_t sz, void **data);
BopStatus safe_open(const char *fname, const char *mode, FILE **f);

size_t get_bsize(BopType t);

void get_path(const char *full, char *path);
void get_fname_values(const char *fnbop, char *fnval);

const char * type2str(BopType t);
BopType str2type(const char *str);

void report_err(const char *fmt, ...);



enum {SERIAL = -1};
BopStatus read_header(int rank, const char *fname,      char *dfname, BopData *d);




BopStatus bop_write_header(MPI_Comm comm, const char *name, const BopData *d);
BopStatus bop_write_values(MPI_Comm comm, const char *name, const BopData *d);



BopStatus bop_read_header(MPI_Comm comm, const char *hfname, BopData *d, char *dfname);
BopStatus bop_read_values(MPI_Comm comm, const char *dfname, BopData *d);





BopStatus bop_write_header(const char *name, const BopData *d);
BopStatus bop_write_values(const char *name, const BopData *d);



BopStatus bop_read_header(const char *hfname, BopData *d, char *dfname);
BopStatus bop_read_values(const char *dfname, BopData *d);



#define UDX_BOP_COMMON
#endif
