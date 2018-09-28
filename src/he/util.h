/* string equal? */
int util_eq(const char *a, const char *b);

/* fgets without '\n' */
char *util_fgets(char *s, FILE *stream);

/* fgets without '\n' and skipping '#comments' */
char *util_comment_fgets(char *s, FILE *stream);
