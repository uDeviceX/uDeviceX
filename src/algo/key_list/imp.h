struct KeyList;

void key_list_ini(KeyList**);
void key_list_copy(KeyList*, /**/ KeyList**);
void key_list_fin(KeyList*);

void key_list_append(KeyList*, const char*);
int key_list_has(KeyList*, const char*);
int key_list_offset(KeyList*, const char*);
int key_list_width(KeyList*, const char*);
int key_list_size(KeyList*);

void key_list_mark(KeyList*, const char*);
int  key_list_marked(KeyList*);
void key_list_unmark(KeyList*);

void key_list_log(KeyList*);
