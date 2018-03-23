struct KeyList;

void KeyList_ini(KeyList**);
void KeyList_copy(KeyList*, /**/ KeyList**);
void KeyList_fin(KeyList*);

void KeyList_append(KeyList*, const char*);
int KeyList_has(KeyList*, const char*);
int KeyList_offset(KeyList*, const char*);
int KeyList_width(KeyList*, const char*);
int KeyList_size(KeyList*);

void KeyList_mark(KeyList*, const char*);
void KeyList_clear(KeyList*);
int  KeyList_marked(KeyList*, const char*);
void KeyList_log_marked(KeyList*);
