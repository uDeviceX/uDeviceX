#include <stdio.h>
#include <string.h>
#include "bop_common.h"
#include "bop_serial.h"

#include "pybop.h"

using namespace std;

#define BPC(ans) do {                           \
        BopStatus s = (ans);                    \
        if (!bop_success(s)) {                  \
            fprintf(stderr, ":%s:%d: %s\n%s\n", \
                    __FILE__, __LINE__,         \
                    bop_report_error_desc(s),   \
                    bop_report_error_mesg());   \
            exit(1);                            \
        }} while (0)

PyBop::PyBop()
    : d(nullptr)
{
    BPC(bop_ini(&d));
}

PyBop::~PyBop() {
    BPC(bop_fin(d));
}

void PyBop::reset() {
    BPC(bop_fin(d));
    BPC(bop_ini(&d));
}

void PyBop::alloc() {
    BPC(bop_alloc(d));
}

void PyBop::set_n(long n) {
    BPC(bop_set_n(n, d));
}

void PyBop::set_vars(int n, const string &vars) {
    BPC(bop_set_vars(n, vars.c_str(), d));
}

static const string type_str[] = {
    "float",
    "asciifloat",
    "double",
    "int",
    "asciiint"
};

static const BopType type_key[] = {
    BopFLOAT, BopFASCII, BopDOUBLE, BopINT, BopIASCII
};

void PyBop::set_type(const std::string &type) {
    BopType t;
    int found, i, n;
    n = sizeof(type_str) / sizeof(type_str[0]);
    
    for (found = i = 0; i < n; ++i) {
        if (type == type_str[i]) {
            t = type_key[i];
            found = 1;
            break;
        }
    }
            
    if (!found) {
        fprintf(stderr, "Wrong type given: %s\n", type.c_str());
        exit(1);
    }
    BPC(bop_set_type(t, d));
}

template <typename T>
static void set_data(const vector<T> &src, BopData *d) {
    T *dst = (T*) bop_get_data(d);
    memcpy(dst, src.data(), src.size() * sizeof(T));
}

void PyBop::set_datai(const std::vector<int> &data) {
    set_data(data, d);
}

void PyBop::set_dataf(const std::vector<float> &data) {
    set_data(data, d);
}

void PyBop::set_datad(const std::vector<double> &data) {
    set_data(data, d);
}

long PyBop::get_n() {
    long n;
    BPC(bop_get_n(d, &n));
    return n;
}

string PyBop::get_vars() {
    const char *cvars = nullptr;
    BPC(bop_get_vars(d, &cvars));
    
    return string(cvars);
}

string PyBop::get_type() {
    BopType t;
    BPC(bop_get_type(d, &t));
    return type_str[t];
}

template <typename T>
static vector<T> get_data(const BopData *d) {
    long n;
    BPC(bop_get_n(d, &n));
    const T *src = (const T*) bop_get_data(d);
    vector<T> arr(n);
    memcpy(arr.data(), src, n * sizeof(T));
    return arr;
}

vector<int> PyBop::get_datai() {
    return get_data<int>(d);
}

vector<float> PyBop::get_dataf() {
    return get_data<float>(d);
}

vector<double> PyBop::get_datad() {
    return get_data<double>(d);
}

void PyBop::write(string basename) {
    BPC(bop_write_header(basename.c_str(), d));
    BPC(bop_write_values(basename.c_str(), d));
}

void PyBop::read(std::string hname) {
    char dfname[FILENAME_MAX];
    BPC(bop_read_header(hname.c_str(), d, dfname));
    this->alloc();
    BPC(bop_read_values(dfname, d));
}
