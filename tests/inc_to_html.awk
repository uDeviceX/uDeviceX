# syntax: #include test_data/test.html

$1 ~ /^#include/ {
    system("cat " $2); next
}

{
    print
}
