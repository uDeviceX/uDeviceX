/*

var sum = 0.0
    var c = 0.0
    for i = 1 to input.length do
        var y = input[i] - c
        var t = sum + y
        c = (t - sum) - y
        sum = t
    next i
    return sum

 */

struct KahanSum {
    double c, y, sum;
};
