//
// Created by Hughian on 2020/4/4.
//


// 快速幂的方法，唯一的坑是注意指数的正负号。
class Solution {
public:
    double Power(double base, int exponent) {
        // 快速幂
        double b = base;
        int sign = exponent >= 0?1:-1;
        int e = sign * exponent;

        double res = 1.0;
        while (e){
            if (e % 2){
                res *= b;
            }
            b *= b;
            e /= 2;
        }
        if (sign == -1)
            return 1/res;
        return res;
    }
};