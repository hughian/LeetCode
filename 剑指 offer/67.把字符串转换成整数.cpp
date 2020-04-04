//
// Created by Hughian on 2020/4/4.
//


// 将一个字符串转换成一个整数，字符串包括数字字母符号,可以为空
// 非法的数字都返回 0
class Solution {
public:
    int helper(int sign, string s){
        long long res = 0;
        for(auto i=0;i<s.size();i++){
            if (s[i] >= '0' && s[i] <= '9')
                res = res * 10 + s[i] - '0';
            else
                return 0;
        }
        if (sign * res < INT_MIN || sign * res > INT_MAX)
            return 0;
        return res * sign;
    }
    int StrToInt(string str) {
        // 空字符串
        if (str.size() == 0)
            return 0;
        // 处理正负号
        int sign = 1;
        if (str[0] == '+' || str[0] == '-'){
            sign = (str[0] == '-')?-1:1;
            return helper(sign, str.substr(1, str.size()-1));
        }
        return helper(1, str);
    }
};