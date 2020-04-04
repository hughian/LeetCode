//
// Created by Hughian on 2020/4/4.
//


// 这题情况太多了，用了 书上的写法
class Solution {
public:
    // 处理无符号数
    bool scanUnsignedInteger(const char ** str){
        const char* before = *str;
        while(**str != 0 && **str >= '0' && **str <= '9')
            ++(*str);
        return *str > before;
    }

    // 处理有符号数
    bool scanInteger(const char ** str){
        if (**str == '+' || **str == '-')
            ++(*str);
        return scanUnsignedInteger(str);
    }

    //判断是否是个数字
    bool isNumeric(const char* string)
    {
        // 格式 A[.[B]][e|EC] 或者 .B[e|EC]
        // 其中 A, C 都是整数（可能有符号），B 是无符号整数
        if (string == nullptr) return false;
        bool numeric = scanInteger(&string);

        if (*string == '.'){
            ++string;
            numeric = scanUnsignedInteger(&string) || numeric; //可以没有整数部分
        }
        if (*string == 'e' || *string == 'E'){
            ++string;
            numeric = numeric && scanInteger(&string);
        }
        return numeric && *string == 0;
    }

};