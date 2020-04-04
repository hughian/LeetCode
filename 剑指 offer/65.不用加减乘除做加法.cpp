//
// Created by Hughian on 2020/4/4.
//


// 使用位操作
class Solution {
public:
    int Add(int num1, int num2){
        int sum, c;
        do{
            sum = num1 ^ num2; // 异或同位加
            c = (num1 & num2) << 1; // & 得到进位，然后左移一位
            num1 = sum;
            num2 = c;
        }while (num2 != 0);

        return num1;
    }
};