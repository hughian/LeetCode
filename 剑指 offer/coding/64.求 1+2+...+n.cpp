//
// Created by Hughian on 2020/4/4.
//


// 不能使用乘除法，for, while, if else, switch, case 及条件判断语句
class Temp{
    static int n, sum;
public:
    Temp(){n++; sum+=n;}
    static void reset(){n = 0;sum=0;}
    static int get(){ return sum;}
};

int Temp::n = 0;
int Temp::sum = 0;

class Solution {
public:
    int Sum_Solution(int n) {
        Temp::reset();
        Temp *a = new Temp[n];
        delete []a;
        a = nullptr;
        return Temp::get();
    }
};