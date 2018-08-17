#include<iostream>
using namespace std;

const int INT_MAX = 2147483647;
const int INT_MIN = -2147483647-1;
class Solution{
public:
    int myAtoi(string str){
        int sign=1;
        int base=0;
        int i=0;
        while(str[i]==' '){i++;}
        if(str[i] == '-' || str[i]=='+'){
            sign = 1 - 2 * (str[i++] == '-');
        }
        while(str[i] >= '0' && str[i] <= '9'){
            if(base > INT_MAX/10 || (base == INT_MAX/10 && str[i]- '0'>7)){
                if(sign == 1)
                    return INT_MAX;
                else
                    return INT_MIN;
            }
            base = 10 * base + (str[i++] - '0');
        }
        return base * sign;
    }
};


int main()
{
    string s="2147483649";
    Solution a;
    cout<<a.myAtoi(s)<<endl;
    return 0;
}
