#include<iostream>
using namespace std;

int reverse(int x) {
    int a,b=x,c=0;
    a = b % 10;
    b = b / 10;
    int max = 214748364;
    int min = -214748364;
    while(a || b){
        c = c * 10 + a;
        if(c > max && b > 7)
            return 0;
        a = b % 10;
        b = b / 10;
    }
    return c;
}
int main()
{
    int x ;
    cin>>x;
    cout<< reverse(x);
    return 0;
}
