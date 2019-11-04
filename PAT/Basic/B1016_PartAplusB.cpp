#include<iostream>
using namespace std;
long long countDigits(long long num,int d)
{
    long long  res = 0;
    while(num){
        if(num%10 == d)
            res = res * 10 + d;
        num = num / 10;
    }
    return res;
}

int main()
{
    long long A,B,Da,Db;
    cin>>A>>Da>>B>>Db;
    long long pa = countDigits(A,Da);
    long long pb = countDigits(B,Db);
    long long sum = pa + pb;
    cout<<sum;
    return 0;  
}
