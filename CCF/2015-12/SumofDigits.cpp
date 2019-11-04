#include<iostream>
using namespace std;
int main(void)
{
    int n,tmp;
    int sum = 0;
    cin>>n;
    while(n){
        sum += n%10;
        n = n / 10;
    }
    cout<<sum;
    return 0;
}
