#include<iostream>
#include<cstdio>
using namespace std;

int main()
{
    int n;
    double sum = 0.0,t;
    cin>>n;
    for(int i=0;i<n;i++){
        cin>>t;
        sum += t * (n-i)*(i+1);
    }
    printf("%.2f",sum);
    return 0;
}
