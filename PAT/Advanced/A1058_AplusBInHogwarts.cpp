#include<iostream>
using namespace std;

int main()
{
    int a,b,c;
    int d,e,f;
    int ansa,ansb,ansc;
    char ch;
    cin>>a>>ch>>b>>ch>>c>>d>>ch>>e>>ch>>f;
    int tmp = c + f;
    ansc = tmp % 29;
    tmp = tmp / 29;
    tmp = tmp + b + e;
    ansb = tmp % 17;
    tmp = tmp / 17;
    ansa = tmp + a + d;
    cout<<ansa<<ch<<ansb<<ch<<ansc;
    return 0;
}
