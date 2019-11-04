#include<iostream>
#include<vector>
#include<cmath>
using namespace std;
int isPrime(int x)
{
    for(int i=2;i<=(int)(sqrt(x)+0.5);i++)
        if(x%i==0)
            return 0;
    return 1;
}

int main()
{
    int m,n;
    cin>>m>>n;
    vector<int> p;
    int i=2;
    while((int)p.size()<n){
        if(isPrime(i))
            p.push_back(i);
        i++;
    }
    cout<<p[m-1];
    for(int i=m,k=2;i<(int)p.size();i++,k++)
    {
        if(k==1)
            cout<<p[i];
        else
            cout<<" "<<p[i];
        if(k==10){
            cout<<endl;
            k =0;
        }
    }
    return 0;
}
