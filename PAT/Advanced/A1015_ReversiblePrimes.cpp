#include<iostream>
#include<cmath>
using namespace std;

int radix(int n,int d)
{
    int ans = 0;
    while(n){
        ans = ans * d + n%d;
        n = n/d;
    }
    return ans;
}

bool isPrime(int x){
    if(x==1)
        return false;
    if(x==2)
        return true;
    for(int i=2;i<sqrt(x)+1;i++){
        if(x%i==0) return false;
    }
    return true;
}

int main()
{
    int n,d;
    while(cin>>n,n>0){
        cin>>d;
        int x = radix(n,d);
        if(isPrime(x) && isPrime(n))
            cout<<"Yes"<<endl;
        else
            cout<<"No"<<endl;
    }
    return 0;
}
