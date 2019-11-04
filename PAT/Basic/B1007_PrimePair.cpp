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
    int n;
    cin>>n;
    vector<int> res;
    for(int i=2;i<=n;i++){
        if(isPrime(i))
            res.push_back(i);
    }
    int sum = 0;
    for(int i=1;i<(int)res.size();i++){
        if(res[i]-res[i-1]==2)
            sum++;
    }
    cout<<sum;
    return 0;
}
