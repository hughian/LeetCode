#include<iostream>
#include<vector>
#include<cmath>
using namespace std;
//10007 is prime
vector<bool> used(10010,false);
bool isPrime(int x){
    if(x<=1)
        return false;
    for(int i=2;i<(int)(sqrt(x)+1);i++){
        if(x%i == 0)
            return false;
    }
    return true;
}

int main()
{
    int msize,n;
    cin>>msize>>n;
    while(!isPrime(msize)) msize++;
    int t;
    cin>>t;
    cout<<t%msize;
    used[t%msize]=true;
    
    for(int i=1;i<n;i++){
        cin>>t;
        int addr = t % msize;
        bool flag = false;
        for(int j=0;j<msize;j++){ //quadratic probing hi = (h(key) + i*i)%size; 0=<i<=size-1
            int naddr = (addr + j*j)%msize;
            if(!used[naddr]){
                used[naddr] = true;
                cout<<" "<<naddr;
                flag = true;
                break;
            }
        }
        if(!flag)
            cout<<" -";
    }
    return 0;
}
