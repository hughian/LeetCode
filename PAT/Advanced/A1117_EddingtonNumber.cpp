#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;
bool cmp(int a,int b){
    return a>b;
}

int main()
{
    int n;
    cin>>n;
    vector<int> v(n+1,0);
    for(int i=1;i<=n;i++)
        cin>>v[i];
    sort(v.begin()+1,v.end(),cmp);
    int i=1;
    for(;i<=n;i++)
        if(i>=v[i])
            break;
    cout<<i-1;
}
