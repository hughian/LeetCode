#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

int main()
{
    long long N,p;
    cin>>N>>p;
    vector<long long> v(N,0); 
    for(long long i=0;i<N;i++){
        cin>>v[i];
    }
	sort(v.begin(),v.end());
    int ans = 0;
    for(int i=0;i<N;i++){
        int tmp = upper_bound(v.begin()+i,v.end(),v[i]*p) - (v.begin()+i);
        ans = max(ans,tmp);
    }
    cout<<ans;
    return 0;
}
