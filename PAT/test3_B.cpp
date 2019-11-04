#include<iostream>
#include<vector>
using namespace std;
typedef long long ll;
ll missing(vector<ll> &v){
    int left =0 ;
    int right = v.size();
    while(left<right){
        if(v[left] == left+1)
            left++;
        else if(v[left] < left+1 || v[left] > right || v[v[left]-1]==v[left]){
            right--;
            v[left] = v[right];
        }else{
            swap(v[left],v[v[left]-1]);
        }
    }
    return left+1;
}
//time limit....
int main()
{
    ll n;
    cin>>n;
    vector<ll> vec(n,0);
    for(ll i=0;i<n;i++){
        scanf("%lld",&vec[i]);
    }
    cout<<missing(vec);
    return 0;
}
