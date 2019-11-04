#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;
typedef long long ll;
vector<ll> posv,negv,posc,negc;
bool cmp(ll a,ll b){
    return a>b;
}
int main()
{
    ll nc,np,t;
    cin>>nc;
    for(int i=0;i<nc;i++){
        cin>>t;
        if(t>=0) posc.push_back(t);
        else negc.push_back(t);
    }
    cin>>np;
    for(int i=0;i<np;i++){
        cin>>t;
        if(t>=0) posv.push_back(t);
        else   negv.push_back(t);
    }
    sort(posc.begin(),posc.end(),cmp);
    sort(posv.begin(),posv.end(),cmp);
    sort(negc.begin(),negc.end());
    sort(negv.begin(),negv.end());

    unsigned i=0,j=0;
    ll ans = 0;
    while(i<posc.size() && i<posv.size()){
        ans += posc[i] * posv[i];
        i++;
    }
    while(j<negc.size() && j<negv.size()){
        ans += negc[j] * negv[j];
        j++;
    }
    cout<<ans;
    return 0;
}
