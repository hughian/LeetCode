#include<iostream>
#include<vector>
using namespace std;
typedef long long ll;
ll gcd(ll a,ll b){
    if(b==0) return a;
    return gcd(b,a%b);
}

void rec(pair<ll,ll> &p){
    ll g = gcd(abs(p.first),abs(p.second));
    p.first /= g;
    p.second /= g;
}

pair<ll,ll> add(pair<ll,ll> &a,pair<ll,ll> &b){
    pair<ll,ll> ans = make_pair(a.first*b.second + a.second * b.first,a.second * b.second);
    rec(ans);
    return ans;
}
void show(pair<ll,ll> &p){
    ll g = gcd(abs(p.first),abs(p.second));
    ll k = p.first/p.second;
    bool neg = false;
    if(p.first<0){
        p.first = -p.first;
        neg = true;
    }
    p.first /= g;
    p.second /= g;
    p.first = (p.first%p.second + p.second )%p.second;
    if(k==0){
        if(neg) cout<<"-";
        if(p.first == 0)
            cout<<p.first;
        else
            cout<<p.first<<"/"<<p.second;
    }else{
        if(p.first==0)
            cout<<k;
        else
            cout<<k<<" "<<p.first<<"/"<<p.second;
    }
}
int main(){
    int n;
    cin>>n;
    vector<pair<ll,ll> > vp(n);
    char c;
    for(int i=0;i<n;i++){
        cin>>vp[i].first>>c>>vp[i].second;
    }
    pair<ll,ll> ans = vp[0];
    for(int i=1;i<n;i++)
        ans = add(ans,vp[i]);
    show(ans);
    return 0;
}
