#include<iostream>
#include<vector>
using namespace std;
typedef long long ll;
vector<ll> nt(1010,-1);
const ll mod = 1000000007;
ll getNt(ll n){
	if(nt[n]!=-1){
		return nt[n];
	}else{
		ll sum = 0;
		for(int i=0;i<n;i++){
			sum = (sum+getNt(i) * getNt(n-i-1) %1000000007) %1000000007;
		}
		nt[n] = sum;
		return sum;
	}
	
}
ll extgcd(int a,int b,int &x,int &y)
{
	if(!b){
		x=1;y=0;
		return a;
	}else{
		extgcd(b,a%b,y,x);
		y -= (a/b)*x;
	}
	
}
vector<ll> cat(1010,-1);
ll catalan(){
	cat[0]=1;
	cat[1]=1;
	int x,y;
	for(ll i=2;i<1010;i++){
		cat[i] = (4*i-2)%mod;
		cat[i] = (cat[i]*cat[i-1])%mod;
		extgcd(i+1,mod,x,y);
		cat[i] = (cat[i]*(x+mod)%mod)%mod;
	}
}

int main()
{
    ll  f,n;
    cin>>f>>n;
	nt[0] = 1;
	catalan();
    cout<<cat[n];
	return 0;
}
