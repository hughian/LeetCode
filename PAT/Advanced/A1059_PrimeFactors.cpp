#include<iostream>
#include<cstdio>
#include<vector>
#include<map>
#include<cmath>
using namespace std;
typedef long long ll;
vector<ll> prime;
vector<bool> mark(10000,true); 
map<ll,ll> mp;
void init(){
	mark[0] = false;
	mark[1] = false;
	int i,j;
	for(i=2;i<10000;i++){
		if(mark[i]){
			prime.push_back(i);
			for(j=i*i;j<10000;j+=i)
				mark[j] = false;
		}
	}
}

bool isPrime(ll x){
	if(x==1) return false;
	if(x==2) return true;
	ll i;
	for(i=2;i<sqrt(x)+1;i++){
		if(x%i==0) return false;
	}
	return true;
}

int main()
{
	ll N;
	scanf("%lld",&N);
	ll tmp = N;
	init();
	ll pos = 0;
	if(N==1) printf("1=1");
	else{
		while(!isPrime(N) && N > 0){
			while(pos<(int)prime.size()){
				if(N%prime[pos]==0){
					mp[prime[pos]] ++;
					N = N/prime[pos];
					break;
				}else{
					pos++;
				}
			}
		}
		if(N>0)
			mp[N]++;
		map<ll,ll>::iterator it = mp.begin();
		int i = 0,cnt = mp.size();
		printf("%lld=",tmp);
		for(;it!=mp.end();it++){
			printf("%lld",it->first);
			if(it->second!=1)
				printf("^%lld",it->second);
			if(i<cnt-1) cout<<"*";
			i++;
		}
	}
}
