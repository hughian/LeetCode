#include<bits/stdc++.h>
#define maxn 5005
using namespace std;

struct Opt{
	char op;
	int s,no;
	double p;	
	Opt(double p=0.0,int s=0,int no=0):p(p),s(s),no(no){}
};
bool lessp(const Opt &a,const Opt &b){return a.p<b.p;}
bool morep(const Opt &a,const Opt &b){return a.p>b.p;}

Opt input[maxn];
vector<Opt> buy,sell;
vector<double> price;

void SOLVE()
{
	double pmax=0;
	long long smax=0;
	int bsize=buy.size(),ssize=sell.size(),psize=price.size();
	
	for(int i=0;i<psize;i++){
		if(i!=psize-1&&price[i]>=price[i+1]) continue;
		double p0=price[i];
		long long s1=0,s2=0;
		for(int j=0;j<ssize&&sell[j].p<=p0;j++) s1+=sell[j].s;
		for(int j=0;j<bsize&&p0<=buy[j].p;j++) s2+=buy[j].s;
		long long s0=min(s1,s2);
		if(smax<=s0) smax=s0,pmax=p0;
	}
	printf("%.2f %lld\n",pmax,smax);
}

void INPUT()
{
	int cnt=0;
	char op[10];
	while(~scanf("%s",op)){
		++cnt;
		if(op[0]=='b'||op[0]=='s'){
			input[cnt].op=op[0];
			scanf("%lf%d",&input[cnt].p,&input[cnt].s);
		}else{//cancel
			int del;
			scanf("%d",&del);
			input[del].op=0;
		}
	}
	
	for(int i=1;i<=cnt;i++)
		if(input[i].op=='b') buy.push_back(Opt(input[i].p,input[i].s,0)),price.push_back(input[i].p);
		else if(input[i].op=='s') sell.push_back(Opt(input[i].p,input[i].s,0)),price.push_back(input[i].p);
		
	sort(price.begin(),price.end());
	sort(buy.begin(),buy.end(),morep);
	sort(sell.begin(),sell.end(),lessp);
}

void MAIN()
{
	INPUT();
	SOLVE();
}

int main()
{
	//freopen("in#CCF.txt","r",stdin);
	//freopen("out#CCF.txt","w",stdout);
	MAIN();
	return 0;
}
/*
sell  8.92 400

sell 9.0 1000

buy 8.88 175

buy 8.95 400

buy 100.0 50

错误的结果为8.92 400；

正确的结果为8.95 400；
*/
