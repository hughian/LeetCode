#include<bits/stdc++.h>
#define maxn 150
using namespace std;
char p[maxn],s[maxn];
int next[maxn],n,sen;

inline bool equal(char a,char b){
	if(sen) return a==b;
	else return a==b||(abs(a-b)==32);
}

int KMP(){
	int i=0,j=0;
	int slen=strlen(s),plen=strlen(p);
	while(i<slen&&j<plen){
		if(j==-1||equal(s[i],p[j])) i++,j++;
		else j=next[j];
	}
	if(j==plen) return i-j;
	else return -1;
}

void GetNext(){
	int plen=strlen(p);
	next[0]=-1;
	int j=0,k=-1;
	while(j<plen-1){
		if(k==-1||equal(p[j],p[k])) ++j,++k,next[j]=k;
		else k=next[k];
	}
}

void SOLVE()
{
	while(n--){
		scanf("%s",s);
		if(KMP()!=-1) printf("%s\n",s);
	}
}

void INPUT()
{
	scanf("%s",p);
	scanf("%d%d",&sen,&n);
	GetNext();
}

void MAIN()
{
	INPUT();
	SOLVE();
}

int main()
{
	//freopen("in#CCF.txt","r",stdin);
	//freopen("out#CFF.txt","w",stdout);
	MAIN();
	return 0;
}
