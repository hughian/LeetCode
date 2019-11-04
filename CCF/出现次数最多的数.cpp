#include<bits/stdc++.h>
#define maxv 10001
using namespace std;
int s[maxv];
int main()
{
	int n,x,maxShow=0;
	scanf("%d",&n);
	for(int i=1;i<=n;i++){
		scanf("%d",&x);
		s[x]++;
		maxShow=max(maxShow,s[x]);
	}
	int minVal=10001;
	for(int i=1;i<=10000;i++)
		if(s[i]==maxShow)
			minVal=min(minVal,i);
	printf("%d\n",minVal);
	return 0;
}
