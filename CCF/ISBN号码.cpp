#include<bits/stdc++.h>
using namespace std;
int main()
{
//	freopen("in#CCF.txt","r",stdin);
	int a[11],ret;
	char c;
	scanf(" %d-%d-%d-",&a[1],&a[2],&a[5]);
	scanf(" %c",&c);
	int ans=0;	
	a[4]=a[2]%10;
	a[3]=a[2]/10%10;
	a[2]=a[2]/100;
	a[9]=a[5]%10;
	a[8]=a[5]/10%10;
	a[7]=a[5]/100%10;
	a[6]=a[5]/1000%10;
	a[5]=a[5]/10000;
	for(int i=1;i<=9;i++) ans+=a[i]*i;
	ans%=11;
	ret=(c=='X')?10:(c-'0');
	if(ans==ret) printf("Right\n");
	else printf("%d-%d%d%d-%d%d%d%d%d-%c",a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],(ans==10)?'X':(ans+'0'));
	return 0;
}
