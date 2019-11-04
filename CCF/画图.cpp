#include<bits/stdc++.h>
using namespace std;
bool table[101][101];
int main()
{
	//freopen("in#CCF.txt","r",stdin);
	int n;
	scanf("%d",&n);
	while(n--){
		int a,b,c,d;
		scanf("%d%d%d%d",&a,&b,&c,&d);
		//printf("%d%d%d%d\n",a,b,c,d);
		for(int i=a+1;i<=c;i++)
			for(int j=b+1;j<=d;j++)
				table[i][j]=true;
	}
	int ans=0;
	for(int i=0;i<=100;i++)
		for(int j=0;j<=100;j++)
			if(table[i][j]) ans++;
	
	
	//for(int i=0;i<10;i++,puts(""))
	//	for(int j=0;j<10;j++)
	//		printf("%d ",table[i][j]);		

	printf("%d\n",ans);
	return 0;
}
