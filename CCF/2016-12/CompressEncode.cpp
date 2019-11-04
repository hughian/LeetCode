#include<cstdio>  
#include<cstring>  
#include<iostream>  
#include<algorithm>  
#include<string>  
#include<queue>  
#define ll long long  
using namespace std;  
const int inf=0x08080808;  
int mmin(int a,int b){  
    return a<b?a:b;  
}  
int num[1010],dp[1010][1010],sum[1010];  
int p[1010][1010];  
int main(){  
    int n;  
    //while(scanf("%d",&n)){  
    scanf("%d",&n);  
    for(int i=1;i<=n;++i)  
        scanf("%d",&num[i]);  
    sum[0]=0;  
    memset(dp,inf,sizeof(dp));  
    for(int i=1;i<=n;++i){  
        sum[i]=sum[i-1]+num[i];  
        dp[i][i]=0;  
        p[i][i]=i;  
    }  
    for(int len=2;len<=n;++len)  
    for(int i=1;i+len-1<=n;++i){  
        int j=i+len-1;  
        for(int k=p[i][j-1];k<=p[i+1][j];++k){  
            int val=dp[i][k]+dp[k+1][j]+sum[j]-sum[i-1];  
            if(dp[i][j]>val) {  
                dp[i][j]=val;  
                p[i][j]=k;  
            }  
        }  
        //cout<<i<<"  "<<j<<"  "<<dp[i][j]<<endl;  
    }  
    printf("%d\n",dp[1][n]);  
   // }  
    return 0;  
}  