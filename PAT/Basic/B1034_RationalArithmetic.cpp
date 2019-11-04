#include<iostream>
#include<cstdio>
#include<string>
using namespace std;
long long gcd(long long a,long long b){  return b==0?a:gcd(b,a%b);   }

void reduction(long long a,long long b,char res[])
{
    if(b==0){
        sprintf(res,"Inf");
        return;
    }
    int sign = (a*b>=0)?0:1;
    long long ta = a>=0?a:-a,tb = b>=0?b:-b;
    long long tt = gcd(ta,tb);
    ta /= tt;
    tb /= tt;
    long long k = ta/tb;
    ta = ta%tb;
    char tmp[1000];
    if(ta==0)
        sprintf(tmp,"%lld",k);
    else if(k==0)
        sprintf(tmp,"%lld/%lld",ta,tb);
    else
        sprintf(tmp,"%lld %lld/%lld",k,ta,tb);

    if(sign)
        sprintf(res,"(-%s)",tmp);
    else
        sprintf(res,"%s",tmp);
}

int main()
{
    long long a1,a2,b1,b2;
    scanf("%lld/%lld %lld/%lld",&a1,&b1,&a2,&b2);
    char ab1[1000],ab2[1000],ans[1000];
    reduction(a1,b1,ab1);
    reduction(a2,b2,ab2);
    reduction(a1*b2+a2*b1,b1*b2,ans);
    printf("%s + %s = %s\n",ab1,ab2,ans);
    reduction(a1*b2-a2*b1,b1*b2,ans);
    printf("%s - %s = %s\n",ab1,ab2,ans);
    reduction(a1*a2,b1*b2,ans);
    printf("%s * %s = %s\n",ab1,ab2,ans);
    reduction(a1*b2,a2*b1,ans);
    printf("%s / %s = %s\n",ab1,ab2,ans);
    return 0;
}
