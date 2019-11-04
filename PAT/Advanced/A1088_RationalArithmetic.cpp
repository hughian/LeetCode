#include<iostream>
#include<cstdio>
using namespace std;
int scanf(const char *,...);
int printf(const char *,...);
typedef long long ll;
ll gcd(ll a,ll b){
    if(b==0) return a;
    return gcd(b,a%b);
}

ll rec(pair<ll,ll>&a){
    int k = a.first/a.second;
    bool flag = false;
    if(a.first<0){ 
        a.first = -a.first;
        flag = true;
    }
    ll t = gcd(a.first,a.second);
    a.first = a.first/t;
    a.second = a.second/t;
    a.first = (a.first % a.second + a.second)%a.second;
    if(flag && k==0)
        a.first = -a.first;
    return k;
}

void print(pair<ll,ll> p)
{
    ll k = rec(p);
    if(p.first==0 && k==0)
        printf("%lld",0);
    else if(p.first==0 && k>0)
        printf("%lld",k);
    else if(p.first==0 && k<0)
        printf("(%lld)",k);
    else if(p.first<0 && k==0)
        printf("(%lld/%lld)",p.first,p.second);
    else if(p.first>0 && k==0)
        printf("%lld/%lld",p.first,p.second);
    else if(p.first !=0 && k>0)
        printf("%lld %lld/%lld",k,p.first,p.second);
    else 
        printf("(%lld %lld/%lld)",k,p.first,p.second);
}
int main()
{
    pair<ll,ll> a,b;
    scanf("%lld/%lld %lld/%lld",&a.first,&a.second,&b.first,&b.second);
    pair<ll,ll> sa=a,sb=b;
    
    pair<ll,ll> add = make_pair(a.first*b.second + a.second*b.first,a.second*b.second);
    print(sa);
    printf(" + ");
    print(sb);
    printf(" = ");
    print(add);
    printf("\n");

    pair<ll,ll> sub = make_pair(a.first*b.second-a.second*b.first,a.second*b.second);
    print(sa);
    printf(" - ");
    print(sb);
    printf(" = ");
    print(sub);
    printf("\n");

    pair<ll,ll> mul = make_pair(a.first*b.first,a.second*b.second);
    print(sa);
    printf(" * ");
    print(sb);
    printf(" = ");
    print(mul);
    printf("\n");

    pair<ll,ll> div = make_pair(a.first*b.second,a.second*b.first);
    print(sa);
    printf(" / ");
    print(sb);
    printf(" = ");
    if(div.second==0)
        printf("Inf\n");
    else{
        if(div.second<0){
            div.second = -div.second;
            div.first = -div.first;
        }
        print(div);
        printf("\n");
    }
    return 0;
}
