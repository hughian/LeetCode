#include<iostream>
#include<map>
#include<cstdio>
using namespace std;
map<int,int> tmp;
int M,N;

int main()
{
    cin>>M>>N;
    int t;
    int max = 0,ans = 0;
    for(int i=0;i<N;i++){
        for(int j=0;j<M;j++){
            scanf("%d",&t); //cin»á³¬Ê±¡£
            tmp[t]++;
            if(tmp[t]>max){
                max = tmp[t];
                ans = t;
            }
        }
    }
    cout<<ans;
    return 0;
}
