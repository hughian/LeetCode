#include<iostream>
#include<string>
#include<vector>
#include<algorithm>
#include<cstdio>
#include<set>
using namespace std;
int scanf(const char*,...);
int printf(const char*,...);
vector< set<int> > vsi(200100);
int getidx(char id[]){ //使用map映射会超时，直接使用名字进行hash
    int idx = 0;
    for(int i=0;i<3;i++)
        idx = idx * 26 + (id[i]-'A');
    idx = idx * 10 + (id[3]-'0');
    return idx;
}
int main()
{
    int N,K;
    scanf("%d%d",&N,&K);
	char buf[10];
    int t,Nt;
    for(int i=1;i<=K;i++){
        scanf("%d%d",&t,&Nt);
        for(int j=0;j<Nt;j++){
            scanf("%s",buf);
            int idx = getidx(buf);
            vsi[idx].insert(t);
        }
    }
	set<int>::iterator it;
    for(int i=1;i<=N;i++){
        scanf("%s",buf);
        int idx = getidx(buf);
        printf("%s %d",buf,vsi[idx].size());
        //sort(vsi[idx].begin(),vsi[idx].end());
        //for(unsigned j=0;j<vsi[idx].size();j++)
        //    printf(" %d",vsi[idx][j]);
        it = vsi[idx].begin();
		for(;it!=vsi[idx].end();it++)
			printf(" %d",*it);
		printf("\n");
    }
    return 0;
}

