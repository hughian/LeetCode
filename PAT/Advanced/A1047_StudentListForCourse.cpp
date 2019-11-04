#include<iostream>
#include<string>
#include<vector>
#include<set>
#include<map>
#include<cstdio>
#include<cstring>
#include<algorithm>
using namespace std;
int scanf(const char*,...);
int printf(const char *,...);

vector<string> names(40010);
vector< vector<int> > course(2510);
bool cmp(int a,int b)
{
    return names[a]<names[b];//strcmp(names[a],names[b])<0;
}
int main()
{
    int N,K;
    scanf("%d%d",&N,&K);
    int C,t;
    char buf[10];
    for(int i=1;i<=N;i++){
        scanf("%s %d",buf,&C);
        names[i] = string(buf);
        for(int j=0;j<C;j++){
            scanf("%d",&t);
            course[t].push_back(i);
        }
    }
    for(int i=1;i<=K;i++){
        int len = course[i].size();
        printf("%d %d\n",i,len);
        sort(course[i].begin(),course[i].end(),cmp);
        for(int j=0;j<len;j++){
                printf("%s\n",names[course[i][j]].c_str());
        }
    }
    return 0;
}
