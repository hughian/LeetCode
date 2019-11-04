#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;
vector<int> num(1010,0); 
vector<int> ufs(1010,-1); //并查集
vector<int> hob(1010,0);
bool mycmp(int a,int b){
    return a>b;
}
int findf(int x){
    int r = x;
    while(ufs[r] != -1){
        r = ufs[r];
    }
    /*
    int f = x; //路径压缩。不使用并不影响AC
    while(ufs[f]!=-1){
        int t = ufs[f];
        ufs[f] = r;
        f = t;
    }*/
    return r;
}

void ufs_union(int a,int b)
{
    int fa = findf(a);
    int fb = findf(b);
    if(fa != fb)
        ufs[fa] = fb;
}

int main()
{
    int n;
    cin>>n;
    int k,tmp;
    char c;
    for(int i=1;i<=n;i++){
        cin>>k>>c;
        for(int j=0;j<k;j++){
            cin>>tmp;
            if(hob[tmp]==0)
                hob[tmp]=i;
            else
                ufs_union(i,findf(hob[tmp]));
        }
    }
    for(int i=1;i<=n;i++){
        num[findf(i)]++;
    }
    int sum = 0;
    for(int i=1;i<=n;i++){
        if(num[i]!=0) sum++;
    }
    sort(num.begin()+1,num.begin()+1+n,mycmp); //排序要从1 到 n
    cout<<sum<<endl;
    for(int i=1;i<=sum;i++)
        cout<<num[i]<<(i<sum?" ":"");
    return 0;
}
