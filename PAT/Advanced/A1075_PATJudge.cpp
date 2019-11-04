#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;
int printf(const char *,...);
vector<int> p;
int n,k,m;
struct Usr{
    int id;
    int rank;
    int total;
    int perfect;
    bool valid;
    vector<int> ps;
    vector<bool> vb;
    Usr():id(0),total(-1),perfect(0),valid(false),ps(vector<int>(6,-1)),vb(vector<bool>(6,false)){}
    void calc(){
        int sum = 0;
        bool flag = false;
        for(int i=1;i<=k;i++){
            if(vb[i]) valid = true;
            if(ps[i]!=-1){
                flag = true;
                sum += ps[i];
            }
            if(ps[i]==p[i])
                perfect++;
        }
        if(flag)
            total = sum;
    }
    bool operator < (const Usr &u){
        return this->total > u.total;
    }
};
vector<Usr> vec(10001);
bool cmp(Usr &a,Usr &b){
    if(a.rank == b.rank){
        if(a.perfect == b.perfect)
            return a.id<b.id;
        return a.perfect > b.perfect;
    }
    return a.rank < b.rank;
}
int main()
{
    cin>>n>>k>>m;
    p.resize(k+1);
    for(int i=1;i<=k;i++)
        cin>>p[i];
    int id,pid,sc;
    for(int i=0;i<m;i++){
        cin>>id>>pid>>sc;
        vec[id].id = id;
        if(sc != -1)
            vec[id].vb[pid] = true;
        if(vec[id].ps[pid] == -1)
            vec[id].ps[pid] = 0;
        if(sc > vec[id].ps[pid])
            vec[id].ps[pid]=sc;
    }
    for(int i=1;i<=n;i++){
        vec[i].calc();
    }
    sort(vec.begin()+1,vec.begin()+n+1);
    
    vec[1].rank = 1;
    for(int i=2;i<=n;i++){
        vec[i].rank =vec[i-1].rank;
        if(vec[i].total < vec[i-1].total)
            vec[i].rank = i;
    }
    sort(vec.begin()+1,vec.begin()+n+1,cmp);

    for(int i=1;i<=n;i++){
        if(vec[i].valid){ //得分为0分要输出。全部没有提交通过的才不输出
            printf("%d %05d %d",vec[i].rank,vec[i].id,vec[i].total);
            for(int j=1;j<=k;j++){
                if(vec[i].ps[j]==-1)
                    printf(" -");
                else
                    printf(" %d",vec[i].ps[j]);
            }
            printf("\n");
        }
    }
    return 0;
}
