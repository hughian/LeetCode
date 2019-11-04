#include<iostream>
#include<vector>
#include<set>
#include<map>
#include<string>
#include<cstdio>
#include<algorithm>
using namespace std;

map<int,int> id2index;
struct Person{
    int id;
    int estate,area;
    Person():id(-1),estate(0),area(0){}
}info[10011]; //一开始数组开小了，最后一组数据段错误
vector< vector<int> > edge(10011);
struct Family{
    int mem_num;
    int mem_min;
    int total;
    int area;
    float ave_sets;
    float ave_area;
    Family()
        :mem_num(0),mem_min(10000),total(0),area(0),ave_sets(0.0),ave_area(0.0){}
};
int cnt = 0;
int getidx(int id){
    if(id2index.count(id)==0)
        id2index[id] = cnt++;
    return id2index[id];
}
int getid(int index){
    return info[index].id;
}
vector<bool> visited(10011,false);
void dfs(int x,Family &f)
{
    f.mem_num++;
    if(f.mem_min > getid(x))
        f.mem_min = getid(x);
    f.total += info[x].estate;
    f.area += info[x].area;
    visited[x] = true;
    for(int i=0;i<(int)edge[x].size();i++){
        if(visited[ edge[x][i] ]==false)
            dfs(edge[x][i],f);
    }
}
bool mycmp(Family& a,Family &b){
    if(a.ave_area == b.ave_area)
        return a.mem_min < b.mem_min;
    return a.ave_area > b.ave_area;
}
int main()
{
    int n;
    cin>>n;
    int id,k,tmp;
    int fa,mo;
    for(int i=0;i<n;i++){
        cin>>id;
        int idx = getidx(id);
        info[idx].id = id;
        cin>>fa>>mo>>k;
        if(fa != -1 && mo != -1){
            int fai = getidx(fa);
            info[fai].id = fa;
            int moi = getidx(mo);
            info[moi].id = mo;
            edge[fai].push_back(moi);
            edge[moi].push_back(fai);
            //
            edge[idx].push_back(fai);
            edge[fai].push_back(idx);
            //
            edge[idx].push_back(moi);
            edge[moi].push_back(idx);
        }else if(fa == -1 && mo!=-1){
            int moi = getidx(mo);
            info[moi].id=mo;
            edge[idx].push_back(moi);
            edge[moi].push_back(idx);
        }else if(fa!=-1 && mo == -1){
            int fai = getidx(fa);
            info[fai].id = fa;
            edge[fai].push_back(idx);
            edge[idx].push_back(fai);
        }else{;}
        for(int i=1;i<=k;i++){
            cin>>tmp;
            int ti = getidx(tmp);
            info[ti].id = tmp;
            edge[idx].push_back(ti);
            edge[ti].push_back(idx);
        }
        cin>>info[idx].estate>>info[idx].area;
    }
    /*
    for(map<int,int>::iterator it = id2index.begin();it!=id2index.end();it++)
        printf("%04d:\t%d\n",it->first,it->second);
    cout<<"+++++++++++++++++++++++++++++++\n";
    for(int i=0;i<cnt;i++){
        printf("%d:\t%04d\n",i,info[i].id);
    }
    cout<<"-------------------------------\n";
    */
    vector<Family> ans;
    for(int i=0;i<cnt;i++){
        if(!visited[i]){
            Family f;
            dfs(i,f);
            f.ave_sets =(float) f.total / f.mem_num;
            f.ave_area = (float) f.area / f.mem_num; 
            ans.push_back(f);
        }
    }
    sort(ans.begin(),ans.end(),mycmp);
    cout<<ans.size()<<endl;
    for(unsigned i=0;i<ans.size();i++){
        int n = ans[i].mem_num;
        printf("%04d %d %.3f %.3f\n",ans[i].mem_min,n,ans[i].ave_sets,ans[i].ave_area);
    }
    return 0;
}
