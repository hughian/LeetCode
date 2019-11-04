#include<iostream>
#include<vector>
#include<string>
#include<map>
using namespace std;
const int Inf = 1e2+1;
struct Node{
    int id;
    float min,avg;
    Node(int i,float m,float a):id(i),min(m),avg(a){}
};
vector< vector<int> >  cost(1020,vector<int>(1020,Inf));
int N,M,K,Ds;
int toNum(string str){
    if(str[0]=='G'){
        string s = str.substr(1,str.length()-1);
        return stoi(s)+1000;
    }
    else
        return stoi(str);
}

void print(vector<int>&v){
    for(int i=1;i<=N;i++){
        cout<<v[i]<<"\t";
    }
    for(int i=1001;i<=1000+M;i++)
        cout<<v[i]<<"\t";
    cout<<endl;
}

void shortPath(int v,vector<int>& dist){
    vector<bool> used(1020,false);
    int u,min;
    for(int i=1;i<1020;i++){
        dist[i] = cost[v][i];
    }
    dist[v] = 0;
    used[v] = true;
    
    for(int i=2;i<1020;i++){
        u = -1;
        min = Inf;
        for(int j=1;j<1020;j++){ //结点搜索时包括gas station 的候选结点
            if(!used[j] && dist[j] < min){
                u = j;
                min = dist[j];
            }
        }
        if(u==-1) return ;
        used[u] = true;
        for(int j=1;j<1020;j++){
            if(!used[j] && dist[j] > dist[u] + cost[u][j]){
                dist[j] = dist[u]+cost[u][j];
            }
        }
    }

}
int main()
{
    cin>>N>>M>>K>>Ds;
    string a,b;
    int c;
    for(int i=0;i<K;i++){  //循环的上界写成了M导致了图不完整
        cin>>a>>b>>c;
        int ai = toNum(a);
        int bi = toNum(b);
        cost[ai][bi] = cost[bi][ai] = c;
    }

    int i = 1001;
    vector<Node> res;
    vector<int> d(1020,Inf);
    for(;i<=1000+M;i++){
        shortPath(i,d);
        bool flag = true;
        int min = Inf;
        int ans = 0;
        for(int j=1;j<=N;j++){
            if(d[j] > Ds){
                flag = false;
                break;
            }
            if(d[j] < min){
                min = d[j];
            }
            ans += d[j];
        }
        if(flag){
            res.push_back(Node(i,min,((float)ans)/N));
        }
    }
    if(res.size()==0){
        cout<<"No Solution";
    }else{
        float min = Inf;
        float max = -1;
        int u = -1;
        for(unsigned i=0;i<res.size();i++){
            if(res[i].min > max){ //最小距离应该最远
                max = res[i].min;
                min = res[i].avg;
                u = i;
            }else if(res[i].min == max){
                if(res[i].avg < min){
                    max = res[i].min;
                    min = res[i].avg;
                    u = i;
                }else if(res[i].avg == min){
                    if(res[i].id < res[u].id)
                        u = i;
                }
            }
        }
        cout<<"G"<<res[u].id-1000<<endl;
        printf("%.1f %.1f",res[u].min,res[u].avg);
    }
    return 0;
}
