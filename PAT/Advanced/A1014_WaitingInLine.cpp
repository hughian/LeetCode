#include<iostream>
#include<string>
#include<queue>
#include<vector>
using namespace std;
struct Node{
    int id;
    int time,out;
};
int N,M,K,Q;
bool isempty(vector<queue<Node> > &ques)
{
    for(int i=0;i<N;i++){
        if(!ques[i].empty()) return false;
    }
    return true;
}
int main()
{
    cin>>N>>M>>K>>Q;
    vector< queue<Node> > ques(N);
    vector<Node> vc(K+1);
    for(int i=0;i<K;i++){
        cin>>vc[i].time;
        vc[i].id = i;
    }
    int t=0;
    int cur = 8*60;
    for(int j=0;j<M;j++){
        for(int i=0;i<N && t<K;i++){
            ques[i].push(vc[t++]);
        }
    }
    int min,u;
    vector<int> us;
    while(!isempty(ques) && cur<=17*60){
        min = 20*60;
        u = -1;
        for(int i=0;i<N;i++){
            if(ques[i].size()>0){
                if(ques[i].front().time < min){
                    min = ques[i].front().time ;
                    us.clear();
                    us.push_back(i);
                }else if(ques[i].front().time == min){
                    us.push_back(i);
                }
            }
        }
        if(us.size()==0) return 0;
        for(unsigned i=0;i<us.size();i++){
            u = us[i];
            Node q = ques[u].front();
            ques[u].pop();
            if(t<K)
                ques[u].push(vc[t++]);
            cur += min;
            vc[q.id].out = cur;
            ques[u].front().time += min;
        }
        for(int i=0;i<N;i++){
            if(ques[i].size()>0){
                ques[i].front().time -= min;
            }
        }
    }
    int id;
    for(int i=0;i<Q;i++){
        cin>>id;
        if(vc[id-1].out >= 8*60 && vc[id-1].out <=17*60){
            printf("%02d:%02d\n",vc[id-1].out/60,vc[id-1].out%60);
        }else{
            printf("Sorry\n");
        }
    }
    return 0;
}
