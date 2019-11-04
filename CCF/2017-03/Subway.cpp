#include<iostream>
#include<vector>
#include<limits.h>
#include<queue>
using namespace std;
struct Node{
    int v,c;
    struct Node *next;
    Node():v(0),c(INT_MIN),next(0){}
    Node(int _v,int _c):v(_v),c(_c),next(0){}
};

class Solution{
    vector<Node *> VList;
    vector<int> cost;
    vector<bool> visited;
    queue<int> que;
    int GetCost(int v1,int v2){
        Node *p = VList[v1]->next;
        while(p){
            if(p->v == v2)
                return p->c;
            p = p->next;
        }
        return INT_MIN;
    }
    void BFS(int v){
        que.push(v);
        visited[v] = true;
        cost[v] = 0;
        while(!que.empty()){
            int vi = que.front();que.pop();
			visited[vi] = false;
            Node *p = VList[vi]->next;
            while(p){
                int vt = p->v;
				int ct = p->c;
                int tmp  = cost[vi] > ct ? cost[vi] : ct;
                if(tmp < cost[vt]){
                    cost[vt] = tmp;
                    if(!visited[vt]){
                        que.push(vt);
                        visited[vt] = true;
                    }
                }
                p = p->next;
            }
        }
    }
public:
    void Subway(){
        int n,m;
        cin>>n>>m;
        VList.resize(n+1);
        cost.resize(n+1);
        visited.resize(n+1);
        for(int i=0;i<n+1;i++){
            VList.at(i) = new Node();
            cost.at(i) = INT_MAX;
            visited.at(i) = false;
        }
        int v1,v2,c;
        for(int i=0;i<m;i++){
            cin>>v1>>v2>>c;
            Node *p = new Node(v2,c);
            p->next = VList.at(v1) -> next;
            VList.at(v1)->next = p;
        }
        BFS(1);
        cout<<cost[n]<<endl;
    }
};

int main()
{
    Solution s;
    s.Subway();
    return 0;
}
