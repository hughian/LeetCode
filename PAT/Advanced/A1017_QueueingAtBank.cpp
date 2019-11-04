#include<iostream>
#include<vector>
#include<string>
#include<cstdio>
#include<algorithm>
using namespace std;
struct Node{
    int intime,outtime,p;
    int wait;
    Node(int h,int m,int s,int _p)
        :intime(h*60*60+m*60+s),outtime(0),p(_p),wait(0){}
    bool operator < (const Node &a){
        return this->intime < a.intime;
    }
};
vector<pair<Node,bool> > wid(120);
vector<Node> vc;
int N,K;
int main()
{
    cin>>N>>K;
    int hh,mm,ss,p;
    double ans = 0;
    for(int i=0;i<N;i++){
        scanf("%d:%d:%d %d",&hh,&mm,&ss,&p);
        if(p>60) p = 60;
        vc.push_back(Node(hh,mm,ss,p));
    }
    for(int i=0;i<K;i++){
        wid[i].second = false;
    }
    sort(vc.begin(),vc.end());
    for(int i=0;i<N;i++){
        if(vc[i].intime < 8*60*60){
            vc[i].wait = 8*60*60 - vc[i].intime;
            vc[i].intime = 8*60*60;
        }else break;
    }
    int que = 0,del=0,cur=8*60*60;
    while(del<N){
        for(int i=0;i<K;i++){
            if(wid[i].second == false && que < N){
                wid[i].first = vc[que++];
                wid[i].second = true;
            }
        }
        int minP = 61;
        vector<int> vi;
        for(int i=0;i<K;i++){
            if(wid[i].second == true){
                if(wid[i].first.p < minP){
                    minP = wid[i].first.p;
                    vi.clear();
                    vi.push_back(i);
                }else if(wid[i].first.p == minP){
                    vi.push_back(i);
                }
            }
        }
        for(int i=0;i<(int)vi.size();i++){
            if(wid[i].second == true){
                wid[i].first.wait += cur-wid[i].first.intime;
                cur+= minP;
                ans += wid[i].first.wait;
                wid[i].second = false;
            }
        }
    }
    
}
