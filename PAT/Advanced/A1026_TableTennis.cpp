#include<iostream>
#include<string>
#include<queue>
#include<vector>
#include<cstdio>
#include<cmath>
#include<algorithm>
using namespace std;
int printf(const char*,...);
int scanf(const char*,...);
struct Player{
    int hh,mm,ss;
    int arrt,wait,sevt,playt;
    Player(int h,int m,int s,int p)
        :hh(h),mm(m),ss(s),wait(0),sevt(0){
        arrt  = h*60*60 + m*60 + s;
        playt = p;
    }
    int getCur(){
        return this->arrt + this->wait;
    }
    bool operator < (const Player &a)const{
        return this->arrt < a.arrt;
    }
};
vector<Player> vipQueue;
vector<Player> cusQueue;
vector<Player> ansQueue;
vector<int> table(110,0);
vector<int> vipTable;
bool flagc,flagv;
const int CloseTime = 21*60*60;
int tableNum;
void print(vector<Player> &v){
    int len = v.size();
    int h,m,s;
    for(int i=0;i<len;i++){
        h = v[i].sevt /3600;
        m = (v[i].sevt/60)%60;
        s = v[i].sevt%60;
        printf("%02d:%02d:%02d %02d:%02d:%02d %d\n",\
                v[i].hh,v[i].mm,v[i].ss,h,m,s,(int)(v[i].wait/60.0+0.5));
    }
    printf("\n");
}
void printTable()
{
    int len = tableNum;
    int hh,mm,ss;
    for(int i=0;i<len;i++){
        hh = table[i]/3600;
        mm = (table[i]/60)%60;
        ss = table[i]%60;
        printf("%dth:%05d %02d:%02d:%02d\n",i,table[i],hh,mm,ss);
    }
    printf("++++++++++++++++++++++++\n");
}
void getFlag(){
    flagc = cusQueue.size()>0 && cusQueue[0].getCur() < CloseTime;
    flagv = vipQueue.size()>0 && vipQueue[0].getCur() < CloseTime;
}
int findTable(int cur){
    int len = tableNum;
    for(int i=0;i<len;i++){
        if(cur >= table[i])
            return i;
    }
    return -1;
}

int findVipTable(int cur)
{
    int len = vipTable.size();
    for(int i=0;i<len;i++){
        if(cur >= table[vipTable[i]])
            return vipTable[i];
    }
    return findTable(cur);
}

void update()
{
    int minTable = CloseTime+600;
    int minPlayer = CloseTime+600;
    int len = tableNum;
    for(int i=0;i<len;i++){
        if(table[i] < minTable)
            minTable = table[i];
    }
    if(flagc && flagv){
        if(vipQueue[0].getCur() <= cusQueue[0].getCur())
            minPlayer = vipQueue[0].getCur();
        else
            minPlayer = cusQueue[0].getCur();
    }else if(flagc && !flagv)
        minPlayer = cusQueue[0].getCur();
    else if(!flagc && flagv)
        minPlayer = vipQueue[0].getCur();

    int diff = minTable - minPlayer;
    len = cusQueue.size();
    for(int i=0;i<len;i++)
        cusQueue[i].wait += diff;
    len = vipQueue.size();
    for(int i=0;i<len;i++)
        vipQueue[i].wait += diff;
}

void serveVip()
{
    int ix = findVipTable(vipQueue[0].getCur());
    if(ix==-1){
        update();
    }else{
        vipQueue[0].sevt = vipQueue[0].getCur();
        table[ix] = vipQueue[0].getCur() + vipQueue[0].playt*60;
        ansQueue.push_back(vipQueue[0]);
        vipQueue.erase(vipQueue.begin());
    }
}
void serveCus()
{
    int ix = findTable(cusQueue[0].getCur());
    if(ix == -1){
        update();
    }else{
        cusQueue[0].sevt = cusQueue[0].getCur();
        table[ix] = cusQueue[0].getCur() + cusQueue[0].playt*60;
        ansQueue.push_back(cusQueue[0]);
        cusQueue.erase(cusQueue.begin());
    }
}


int main()
{
    int N,K,M;
    cin>>N;
    int hh,mm,ss,pt,tag;
    for(int i=0;i<N;i++){
        scanf("%d:%d:%d %d %d",&hh,&mm,&ss,&pt,&tag);
        if(pt>120) pt=120; //最多玩两个小时
        if(tag == 0)
            cusQueue.push_back(Player(hh,mm,ss,pt));
        else
            vipQueue.push_back(Player(hh,mm,ss,pt));
    }
    cin>>K>>M;
    tableNum = K;
    int t;
    for(int i=0;i<M;i++){
        cin>>t;
        vipTable.push_back(t);
    }
    sort(vipQueue.begin(),vipQueue.end());
    sort(cusQueue.begin(),cusQueue.end());

    getFlag();
    while(flagc || flagv){
        printTable();
        if(flagc && flagv){ //都有顾客在等待
            if(vipQueue[0].arrt <= cusQueue[0].arrt){ //vip用户先到
                serveVip(); //这里逻辑有错误。并不是vip先到先服务，
                            //而是找到的桌子如果是vip。则先服务vip用户
            }else{
                serveCus();
            }
        }else if(flagc && !flagv){
            serveCus();
        }else if(!flagc && flagv){
            serveVip();
        }
        getFlag();
    }
    
    print(ansQueue);
    return 0;
}
