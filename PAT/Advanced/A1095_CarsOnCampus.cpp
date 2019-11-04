#include<iostream>
#include<vector>
#include<string>
#include<map>
#include<set>
#include<cstdio>
#include<algorithm>
using namespace std;
struct Car{
    string plate;
    int in_time;
    int sum;
    int state,loc;
    Car(string s,int in):plate(s),in_time(in){}
    Car():plate(""),in_time(-1){}
    Car& setUp(string s,int in,int sm,int st,int lo){
        plate   = s;
        in_time = in;
        sum     = sm;
        state   = st;
        loc     = lo;
        return *this;
    }
};

struct Record{
    string plate;
    int time;
    int state;
    int flag;
    bool operator < (const Record &a){
        return this->time < a.time;
    }
};
vector<Car> vc(100001);
vector<Record> re(10010);
map<string,int> mp;
set<string> w;

int time2int(string time){
    int hh,mm,ss;
    sscanf(time.c_str(),"%d:%d:%d",&hh,&mm,&ss);
    return hh*60*60 + mm*60 + ss;
}

string int2time(int time){
    int sec = time %60;
    int min = (time/60) % 60;
    int hour = time/3600;
    char buf[10];
    sprintf(buf,"%02d:%02d:%02d",hour,min,sec);
    return string(buf);
}
int cnt = 0;
int getidx(string plate){
    if(mp.count(plate)==0)
        mp[plate]=cnt++;
    return mp[plate];
}

int main()
{
    int N,K;
    cin>>N>>K;
    string plate,time,status;
    for(int i=0;i<N;i++){
        cin>>plate>>time>>status;
        re[i].flag = 0;
        re[i].plate = plate;
        re[i].time = time2int(time);
        re[i].state = status.compare("in") == 0 ? 1 : 0;
    }
    sort(re.begin(),re.begin()+N);
    for(int i=0;i<N;i++){
        plate = re[i].plate;
        if(mp.count(plate)==0){
            int idx = getidx(plate);
            vc[idx].setUp(plate,re[i].time,0,re[i].state,i);
        }else{
            int idx = getidx(plate);
            if(vc[idx].state == 1 && re[i].state == 1){ //both in
                re[vc[idx].loc].flag = 1;
                vc[idx].loc = i;
            }else if(vc[idx].state == 1 && re[i].state == 0) //in -> out
                vc[idx].state = 0;
            else if(vc[idx].state == 0 && re[i].state == 1) {//out -> in
                vc[idx].state = 1;
                vc[idx].loc = i;
            }else if(vc[idx].state ==0 && re[i].state == 0)// both out
                re[i].flag = 1;
        }
    }

    for(int i=0;i<cnt;i++){
        if(vc[i].state == 1){
            re[vc[i].loc].flag = 1;
            vc[i].state = 0;
        }
    }
    vector<int> query;
    for(int i=0;i<K;i++){
        cin>>time;
        int t = time2int(time);
        query.push_back(t);
    }

    int num=0,ans=0,res = 0;
    for(int i=0;i<N;i++){
        if(re[i].flag==1) continue;
        while(num<K && query[num] < re[i].time){
            num++;
            printf("%d\n",ans);
        }
        int idx = getidx(re[i].plate);
        if(vc[idx].state == 1 && re[i].state == 0){
            ans --;
            vc[idx].state =0;
            vc[idx].sum += re[i].time - vc[idx].in_time;
            if(vc[idx].sum > res){
                res = vc[idx].sum;
                w.clear();
                w.insert(vc[idx].plate);
            }else if(vc[idx].sum == res){
                w.insert(vc[idx].plate);
            }
        }else if(vc[idx].state == 0 && re[i].state == 1){
            ans++;
            vc[idx].state = 1;
            vc[idx].in_time = re[i].time;
        }
    }
    while(num<K){
        printf("%d\n",ans);
        num++;
    }
    set<string>::iterator it = w.begin();
    for(;it!=w.end();it++)
        cout<<*it<<" ";
    cout<<int2time(res);
    return 0;
}
