#include<iostream>
#include<vector>
#include<string>
#include<cstdio>
#include<map>
using namespace std;
int scanf(const char*,...);
int printf(const char*,...);
struct DateTime{
    int valid;
    int mm,dd,hh,MM;
    DateTime():valid(0){}

};
vector< pair<DateTime,DateTime> > 
map<string,int> mp;
vector<int> toll(25,0);
vector<string> ns(1010);
int cnt = 0;
int getidx(string s){
    if(mp.count(s)==0){
        mp[s] = cnt;
        ns[cnt] = s;
        cnt++;
    }
}

int main()
{
    int N;
    int mm,dd,hh,MM;
    string name,status;
    for(int i=0;i<24;i++) cin>>toll[i];
    cin>>N;
    for(int i=0;i<N;i++){
        cin>>name;
        scanf("%d:%d:%d:%d",&mm,&dd,&hh,&MM);
        cin>>status;
        int ix = getidx(name);

    }

}
