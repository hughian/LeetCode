#include<iostream>
#include<vector>
#include<set>
#include<algorithm>
using namespace std;

//vector<int> coin(100010,0);
multiset<int> coin;
int N,M;
int main()
{
    cin>>N>>M;
    int sum = 0;
    int t;
    for(int i=0;i<N;i++){
        cin>>t;
        coin.insert(t);
        sum += t;
    }
    if(sum<M){
        cout<<"No Solution";
        return 0;
    }
    multiset<int>::iterator it = coin.begin(),tt;
    for(;it!=coin.end();it++){
        int tmp = M - (*it);
        if(*it >= M) break;
        if((coin.count(tmp)==1 && tmp!=(*it))||coin.count(tmp)>1){
            cout<<(*it)<<" "<<tmp;
            return 0;
        }
    }
    cout<<"No Solution";
    return 0;
}
