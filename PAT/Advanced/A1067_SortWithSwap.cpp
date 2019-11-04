#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;
vector<int> v;
int N;
int main()
{
    cin>>N;
    v.resize(N);
    vector<bool> used(N,false);
    for(int i =0;i<N;i++){
        cin>>v[i];
    }
    int cnt = 1,ans=0;
    int p = v[0];
    used[0] = true;
    while(p!=0){
        used[p] = true;
        p = v[p];
        cnt++;
    }
    ans += (cnt-1);
    cnt = 0;
    for(int i=1;i<N;i++){
        if(!used[i]){
            p = v[i];
            cnt = 1;
            used[i] = true;
            while(p!=i){
                used[p] = true;
                p = v[p];
                cnt++;
            }
            if(cnt>1)
                ans += (cnt+1);
        }
    }
    cout<<ans;
    return 0;
}
