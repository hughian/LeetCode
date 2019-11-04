#include<iostream>
#include<vector>
using namespace std;

int main()
{
    int n;
    cin>>n;
    vector<int> ans(1,n),tpv;
    int tmpi;
    for(long long i=2;i*i<=n;i++){
        tpv.clear();
        tmpi = n;
        for(long long j=i;j<=tmpi;j++){  //只需要找到n的一段连续的n的因子即可，不必将n全部分解
            if(tmpi%j==0){
                tpv.push_back(j);
                tmpi /= j;
            }
            else
                break;
        }
        if(tpv.size() > ans.size())
            ans = tpv;
        else if(tpv.size() == ans.size() && tpv[0]<ans[0])
            ans = tpv;
    }
    cout<<ans.size()<<endl;
    cout<<ans[0];
    for(int i=1;i<(int)ans.size();i++)
        cout<<"*"<<ans[i];

    return 0;
}
