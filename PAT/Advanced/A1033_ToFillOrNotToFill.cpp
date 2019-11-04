#include<iostream>
#include<string>
#include<algorithm>
#include<vector>
using namespace std;
const int Inf = 1e9;
vector< pair<double,double> > vpd(510);

bool cmp(pair<double,double> &a,pair<double,double>&b){
    return a.first < b.first;
}

int main()
{
    int n;
    double Cmax,D,Davg;
    cin>>Cmax>>D>>Davg>>n;
    for(int i=0;i<n;i++){
        cin>>vpd[i].second>>vpd[i].first;
    }
    vpd[n].first = D;
    vpd[n].second = 0.0;
    sort(vpd.begin(),vpd.begin()+n,cmp);  //按照距离排序

    if(vpd[0].first != 0){
        cout<<"The maximum travel distance = 0.00\n";
    }else{
        int now = 0; //当前加油站
        double ans= 0.0,nowTank=0.0,MAX=Cmax*Davg;//nowTank 是当前油量，MAX是满油行驶的最大距离
        while(now < n){
            int ix = -1;
            double priceMin = Inf;
			//从当前加油站向后搜索，同时，不能超过满油行驶的最大距离
            for(int i=now+1;i<=n && vpd[i].first - vpd[now].first<=MAX;i++){
                if(vpd[i].second < priceMin){ //在右边找油价最低的
                    priceMin = vpd[i].second;
                    ix = i;
                }
                if(priceMin < vpd[now].second) //如果找到一个比当前地，直接前往这个
					break;
            }
            if(ix==-1) break;
            double need = (vpd[ix].first - vpd[now].first)/Davg;
            if(priceMin < vpd[now].second){
                if(nowTank<need){
                    ans += (need-nowTank)*vpd[now].second;
                    nowTank = 0;
                }else{
                    nowTank -= need;
                }
            }else{
                ans += (Cmax - nowTank)*vpd[now].second;
                nowTank = Cmax - need;
            }
            now = ix;
        }
        if(now == n){
            printf("%.2f",ans);
        }else{
            printf("The maximum travel distance = %.2f\n",vpd[now].first+MAX);
        }
    }

}
