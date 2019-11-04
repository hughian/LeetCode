#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

vector<int> vec(10010,0);
int N,M;
vector<int> ans;
vector<int> tmp;

void traceb(int m,int i,long long sum){
    if(ans.size()!=0) //进行排序后，第一个选中就是最后的结果
        return;
    if(m==0){
        if(ans.size()==0)
            ans = tmp;
        return ;
    }
    if(i>=N) return;
    if(m>sum || m<0 || m-vec[i]<0)return; //分支限界
	
    tmp.push_back(vec[i]);
    traceb(m-vec[i],i+1,sum-vec[i]);
    tmp.pop_back();
    traceb(m,i+1,sum-vec[i]);
}

int main()
{
    long long sum = 0;
    cin>>N>>M;
    for(int i=0;i<N;i++){
        cin>>vec[i];
        sum += vec[i];
    }
    sort(vec.begin(),vec.begin()+N);
    //for(int i=0;i<N;i++)
    //    cout<<vec[i]<<" ";
    //cout<<endl;
    traceb(M,0,sum);
    if(ans.size()==0)
        cout<<"No Solution";
    for(unsigned i=0;i<ans.size();i++){
        cout<<ans[i];
        if(i<ans.size()-1)
            cout<<" ";
    }
    return 0;
}
