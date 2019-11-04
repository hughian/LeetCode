#include<iostream>
#include<vector>
#include<algorithm>

using namespace std;


//递归复杂度高，不如直接遍历查找
int MAX = 0;
int isPerfect(vector<long long> &vec,int low,int high,long long p)
{
    if(high<low)
        return 0;
    int ans;
    //static int i = 0;
    //cout<<++i<<" "<<vec[low]<<" "<<vec[high]<<" "<<high-low+1<<endl;
    if(vec[high] <= vec[low] * p)
        ans = high-low+1;
    else{
        if(MAX < high-low) //只能减掉半支，复杂度没有降低
            ans = max(isPerfect(vec,low+1,high,p),isPerfect(vec,low,high-1,p));
    }
    MAX = ans;
    return ans;
}



int main()
{
    //数据元素和p的取值范围都是<=10^9,而且有min * p 所以要使用long long
    long long n,p;
    vector<long long> vec;
    cin>>n>>p;
    vec.resize(n);
    for(int i=0;i<n;i++)
        cin>>vec[i];
    sort(vec.begin(),vec.end());
    int ans = 0;
    //ans=isPerfect(vec,0,n-1,p);
    int tmp; 
    for(int i=0;i<n;i++){
        //upper_bound函数返回指向第一个大于vec[i]*p元素的迭代器
        tmp = upper_bound(vec.begin()+i,vec.end(),vec[i]*p) - vec.begin();
        ans = max(ans,tmp-i);
    }
    cout<<ans;
    return 0;
}
