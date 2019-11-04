#include<iostream>
#include<vector>
#include<cmath>
#include<algorithm>
using namespace std;
vector<long long> pvec(22,0);
vector<vector<int> > ans;
vector<int> tmpAns;
int N,K,P;
int getsum(vector<int> &v){
    int sum = 0;
    for(unsigned i=0;i<v.size();i++)
        sum += v[i];
    return sum;
}
bool cmp(vector<int>&a,vector<int>&b)
{
    int sum1 = getsum(a),sum2 = getsum(b);
    if(sum1 == sum2){
        for(int i=0;i<K;i++){
            if(a[i]!=b[i])
                return a[i]>b[i];
        }
    }
    return sum1>sum2;
}
void print(vector<int> &v){
    int len = v.size();
    for(int i=0;i<len;i++)
        cout<<v[i]<<" ";
    cout<<endl;
}
void select(int n,vector<int>&tmpAns,int i){
    int len = tmpAns.size();
    if(n==0 && len==K){
        ans.push_back(tmpAns);
        return;
    }
    if(n<0 || i<1 || len >K ){
        return;
    }
    tmpAns.push_back(i);
    select(n-pvec[i],tmpAns,i);
    tmpAns.pop_back();
    select(n,tmpAns,i-1);
}
int main()
{
    cin>>N>>K>>P;
    for(int i=1;i<22;i++){
        pvec[i] = round(pow(i,P));
    }
    int i= floor(sqrt(N));
    select(N,tmpAns,i);
    if(ans.size()==0)
        cout<<"Impossible";
    else{
        sort(ans.begin(),ans.end(),cmp);
        /*for(int i=0;i<(int)ans.size();i++){
            print(ans[i]);
        }*/
        cout<<N<<" =";
        for(int i=0;i<K;i++){
            cout<<" "<<ans[0][i]<<"^"<<P;
            if(i<K-1)
                cout<<" +";
        }
    }
    
    return 0;
}
