#include<iostream>
#include<vector>
using namespace std;

int main()
{
    int k,flg = 0;
    vector<int> v,res;
    cin>>k;
	
	if(k<=0)  return 0;
    
	v.resize(k);
    res.resize(k);
    for(int i=0;i<k;i++){
        cin>>v[i];
        if(v[i]>=0)
            flg = 1;
    }
    if(!flg){
        cout<<0<<" "<<v[0]<<" "<<v[k-1];
        return 0;
    }
    
	res[0] = v[0];
    int max = v[0];
    pair<int,int> mt;
    int tmpfirst = mt.first = mt.second = v[0];
 
    for(int i=1;i<k;i++){
        if(res[i-1] >= 0){
            res[i] = res[i-1] + v[i];
        }
        else{
            res[i] = v[i];
            tmpfirst = v[i];
        }
        if(max < res[i] ){
            max = res[i];
            mt.first = tmpfirst;
            mt.second = v[i];
        }
    }
    cout<<max<<" "<<mt.first<<" "<<mt.second;
    return 0;
}


