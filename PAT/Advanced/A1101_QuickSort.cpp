#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

int main(){
    int N;
    cin>>N;
    vector<int> v(N);
	vector<bool> flags(N,true);
    for(int i=0;i<N;i++)
        cin>>v[i];
    int leftMax=-1,rightMin=1e9;
    vector<int> ans;
    for(int i=0;i<N;i++){
        if(v[i]<leftMax)
            flags[i] = false;
        if(v[i]>leftMax)
            leftMax = v[i];
    }
	for(int i=N-1;i>=0;i--){
		if(v[i]>rightMin)
			flags[i] = false;
		if(v[i]<rightMin)
			rightMin = v[i];
	}
    for(int i=0;i<N;i++){
		if(flags[i])
			ans.push_back(v[i]);
	}
	int len = ans.size();
	cout<<len<<endl;
    if(len==0) cout<<endl; //结果为0个时也要多输出一个空行
	sort(ans.begin(),ans.end());
	for(int i=0;i<len;i++){
		cout<<ans[i];
		if(i<len-1)
			cout<<" ";
	}
	return 0;
}
