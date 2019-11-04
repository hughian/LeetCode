#include<iostream>
#include<algorithm>
#include<vector>
using namespace std;

vector<int> first;
vector<int> second;
int N;
int min(int a,int b)
{
	return a>b?b:a;
}
void adjust(int len){
	int ix = 0;
	int t;
	bool flag;
	while(2*ix+1<=len){
		int left = 2*ix+1;
		if(left+1<=len && second[left] < second[left+1])
			left++;
		flag = false;
		if(second[ix] < second[left]){
			t = second[ix];
			second[ix] = second[left];
			second[left] = t;
			ix = left;
			flag = true;
		}
		if(!flag){
			break;
		}
	}
}

int main()
{
	cin>>N;
	int i,t;
	for(i=0;i<N;i++){
		cin>>t;
		first.push_back(t);
	}
	for(i=0;i<N;i++){
		cin>>t;
		second.push_back(t);
	}
	int pos;
	for(i=1;i<N;i++){
		if(second[i] < second[i-1]){
			pos = i;
			break;
		}
	}
	int f;
	for(f=pos;f<N;f++){
		if(first[f] != second[f])
			break;
	}

	if(f==N){	//insertion sort
		sort(second.begin(),second.begin()+min(pos+1,N));
		cout<<"Insertion Sort"<<endl;
		for(i=0;i<N;i++){
			cout<<second[i];
			if(i<N-1) cout<<" ";
		}
	}else{ //heap sort
		vector<int> tmp = first;
		sort(tmp.begin(),tmp.end());
		for(i=N-1;i>=0;i--){
			if(tmp[i] != second[i]) break;
		}
		int t = second[i];
		second[i] = second[0];
		second[0] = t;

		adjust(i-1);
		cout<<"Heap Sort"<<endl;
		for(i=0;i<N;i++){
			cout<<second[i];
			if(i<N-1) cout<<" ";
		}
	}
}