#include<iostream>
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
vector<int> first;
vector<int> second;
int N;
int min(int a,int b)
{
  return a<b?a:b;
}
bool cmp(vector<int>& va,vector<int>& vb)
{
  for(int i=0;i<N;i++){
    if(va[i]!=vb[i])
      return false;
  }
  return true;
}
bool InsertSort(){
  vector<int> tmp = first;
  for(int i=1;i<=N;i++){
    sort(tmp.begin(),tmp.begin()+i);
    if(cmp(tmp,second)){ //是插入，再做一次
		while(cmp(tmp,second) && i<= N) {
			i++;
			sort(tmp.begin(),tmp.begin()+min(i,N));
		}
      cout<<"Insertion Sort"<<endl;
      //sort(tmp.begin(),tmp.begin()+min(i+1,N));
      for(int i=0;i<N;i++){
        cout<<tmp[i];
        if(i<N-1) cout<<" ";
      }
      return true;
    }
  }
  return false;
}

int mergeSort(vector<int>& v)
{
  for(int step = 2;step/2 <= N;step *=2){
    int k,i;
    for(i=0;i<N;i+=step){
      sort(v.begin()+i,v.begin()+min(i+step,N));
    }
    for(k=0;k<N;k++){
      if(v[k]!=second[k])
        break;
    }
    if(k==N){
      step *=2;
      for(i=0;i<N;i+=step){
        sort(v.begin()+i,v.begin()+min(i+step,N));
      }
      for(i=0;i<N;i++){
        cout<<v[i];
        if(i<N-1) cout<<" ";
      }
      return 0;
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
  vector<int> tmp = first;
  if(!InsertSort()){
    cout<<"Merge Sort"<<endl;
    mergeSort(tmp);
  }
  return 0;
}
