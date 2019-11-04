#include<iostream>
#include<vector>
#include<stack>
using namespace std;
int N,M,K;
int main()
{
    cin>>M>>N>>K;
    vector<int> vec(N);
    for(int i=0;i<K;i++){
        for(int j=0;j<N;j++)
            cin>>vec[j];
        int n = 1,k=0;
        stack<int> s;
        while(k<N && n<=N+2){
            if(s.empty()){
                s.push(n++);
            }else if(s.top()==vec[k]){
                s.pop();
                k++;
            }else if((int)s.size()==M){
                break;
            }else{
                s.push(n++);
            }
        }
        if(s.empty()) cout<<"YES\n";
        else     cout<<"NO\n";
    }
    return 0;
}
