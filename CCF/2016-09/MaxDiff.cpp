#include<iostream>
#include<vector>
using namespace std;
int abs(int a,int b){
    return (a>b) ? (a-b) : (b-a);
}

int main()
{
    int n,tmp;
    cin>>n;
    vector<int> v;
    for(int i=0;i<n;i++){
        cin>>tmp;
        v.push_back(tmp);
    }
    int max = abs(v.at(0),v.at(1));
    for(int i=1;i<n-1;i++){
        tmp = abs(v.at(i),v.at(i+1));
        if(max < tmp)
            max = tmp;
    }
    cout<<max<<endl;
    return 0;
}
