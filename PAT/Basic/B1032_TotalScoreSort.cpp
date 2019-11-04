#include<iostream>
#include<map>

using namespace std;

int main()
{
    map<int,int> im;
    int n;
    cin>>n;
    int index,score;
    for(int i=0;i<n;i++){
        cin>>index>>score;
        im[index] += score;
    }
    score  =-1;
    for(map<int,int>::iterator it=im.begin();it!=im.end();it++){
        if(it->second > score){
            index = it->first;
            score = it->second;
        }
    }
    cout<<index<<" "<<score;
    return 0;
}
