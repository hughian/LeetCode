#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;
vector<int> va(1e6+10,0);
vector<int> vb(1e6+10,0);
vector<int> vc(2e6+20,0);
int na,nb;
int nc = 0;
void merge(){
    int i=0,j=0;
    while(i<na && j<nb){
        if(va[i]<vb[j])
            vc[nc++] = va[i++];
        else
            vc[nc++] = vb[j++];
    }
    while(i<na){
        vc[nc++] = va[i++];
    }
    while(j<nb){
        vc[nc++] = vb[j++];
    }
}
int main()
{
    cin>>na;
    for(int i=0;i<na;i++) cin>>va[i];
    cin>>nb;
    for(int i=0;i<nb;i++) cin>>vb[i];
    merge();
    int mid = (nc-1)/2;
    cout<<vc[mid];
    return 0;
}
