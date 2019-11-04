#include<iostream>
#include<string>
using namespace std;

int main()
{
    int n;
    string name,id,maxn,maxid,minn,minid;
    int grade,max = -1,min=101;
    cin>>n;
    for(int i=0;i<n;i++){
        cin>>name>>id>>grade;
        if(max < grade){
            max = grade;
            maxn = name;
            maxid = id;
        }
        if(min > grade){
            min = grade;
            minn = name;
            minid = id;
        }
    }
    cout<<maxn<<" "<<maxid<<endl;
    cout<<minn<<" "<<minid;
    return 0;
}
