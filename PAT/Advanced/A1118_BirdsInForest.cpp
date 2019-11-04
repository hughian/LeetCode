#include<iostream>
#include<vector>
using namespace std;

int pic[10001];
int bird[10001];
int find(int x){
    int r = x;
    while(pic[r] != -1){
        r = pic[r];
    }
    return r;
}
void join(int a,int b)
{
    int fa = find(a);
    int fb = find(b);
    if(fa!=fb)
        pic[fa] = fb;
}
int main()
{
    int n;
    cin>>n;
    int k,tmp;
    int upbound = -1;
    for(int i=0;i<10001;i++){
        pic[i] = bird[i] = -1;
    }
    for(int i=1;i<=n;i++){
        cin>>k;
        for(int j=1;j<=k;j++){
            cin>>tmp;
            if(upbound < tmp)
                upbound = tmp;
            if(bird[tmp]==-1)
                bird[tmp] = i;
            join(i,bird[tmp]);
        }
    }
    int cnt = 0;
    for(int i=1;i<=n;i++)
        if(pic[i]==-1) cnt++;
    cout<<cnt<<" "<<upbound<<endl;

    int q;
    cin>>q;
    int a,b;
    for(int i=0;i<q;i++){
        cin>>a>>b;
        int fa = find(bird[a]);
        int fb = find(bird[b]);
        if(fa==fb)
            cout<<"Yes"<<endl;
        else
            cout<<"No"<<endl;
    }
    return 0;
}
