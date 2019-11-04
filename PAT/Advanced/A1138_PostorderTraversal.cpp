#include<iostream>

using namespace std;


void func(int pre[],int in[],int prelow,int prehigh,int inlow,int inhigh)
{
    while(prelow<=prehigh && inlow <= inhigh){
        int root = pre[prelow];
        int j =inlow ;
        for(;j<inhigh;j++){
            if(root == in[j])
                break;
        }
        if(j==inlow && j==inhigh){ //叶节点
            cout<<in[j]<<endl;
            return ;
        }else if(j>inlow) { //有左子树
            int len = j - inlow;
            inhigh = j-1;
            prehigh = prelow + len;
            prelow += 1;
        }else{ //只有右子树
            prelow ++;
            inlow ++;
        }
    }
}
int main()
{
    int N;
    cin>>N;
    int pre[N],in[N];
    
    for(int i=0;i<N;i++)
        cin>>pre[i];
    for(int i=0;i<N;i++)
        cin>>in[i];
    func(pre,in,0,N-1,0,N-1);
    return 0;
}
