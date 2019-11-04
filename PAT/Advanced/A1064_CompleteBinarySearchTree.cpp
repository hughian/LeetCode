#include<iostream>
#include<vector>
#include<algorithm>
#include<queue>
using namespace std;
vector<int> vec(1010,0);
vector<int> tree(1010,0);
int N;
int cnt = 0;
void inOrder(int root){
    int left = 2*root + 1;
    int right = left+1;
    if(left >=N){ //р╤вс╫А╣Ц
        tree[root] = vec[cnt++];
    }else if(right >= N){
        inOrder(left);
        tree[root] = vec[cnt++];
    }else{
        inOrder(left);
        tree[root] = vec[cnt++];
        inOrder(right);
    }
}

void lvOrder(int root){
    queue<int> q;
    q.push(root);
    cnt = 0;
    while(!q.empty()){
        int t = q.front();q.pop();
        cout<<tree[t];
        if(cnt<N-1) cout<<" ";
		cnt++;
        int left = 2*t+1;
        int right = left+1;
        if(left<N) q.push(left);
        if(right<N) q.push(right);
    }
}


int main()
{
    cin>>N;
    for(int i=0;i<N;i++)
        cin>>vec[i];
    sort(vec.begin(),vec.begin()+N);
    inOrder(0);
    lvOrder(0);
    return 0;
}
