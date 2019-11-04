#include<iostream>
#include<vector>
#include<algorithm>
#include<cstdio>
using namespace std;
int scanf(const char*,...);
vector<int> dia(100020);
vector<int> sum(100020,0);
int N,M;
struct Node{
    int i,j;
    int sum;
    Node(int _i,int _j,int s):i(_i),j(_j),sum(s){}
    bool operator < (const Node &a)const{
        if(this->sum == a.sum)
            return this->i < a.i;
        return this->sum < a.sum;
    }
};
vector<Node> vn;
int main()
{
	int nearM = 1e8+10;
	scanf("%d%d",&N,&M);
    for(int i=1;i<=N;i++){
		scanf("%d",&sum[i]);
		sum[i] += sum[i-1];
	}
	for(int i=1;i<=N;i++){
		int j = upper_bound(sum.begin()+i,sum.begin()+N+1,sum[i-1]+M)-sum.begin();
        if(sum[j-1] - sum[i-1] == M){
            nearM = M;
            break;
        }else if(j<=N && sum[j]-sum[i-1]<nearM){
            nearM = sum[j] - sum[i-1];
        }
	}
    for(int i=1;i<=N;i++){
		int j = upper_bound(sum.begin()+i,sum.begin()+N+1,sum[i-1]+nearM)-sum.begin();
        if(sum[j-1]-sum[i-1]==nearM)
            printf("%d-%d\n",i,j-1);
    }
	/*
    scanf("%d%d",&N,&M);
    for(int i=1;i<=N;i++) scanf("%d",&dia[i]);
    int i=1,j=1,sum=0;
    bool flag = true;
    while(i<=N && j<=N){
        sum += dia[j];
        if(sum<M){
            j++;
        }else if(sum==M){
			printf("%d-%d\n",i,j);//使用cout有一个测试点超时
            flag = false;
            sum -= dia[i];
            i++;j++;
        }else{//sum > M
            if(dia[j] > M){
                vn.push_back(Node(j,j,dia[j]));
                i = j+1;
                j = j+1;
                sum = 0;
            }else{
                if(vn.size()==0)
                    vn.push_back(Node(i,j,sum));
                else if(vn[0].sum >= sum)
                    vn.push_back(Node(i,j,sum));
                sum -= dia[i];i++;
                sum -= dia[j];
            }
        }
    }
    if(flag){
        sort(vn.begin(),vn.end());
        printf("%d-%d\n",vn[0].i,vn[0].j);
        int lim = vn[0].sum;
        for(int i=1;i<(int)vn.size();i++){
            if(vn[i].sum <= lim){
                printf("%d-%d\n",vn[i].i,vn[i].j);
            }else
                break;
        }
    }
    */
    return 0;
}
