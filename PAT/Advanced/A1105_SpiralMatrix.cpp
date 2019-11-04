#include<iostream>
#include<vector>
#include<algorithm>
#include<cmath>
using namespace std;
int N;
vector<int> v;
bool cmp(int a,int b){
    return a>b;
}

struct Pos{
    int x,y;
    Pos():x(0),y(0){}
	Pos(int r,int c):x(r),y(c){}
};


int main()
{
    cin>>N;
    v.resize(N);
    for(int i=0;i<N;i++){
        cin>>v[i];
    }
    sort(v.begin(),v.end(),cmp);
    int m = sqrt(N);
    int n,min=N+1;
    int mm,nn;
    for(;m<=N;m++){ // m>=sqrt(N) && m<=N,即有可能出现为一列的情况 
        n = N/m;
        if(m*n==N && m-n>=0 && m-n<min){
            min = m-n;
            mm = m;
            nn = n;
        }
    }
    vector< vector<int> > matrix(mm+1,vector<int>(nn+1,0));
    
    Pos lu,rd(mm-1,nn-1);
    int index = 0;
    while(lu.x<=rd.x && lu.y<=rd.y && index<N){
		
		if(lu.x==rd.x && lu.y==rd.y){
			matrix[lu.x][lu.y] = v[index];
			break;
		}else if(lu.x==rd.x){
			for(int i=lu.y;i<=rd.y;i++)
				matrix[lu.x][i] = v[index++];
			break;
		}else if(lu.y==rd.y){
			for(int i=lu.x;i<=rd.x;i++)
				matrix[i][lu.y] = v[index++];
			break;
		}else{
            for(int i=lu.y;i<rd.y && index<N;i++)
                matrix[lu.x][i] = v[index++];
            for(int i=lu.x;i<rd.x && index<N;i++)
                matrix[i][rd.y] = v[index++];
            for(int i=rd.y;i>lu.y && index<N;i--)
                matrix[rd.x][i] = v[index++];
            for(int i=rd.x;i>lu.x && index<N;i--)
                matrix[i][lu.y] = v[index++];
        }
        lu.x += 1;
        lu.y += 1;
        rd.x -= 1;
        rd.y -= 1;
    }
    
    for(int i=0;i<mm;i++){
        for(int j=0;j<nn-1;j++)
            cout<<matrix[i][j]<<" ";
        cout<<matrix[i][nn-1]<<endl;
    }
    return 0;
}
