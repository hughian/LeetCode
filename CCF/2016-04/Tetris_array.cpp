#include<iostream>
using namespace std;
int main(){
    int a[15][10];
    int b[4][4];
    int i,j,s,t;
    int c[4]={14,14,14,14},d[4]={-20,-20,-20,-20};//c[j]=i在下面用来保存原始数组中第j列在第i行可以填充；d[j]=i表示输入数组第j列第i行有1 
    //input data from cin
	for(i=0;i<15;i++)
		for(j=0;j<10;j++)
			cin>>a[i][j];
    for(i=0;i<4;i++)
			for(j=0;j<4;j++)
			cin>>b[i][j];
    cin>>s;
	//finished input
    s=s-1;//从第s列输入，下标从0开始 
    for(j=s;j<s+4;j++)
		for(i=0;i<15;i++){
			if(a[i][j]==1){
				c[j-s]=i-1;
				break;//找到原始数组第i列有1的最小行 
			}
		}
	
    for(j=0;j<4;j++){
        for(i=0;i<4;i++){
            if(b[i][j]==1)
				d[j]=i;//找到输入数组中第i列有1的最大行 
        }
    }
    int minl=15;
    for(j=0;j<4;j++)
		if(c[j]-d[j]<minl){
			minl=c[j]-d[j];//四个列相减最小值约束方块放置位置，只需要通过这个位置求得输入数组第一个元素放在原始数组的哪个位置即可 
			t=j;//列
		}
	
    int h=c[t],h1=d[t];
    for(i=0;i<4;i++)
		for(j=0;j<4;j++){
			if(b[i][j]==1)
				a[h-h1+i][s+j]=b[i][j];//输入数组第[0,0]个元素放在原始数组第[h-h1,s]中 
		}
    for(i=0;i<15;i++){
        for(j=0;j<10;j++){
			cout<<a[i][j]<<' ';		
        }
		cout<<a[i][j]<<endl;
    }
}