#include<iostream>
#include<stack>

using namespace std;

typedef struct {
	int w,h;
}Node;

int main(void)
{
	int i,j,k,t;
	stack<Node> s;
	cin>>i;
	while(cin>>t,~t){
		int max = 0,tmp;
		int width;
		Node *rect = new Node[t+2];
		for(int i=0;i<t;i++){
			cin>>rect[i].h;
			rect[i].w = 1;
			if(s.empty())
				s.push(rect[i]);
			else{
				width = tmp = 0;
				if(rect[i].h >= s.top().h)
					s.push(rect[i]);
				else{
					while(!s.empty()){
						if(rect[i].h < s.top().h){
							width += s.top().w;
							if((tmp = width*s.top().h) > max)
								max = tmp;
							s.pop();
						}
						else
							break;
					}
					width += rect[i].w;
					rect[i].w = width;
					s.push(rect[i]);
				}
			}
		}
		width = tmp = 0;
		while(!s.empty()){
			width += s.top().w;
			if((tmp=width*s.top().h) > max)
				max = tmp;
			s.pop();
		}
		cout<<max<<endl;
		delete []rect;
	}
	return 0;
}