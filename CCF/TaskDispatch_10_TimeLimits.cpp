#include<iostream>
#include<vector>
#include<climits>
using namespace std;
int count = 0;
struct Task{
    int a,b,c,d;
    Task(int _a,int _b,int _c,int _d):a(_a),b(_b),c(_c),d(_d){}
    Task():a(0),b(0),c(0),d(0){}
};

class Solution{
    vector<Task> T;
    vector<int> ans;
	int  minAns;
private:
	inline int max(int a,int b){
		if(a > b)
			return a;
		else 
			return b;
	}
	inline int min(int a,int b){
		if(a > b)
			return b;
		else
			return a;
	}
protected:
	void BackTrack(int cnt,long long mask,int cpu0,int cpu1,int common,int gpu,int sum){
		int tmp;
        if(cnt < (int)T.size()){
            for(int k=0;k<(int)T.size();k++){
			    if(((mask>>k)&1)) continue;
                tmp = min(min(cpu0,cpu1)+common,gpu) + min(min(min(T[k].a,T[k].b),T[k].c),T[k].d);
				if(tmp >= minAns)
					continue;
				cout<<"mask "<<mask<<endl;
                //count ++;
				for(int i=0;i<4;i++){
			    	switch(i){
			    		case 0: //one cpu
                            tmp = max(max(cpu0+T[k].a,cpu1)+common,gpu);
			    			if(tmp >= minAns)
			    				break;
			    			BackTrack(cnt+1,(mask|(1<<k)),cpu0+T[k].a,cpu1,common,gpu,tmp);
			    			
			    			tmp = max(max(cpu0,cpu1+T[k].a)+common,gpu);
			    			if(tmp >= minAns)
			    				break;
			    			BackTrack(cnt+1,(mask|(1<<k)),cpu0,cpu1+T[k].a,common,gpu,tmp);
                            break;
                        case 1: //two cpus
                            tmp = max(max(cpu0,cpu1)+common+T[k].b,gpu);
                            if(tmp >= minAns)
			    				break;
			    			BackTrack(cnt+1,(mask|(1<<k)),cpu0,cpu1,common+T[k].b,gpu,tmp);
                            break;
                        case 2: //one cpu ans one gpu
                            tmp = max(max(cpu0+T[k].c,cpu1)+common,gpu+T[k].c);
                            if(tmp >= minAns)
			    				break;
			    			BackTrack(cnt+1,(mask|(1<<k)),cpu0+T[k].c,cpu1,common,gpu+T[k].c,tmp);
			    			
                            tmp = max(max(cpu0,cpu1+T[k].c)+common,gpu+T[k].c);
                            if(tmp >= minAns)
			    				break;
			    			BackTrack(cnt+1,(mask|(1<<k)),cpu0,cpu1+T[k].c,common,gpu+T[k].c,tmp);
                            break;
                        case 3://all of them
                            tmp = max(max(cpu0,cpu1)+common+T[k].d,gpu+T[k].d);
                            if(tmp >= minAns)
			    				break;
			    			BackTrack(cnt+1,(mask|(1<<k)),cpu0,cpu1,common+T[k].d,gpu+T[k].d,tmp);
                            break;
			    	}
			    }
            }
		}
        else{
			cout<<"mask_"<<mask<<endl;
			//count ++;
			minAns = min(minAns,sum);
			//ans.push_back(sum);
        }
	}
public:
	void TaskDispatch(){
		int n;
        int a,b,c,d;
        cin>>n;
        for(int i=0;i<n;i++){
            cin>>a>>b>>c>>d;
            T.push_back(Task(a,b,c,d));
        }
		ans.clear();
		minAns = INT_MAX;
        BackTrack(0,0,0,0,0,0,0);
		cout<<minAns<<endl;
		//cout<<count<<endl;
	}
};
int main()
{
	Solution s;
	s.TaskDispatch();
	return 0;
}
