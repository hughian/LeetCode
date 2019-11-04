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
    vector<bool> visited;
	int  minAns;
    int cpu0,cpu1,common,gpu,allcom;
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
    inline int abs(int a,int b){
        return a>b?(a-b):(b-a);
    }
protected:
	int GetaMin(){
		int tmp[6];
        int pos[T.size()],min,sel;
        int gmin = INT_MAX;
        for(int k=0;k<(int)T.size();k++){
            if( !visited[k] ){
                tmp[0] = max(max(cpu0+T[k].a,cpu1)+common+allcom,gpu+allcom);
				tmp[1] = max(max(cpu0,cpu1+T[k].a)+common+allcom,gpu+allcom);
                tmp[2] = max(max(cpu0,cpu1)+common+T[k].b+allcom,gpu+allcom);
                tmp[3] = max(max(cpu0+T[k].c,cpu1)+common+allcom,gpu+T[k].c+allcom);
                tmp[4] = max(max(cpu0,cpu1+T[k].c)+common+allcom,gpu+T[k].c+allcom);
                tmp[5] = max(max(cpu0,cpu1)+common+T[k].d+allcom,gpu+T[k].d+allcom);
                min = INT_MAX;
                pos[k] = -1;
                for(int i=0;i<6;i++){
                    if(min > tmp[i]){
                        min = tmp[i];
                        pos[k] = i;
                    }
                }
                if(gmin > min){
                    gmin = min;
                    sel = k;
                }
            }
        }
        visited[sel] = true;
		switch(pos[sel]){
	    	case 0: cpu0 += T[sel].a;                   break; //cpu0
            case 1: cpu1 += T[sel].a;                   break; //cpu1
            case 2: common += T[sel].b;                 break; //cpu0 and cpu1
            case 3: cpu0 += T[sel].c; gpu += T[sel].c;  break; //cpu0 and gpu
            case 4: cpu1 += T[sel].c; gpu += T[sel].c;  break; //cpu1 and gpu
            case 5: allcom += T[sel].d                  ;break; //all
		}
        if(cpu0 > cpu1){ //cpu0 > cpu1
            common += cpu1;
            cpu0 -= cpu1;
            cpu1 = 0;
        }
        else if(cpu0 < cpu1){ //cpu0 < cpu1
            common += cpu0;
            cpu1 -= cpu0;
            cpu0 = 0;
        }
        else{
            common += cpu0;
            cpu0 = 0;
            cpu1 = 0;
        }
        if(common > gpu){
            allcom += gpu;
            common -= gpu;
            gpu = 0;
        }
        else if(common < gpu){
            allcom += common;
            gpu -= common;
            common = 0;
        }
        else{
            allcom += common;
            common = 0;
            gpu = 0;
        }
        cout<<sel<<" "<<pos[sel]<<" cpu0:"<<cpu0<<" cpu1:"<<cpu1<<" common:"<<common<<" gpu:"<<gpu<<" allcom:"<<allcom<<" gmin:"<<gmin<<endl;
        return gmin;
	}
public:
	void TaskDispatch(){
		int n;
        int a,b,c,d;

        cpu0 = cpu1 = common = gpu = allcom = 0;

        cin>>n;
        for(int i=0;i<n;i++){
            cin>>a>>b>>c>>d;
            T.push_back(Task(a,b,c,d));
            visited.push_back(false);
        }
        int ans[n];
        for(int i=0;i<n;i++)
            ans[i] = GetaMin();
		cout<<ans[n-1]<<endl;
	}
};
int main()
{
	Solution s;
	s.TaskDispatch();
	return 0;
}
