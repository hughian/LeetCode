#include<iostream>
#include<vector>

using namespace std;
typedef struct Row{
    int seat[5];
    int available;
    Row(){
        for(int i=0;i<5;i++)
            seat[i] = 0;
        available = 5;
    }
    bool book(int nums,int j,vector<int>& t){
        if(available < nums)
            return false;
        available -= nums;
        int i = 0,k=0;
        int first = 5*j + 1;
        while(i<5 && k<nums){
            if(seat[i]==0){
                seat[i] = 1;
                k++;
                t.push_back(first+i);
            }
			i++;
        }
    }
}ROW;
void print(vector<int> &vec){
    for(int i=0;i<(int)vec.size();i++)
        cout<<vec.at(i)<<" ";
    vec.clear();
}
int main(void)
{
    int n,tmp,flg = 0;
    vector<int> v;
    vector<ROW> ticket(20);
    cin>>n;
    for(int i =0;i<n;i++){
        cin>>tmp;
        v.push_back(tmp);
    }
    vector<int> t;
    for(int i=0;i<n;i++){
        tmp = v.at(i);//购票指令
        flg = 0;
		for(int j=0;j<20;j++){ //找能够放到一排
            if(ticket.at(j).available >= tmp){
                ticket.at(j).book(tmp,j,t);
                print(t);
				cout<<endl;
                flg = 1;break;
            }
        }
        if(flg) continue;
        else{//不能放到一排
            for(int j=0;j<20;j++){
                if(ticket.at(j).available > 0 &&  (tmp -= ticket.at(j).available) > 0){
                    ticket.at(j).book(ticket.at(j).available,j,t);
                    print(t);
                }
                if(ticket.at(j).available > 0){
                    ticket.at(j).book(tmp,j,t);
                    print(t);
					cout<<endl;break;
                }
            }
        }
    }
    return 0;
}
