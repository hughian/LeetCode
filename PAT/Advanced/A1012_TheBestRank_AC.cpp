#include<iostream>
#include<vector>
using namespace std;
char prior[]={'A','C','M','E'};
int main()
{
    int N,M;
    vector<int> vid,vc,vm,ve,va;
    int rank[4];
    cin>>N>>M;
    vid.resize(N);
    vc.resize(N);
    vm.resize(N);
    ve.resize(N);
    va.resize(N);
    for(int i=0;i<N;i++){
        cin>>vid[i]>>vc[i]>>vm[i]>>ve[i];
        va[i] = (vc[i]+vm[i]+ve[i])/3;
    }
    
    int index;
    int id;
    for(int i=0;i<M;i++){
        cin>>id;
        index = -1;
        for(int j=0;j<N;j++){
            if(id == vid[j]){
                index = j;break;
            }
        }

        if(index == -1){
            cout<<"N/A"<<endl;
        }else{
            for(int j=0;j<4;j++)
                rank[j] = 1;

            for(int j=0;j<N;j++){
                if(j != index){
                    if(va[j] > va[index])
                        rank[0]++;
                    if(vc[j] > vc[index])
                        rank[1]++;
                    if(vm[j] > vm[index])
                        rank[2]++;
                    if(ve[j] > ve[index])
                        rank[3]++;
                }
            }

            int min = rank[0];
            int tag = 0;
            for(int j=1;j<4;j++){
                if(min > rank[j]){
                    min = rank[j];
                    tag = j;
                }
            }
            cout<<min<<" "<<prior[tag]<<endl;
        }
    }
}
