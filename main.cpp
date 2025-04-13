#include "config_reader.h"
#include "Bdtrain.h"
#include "branch_info.h"


int main() {

    Bdtrain bdtrain("config.yaml");
    bdtrain.Init();
    bdtrain.Save_root();  // need long time(20 mins for ～500000 events)
    bdtrain.Chain2Csv();  // need long time
    bdtrain.Training();

    return 0;
}
