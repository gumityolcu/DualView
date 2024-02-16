#include <iostream>
#include <fstream>
#include <string>
#include <shark/Data/Dataset.h>
#include <shark/Data/Csv.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
//#include <boost/operators.hpp>
using namespace shark;
typedef RealVector VectorType;

int main(int argc, char *argv[]) {
    LabeledData<VectorType, unsigned int> training;
    std::string path(argv[1]);
    float C=atof(argv[2]);
    std::cout<<"C : "<<C<<std::endl;
    if (path[path.length() - 1] == '/') {
        path = path.substr(0, path.length() - 1);
    }
    importCSV(training, path + "/data.csv", LAST_COLUMN, ',', '#');

    LinearClassifier<VectorType> model;
    LinearCSvmTrainer<VectorType> trainer(C, false);
    trainer.setMcSvmType(shark::McSvm::CS);
    std::pair<RealMatrix, RealMatrix> ret = trainer.train_custom(model, training);
    ZeroOneLoss<unsigned int> loss;
    //Data<unsigned int> output = model(training.inputs());
    //double train_error = loss.eval(training.labels(), output);
    //std::cout << "training error:\t" <<  train_error << std::endl;
    std::cout << ret.second << std::endl;
    std::cout << ret.second(0, 0) << std::endl;
    std::ofstream str;
    str.open(path + "/dualvars.csv");
    for (int i = 0; i < ret.first.size1(); i++) {
        for (int j = 0; j < ret.first.size2(); j++) {
            str << ret.first(i, j);
            if (j < ret.first.size2() - 1)
                str << ", ";
        }
        str << "\n";
    }
    str.close();
    str.open(path+"/weights.csv");
    for (int i = 0; i < ret.second.size1(); i++) {
        for (int j = 0; j < ret.second.size2(); j++) {
            str << ret.second(i, j);
            if (j < ret.second.size2() - 1)
                str << ", ";
        }
        str << "\n";

    }

    return 0;
}
