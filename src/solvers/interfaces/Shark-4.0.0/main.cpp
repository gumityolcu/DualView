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


template <typename T>
std::vector<T> readFileData(const std::string& name) {
    std::ifstream inputFile(name, std::ios_base::binary);

    // Determine the length of the file by seeking
    // to the end of the file, reading the value of the
    // position indicator, and then seeking back to the beginning.
    inputFile.seekg(0, std::ios_base::end);
    auto length = inputFile.tellg();
    inputFile.seekg(0, std::ios_base::beg);
    T temp;
    // Make a buffer of the exact size of the file and read the data into it.
    std::vector<T> buffer(length/sizeof(temp));
    inputFile.read(reinterpret_cast<char*>(buffer.data()), length);

    inputFile.close();
    return buffer;
}

std::vector < std::pair < unsigned int, std::vector < float>> > read_files(std::string& path) {
    if (path[path.length() - 1] == '/') {
        path = path.substr(0, path.length() - 1);
    }
    std::vector<float> data = readFileData<float>(path + "/data.shark");
    std::vector<unsigned int> labels = readFileData<unsigned int>(path + "/labels.shark");
    int num_of_features = (int) labels[0];
    int num_of_datapoints= (int) labels.size()-1;
    std::vector<std::pair<unsigned int, std::vector <float>>> ret(labels.size()-1);
    for(int i=0;i<num_of_datapoints;i++)
    {
        ret[i].first=labels[i+1];
        for(int j=0;j<num_of_features;j++)
        {
            ret[i].second.push_back(data[i*num_of_features+j]);
        }
    }
    return ret;
}

void read_dataset(std::string& path, LabeledData<RealVector, unsigned int> &dataset) {
    std::vector < std::pair < unsigned int, std::vector < float>> > rows = read_files(path);
    if (rows.empty()) {//empty file leads to empty data object.
        dataset = LabeledData<RealVector, unsigned int>();
        return ;
    }

    //check labels for conformity
    bool binaryLabels = false;
    int minPositiveLabel = std::numeric_limits<int>::max();
    {

        int maxPositiveLabel = -1;
        for (std::size_t i = 0; i != rows.size(); ++i) {
            int label = rows[i].first;
            SHARK_RUNTIME_CHECK(label >= -1, "labels can not be smaller than -1");
            if (label == -1)
                binaryLabels = true;
            else if (label < minPositiveLabel)
                minPositiveLabel = label;
            else if (label > maxPositiveLabel)
                maxPositiveLabel = label;
        }
        SHARK_RUNTIME_CHECK(
                minPositiveLabel >= 0 || (minPositiveLabel == -1 && maxPositiveLabel == 1),
                "negative labels are only allowed for classes -1/1"
        );
    }

    //copy rows of the file into the dataset
    std::size_t dimensions = rows[0].second.size();
    std::vector <std::size_t> batchSizes = shark::detail::optimalBatchSizes(rows.size(), LabeledData<RealVector, unsigned int>::DefaultBatchSize);
    dataset = LabeledData<RealVector, unsigned int>(batchSizes.size());
    std::size_t currentRow = 0;
    for (std::size_t b = 0; b != batchSizes.size(); ++b) {
        RealMatrix &inputs = dataset.batch(b).input;
        UIntVector &labels = dataset.batch(b).label;
        inputs.resize(batchSizes[b], dimensions);
        labels.resize(batchSizes[b]);
        //copy the rows into the batch
        for (std::size_t i = 0; i != batchSizes[b]; ++i, ++currentRow) {
            SHARK_RUNTIME_CHECK(rows[currentRow].second.size() == dimensions, "Vectors are required to have same size");

            for (std::size_t j = 0; j != dimensions; ++j) {
                inputs(i, j) = rows[currentRow].second[j];
            }
            int rawLabel = rows[currentRow].first;
            labels[i] = binaryLabels ? 1 + (rawLabel - 1) / 2 : rawLabel - minPositiveLabel;
        }
    }
    dataset.inputs().shape() = {dimensions};
    SIZE_CHECK(currentRow == rows.size());
}

int main(int argc, char *argv[]) {
    std::cout<<"ENTERED SHARK"<<std::endl;
    std::string path(argv[1]);
    LabeledData<VectorType, unsigned int> training;
    read_dataset(path, training);

    float C = atof(argv[2]);
    std::cout << "C : " << C << std::endl;
    if (path[path.length() - 1] == '/') {
        path = path.substr(0, path.length() - 1);
    }
    read_dataset(path, training);
    LinearClassifier <VectorType> model;
    LinearCSvmTrainer <VectorType> trainer(C, false);
    trainer.setMcSvmType(shark::McSvm::CS);
    std::pair <RealMatrix, RealMatrix> ret = trainer.train_custom(model, training);
    /*ZeroOneLoss<unsigned int> loss;
    Data<unsigned int> output = model(training.inputs());
    double train_error = loss.eval(training.labels(), output);
    std::cout << "training error:\t" <<  train_error << std::endl;
    std::cout << ret.second << std::endl;
    std::cout << ret.second(0, 0) << std::endl;
    */
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
    str.open(path + "/weights.csv");
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
