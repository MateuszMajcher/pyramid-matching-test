// Copyright 2007, Massachusetts Institute of Technology.
// The use of this code is permitted for research only. There is
// absolutely no warranty for this software.
//
// Author: John Lee (jjl@mit.edu)
//

#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include "point_set/mutable-point-set-list.h"
#include "point_set/point-ref.h"
#include "clustering/hierarchical-clusterer.h"
#include "util/distance-computer.h"
#include "pyramids/pyramid-matcher.h"
#include "pyramids/input-specific-vg-pyramid-maker.h"
#include "pyramids/input-specific-vg-pyramid-maker.h"
#include "histograms/multi-resolution-histogram.h"
#include "experiment/svm-experiment.h"
#include "experiment/eth-selector.h"

using namespace std;
using namespace libpmk;
using namespace libpmk_util;

void usage(const char* exec_name) {
   cerr << "Usage: " << exec_name << " input.kern output.hc levels branch\n\n";
   cerr << "<input.psl>: A PointSetList file, where each PointSet contains\n"
        << "             features for one image\n";
   cerr << "<levels>:    (int) Number of levels in the cluster tree\n";
   cerr << "<branch>:    (int) The branch factor of the cluster tree\n";
   cerr << "<labels.txt>: Label file\n";
   cerr << "<class size>: class size\n";
   cerr << "<test size>: test size\n";
}

vector<int> ReadLabels(string filename) {
   FILE* f = fopen(filename.c_str(), "r");
   int rows;
   fscanf(f, "%d", &rows);
   
   vector<int> labels;
   for (int ii = 0; ii < rows; ++ii) {
      int value;
      fscanf(f, "%d", &value);
      labels.push_back(value);
   }
   return labels;
}


int main(int argc, char** argv) {
   if (argc < 5) {
      usage(argv[0]);
      exit(0);
   }
   

   // Set the random seed for hierarchical clustering
   srand(time(NULL));

   string input_file(argv[1]);
   string label_file(argv[2]);

   int classes = atoi(argv[3]);
   int test_size = atoi(argv[4]);
   std::string log_file(argv[5]);

   cout<<"input file: "<<input_file<<endl;
   cout<<"label file: "<<label_file<<endl;
   cout<<"classes: "<<classes<<endl;
   cout<<"test_size: "<<test_size<<endl;
   cout<<"log: "<<log_file<<endl;
   
   //utworzenie pliku logów
   std::ofstream log(log_file.c_str(), std::ios_base::app | std::ios_base::out);
	
   KernelMatrix km;
   km.ReadFromFile(input_file.c_str());

   std::cout<<"liczba plików psl : "<<km.size()<<std::endl; 
   vector<int> labels = ReadLabels(label_file);
   double sum = 0;
   std::cout<<"liczba plików psl : "<<labels.size()<<std::endl; 

   // There are 80 different experiments to run; this is what the
   // which_test variable iterates over. There are 400 images total;
   // one run of this experiment takes 5 examples as testing and the
   // rest as training.
   for (int which_test = 0; which_test < 1; ++which_test) {

      ETHSelector selector(labels, which_test, classes, test_size);
      const vector<LabeledIndex>& examples(selector.GetTestingExamples());

      std::cout<<"size test:  "<<examples.size()<<std::endl; 
      std::cout<<"size training:  "<<selector.GetTrainingExamples().size()<<std::endl; 
   	

     SVMExperiment experiment(selector.GetTrainingExamples(),
                               selector.GetTestingExamples(),
                               km, 10000);
      clock_t time_start = clock();
      experiment.Train();
     clock_t time_end = clock();
      experiment.Test();

      double row_time = (double)(time_end - time_start) / CLOCKS_PER_SEC;
      cout.precision(6);
      cout << "time testing \t" << fixed << row_time << " seconds: \t"<< endl;
      
      // Also output what the SVM predicted for these 5 test examples.
      sum += (double)experiment.GetNumCorrect() /
         experiment.GetNumTestExamples();


//log file
	log<<"total file:"<<labels.size()<<"\n";
	log<<"classes:"<<classes<<"\n";
	log<<"test percent:"<<test_size<<"\n";
	log<<"size train:"<<selector.GetTrainingExamples().size()<<"\n";
	log<<"size test:"<<examples.size()<<"\n";
	log<<"time test:"<<row_time<<"\n";
	log<<"accuracy:"<<sum<<"\n";
	


   }

   // Report the average over all 80 runs.
   printf("Average accuracy: %f\n", sum / 1);
   
   
   return 0;
}
