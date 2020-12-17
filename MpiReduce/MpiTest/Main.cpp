#include "stdio.h"
#include "mpi.h"
#include "stdlib.h"
#include "math.h"
#include <random>
#include <iostream>

inline double randomDouble() {
	static std::uniform_real_distribution<double> distribution(0.0, 1.0);
	static std::mt19937 generator;
	return distribution(generator);
}

inline double randomDouble(double min, double max) {
	// Returns a random real in [min,max).
	return min + (max - min) * randomDouble();
}

enum ReduceDir
{
	Vertical,
	Horisontal
};

int main(int argc, char* argv[]) {
	MPI_Status status;
	int procNum;
	int procRank;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
	MPI_Comm_size(MPI_COMM_WORLD, &procNum);

	double *data = new double[procNum];
	double sum = 0;
	for (int i = 0; i < procNum; i++)
	{
		data[i] = randomDouble(-100, 100);
		sum += data[i];
		if (procRank == 0)
		{
			printf("initial data %lf \n", data[i]);
		}
	}
	if (procRank == 0)
	{
		printf("expected sum %lf \n", sum);
	}

	int netWidth = sqrt(procNum);
	int netHeight = sqrt(procNum);

	ReduceDir reduceDir = ReduceDir::Vertical;

	int currRow = netWidth - 1;
	int currColl = netHeight - 1;
	int n = netWidth + netHeight - 2;
	for (int i = 0; i < n; i++)
	{
		if(procRank% netWidth > currColl || procRank / netWidth > currRow)
			continue;
			
		bool shouldSend;
		if (reduceDir == ReduceDir::Vertical)
			shouldSend = procRank % netWidth == currColl;
		else
			shouldSend = procRank / netWidth == currRow;

		bool shouldResv;
		if (reduceDir == ReduceDir::Vertical)
			shouldResv = procRank % netWidth == currColl - 1;
		else
			shouldResv = procRank / netWidth == currRow - 1;

		if (shouldSend) 
		{
			if (reduceDir == ReduceDir::Vertical)
			{
				int dest = procRank - 1;
				MPI_Send(data + procRank, 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
			}
			else
			{
				int dest = procRank - netWidth;
				MPI_Send(data + procRank, 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
			}
		}

		//printf("proc %i iter %i shouldSend %i shouldResv %i\n", procRank, i, shouldSend, shouldResv);
		if (shouldResv)
		{
			if (reduceDir == ReduceDir::Vertical)
			{
				int src = procRank + 1;
				double tmpData;
				MPI_Recv(&tmpData, 1, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, &status);
				data[procRank] += tmpData;
				
			}
			else
			{
				int src = procRank + netWidth;
				double tmpData;
				MPI_Recv(&tmpData, 1, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, &status);
				data[procRank] += tmpData;
			}
		}

		if(reduceDir == ReduceDir::Vertical)
			currColl -= 1;
		else
			currRow -= 1;
		reduceDir = reduceDir == ReduceDir::Vertical ? ReduceDir::Horisontal : ReduceDir::Vertical;
	}

	if (procRank == 0)
	{
		printf("actual sum %lf \n", data[0]);
	}

	MPI_Finalize();
}