#include "stdio.h"
#include "mpi.h"
#include "stdlib.h"
#include "Matrix.h"
#include <chrono>
#include <iostream>
#include <cassert>
#include <vector>

#define STRASSEN_PARALEL 0


const static uint32_t _N = 128;

struct MatrixSubdata
{
	double row;
	double coll;
};
const int sizes[] = { 1 , 1};
const MPI_Aint displacments[] =
{
	offsetof(MatrixSubdata, row),
	offsetof(MatrixSubdata, coll)
};
const MPI_Datatype types[] =
{
	MPI_DOUBLE,
	MPI_DOUBLE,
};
MPI_Datatype mpiType_MatrixSubdata;

void masterInit(int procRank, int procNum, const Matrix& matrixA, const Matrix& matrixB);
void slaveInit(int procRank, int procNum);
void masterFinalize(int procRank, int procNum);
void slaveFinalize(int procRank, int procNum);
void calculation(int procRank, int procNum);
void takeJob(int jobAmount, MatrixSubdata* subdata, int& num);
void addResults(double* newResults, int newResultsSize);
void printSubData(const MatrixSubdata& d);

Matrix MatrixA = Matrix(_N, _N);
Matrix MatrixB = Matrix(_N, _N);
MatrixSubdata* calculationData;
double* results;
int jobAmount;

int main(int argc, char* argv[]) {
	MPI_Status status;
	int procNum;
	int procRank;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
	MPI_Comm_size(MPI_COMM_WORLD, &procNum);

	if (procNum != 8)
	{
		printf("only 8 pocesses supported");
		MPI_Finalize();
		return 0;		
	}

	MPI_Type_create_struct(
		2,
		sizes,
		displacments,
		types,
		&mpiType_MatrixSubdata
	);
	MPI_Type_commit(&mpiType_MatrixSubdata);
	
	Matrix matrixA = randomMatrix(_N, _N, 0, 10);

	Matrix matrixB = randomMatrix(_N, _N, 0, 10);

	if (procRank == 0)
	{
		//printf(to_string(matrixA).c_str());
		//printf(to_string(matrixB).c_str());
		auto tStart = std::chrono::high_resolution_clock::now();
		Matrix m3(matrixA * matrixB);
		//printf(to_string(m3).c_str());
		auto tEnd = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart).count();
		std::cout << "duration of naive matrix multiplication = " << duration << " microseconds" << std::endl;
	}


	auto tStart = std::chrono::high_resolution_clock::now();
	
	switch (procRank)
	{
	case 0:
		masterInit(procRank, procNum, matrixA, matrixB);
		calculation(procRank, procNum);
		masterFinalize(procRank, procNum);
		break;
	default:
		slaveInit(procRank, procNum);
		calculation(procRank, procNum);
		slaveFinalize(procRank, procNum);
		break;
	}

	auto tEnd = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart).count();
	if (procRank == 0)
	{
		std::cout << "duration of parralel matrix multiplication = " << duration << " microseconds" << std::endl;
	}
	MPI_Type_free(&mpiType_MatrixSubdata);




	MPI_Finalize();
	return 0;
}

void masterInit(int procRank, int procNum, const Matrix& matrixA, const Matrix& matrixB)
{
	const int rightRank = 1;
	const int leftRank = procNum / 2;
	MatrixA = matrixA;
	MatrixB = matrixB;


	const int leftSize = _N * _N / 2;
	MatrixSubdata* leftSubdata = new MatrixSubdata[leftSize];
	size_t t = 0;
	for (size_t i = 0; i < _N; i++)
	{
		for (size_t j = 0; j < _N / 2; j++)
		{
			leftSubdata[t].row = i;
			leftSubdata[t].coll = j;
			t++;
		}
	}
	MPI_Send(matrixA.data(), _N* _N, MPI_DOUBLE, leftRank, 0, MPI_COMM_WORLD);
	MPI_Send(matrixB.data(), _N* _N, MPI_DOUBLE, leftRank, 0, MPI_COMM_WORLD);
	MPI_Send(&leftSize, 1, MPI_INT, leftRank, 0, MPI_COMM_WORLD);
	MPI_Send(leftSubdata, leftSize, mpiType_MatrixSubdata, leftRank, 0, MPI_COMM_WORLD);
	
	int rightSize = _N * _N / 2;
	MatrixSubdata* rightSubdata = new MatrixSubdata[rightSize];
	t = 0;
	for (size_t i = 0; i < _N; i++)
	{
		for (size_t j = _N / 2; j < _N; j++)
		{
			rightSubdata[t].row = i;
			rightSubdata[t].coll = j;
			t++;
		}
	}

	jobAmount = (_N * _N) / procNum;
	takeJob(jobAmount, rightSubdata, rightSize);

	rightSize -= jobAmount;
	MPI_Send(matrixA.data(), _N * _N, MPI_DOUBLE, rightRank, 0, MPI_COMM_WORLD);
	MPI_Send(matrixB.data(), _N * _N, MPI_DOUBLE, rightRank, 0, MPI_COMM_WORLD);
	MPI_Send(&rightSize, 1, MPI_INT, rightRank, 0, MPI_COMM_WORLD);
	MPI_Send(rightSubdata + jobAmount, rightSize, mpiType_MatrixSubdata, rightRank, 0, MPI_COMM_WORLD);
	delete[] leftSubdata;
	delete[] rightSubdata;
}

void masterFinalize(int procRank, int procNum)
{
	const int rightRank = 1;
	const int leftRank = procNum / 2;
	MPI_Status mpiStatus;

	int rightResultsSize;
	MPI_Recv(&rightResultsSize, 1, MPI_INT, rightRank, 0, MPI_COMM_WORLD, &mpiStatus);
	double* rightChildResults = new double[rightResultsSize];
	MPI_Recv(rightChildResults, rightResultsSize, MPI_DOUBLE, rightRank, 0, MPI_COMM_WORLD, &mpiStatus);

	addResults(rightChildResults, rightResultsSize);
	delete[] rightChildResults;

	int leftResultsSize;
	MPI_Recv(&leftResultsSize, 1, MPI_INT, leftRank, 0, MPI_COMM_WORLD, &mpiStatus);
	double* leftChildResults = new double[leftResultsSize];
	MPI_Recv(leftChildResults, leftResultsSize, MPI_DOUBLE, leftRank, 0, MPI_COMM_WORLD, &mpiStatus);

	addResults(leftChildResults, leftResultsSize);
	delete[] leftChildResults;

	Matrix resultMat = randomMatrix(_N, _N, 0, 0);

	int coll = _N / 2;
	int row = 0;
	for (size_t i = 0; i < jobAmount; i++)
	{
		resultMat(row, coll) = results[i];

		coll += 1;
		if (i < jobAmount / 2)
		{
			if (coll >= _N)
			{
				row += 1;
				coll = _N / 2;
			}
			if (row >= _N)
			{
				row = 0;
				coll = 0;
			}
		}
		else
		{
			if (coll >= _N / 2)
			{
				row += 1;
				coll = 0;
			}
			if (row >= _N)
				row = 0;
		}
	}

	//printf(to_string(resultMat).c_str());
}

void slaveInit(int procRank, int procNum)
{
	const bool isLeaf = procRank == (procNum - 1) || procRank == (procNum / 2 - 1);
	const int parent = procRank == procNum / 2 ? 0 : procRank - 1;

	MPI_Status mpiStatus;
	MPI_Recv(MatrixA.data(), _N * _N, MPI_DOUBLE, parent, 0, MPI_COMM_WORLD, &mpiStatus);
	MPI_Recv(MatrixB.data(), _N * _N, MPI_DOUBLE, parent, 0, MPI_COMM_WORLD, &mpiStatus);
	int subdataSize;
	MPI_Recv(&subdataSize, 1, MPI_INT, parent, 0, MPI_COMM_WORLD, &mpiStatus);
	MatrixSubdata* subdata = new MatrixSubdata[subdataSize];
	MPI_Recv(subdata, subdataSize, mpiType_MatrixSubdata, parent, 0, MPI_COMM_WORLD, &mpiStatus);

	jobAmount = isLeaf? subdataSize: jobAmount = (_N * _N) / procNum;
	takeJob(jobAmount, subdata, subdataSize);

	if (!isLeaf)
	{
		subdataSize -= jobAmount;
		MPI_Send(MatrixA.data(), _N * _N, MPI_DOUBLE, procRank + 1, 0, MPI_COMM_WORLD);
		MPI_Send(MatrixB.data(), _N * _N, MPI_DOUBLE, procRank + 1, 0, MPI_COMM_WORLD);
		MPI_Send(&subdataSize, 1, MPI_INT, procRank + 1, 0, MPI_COMM_WORLD);
		MPI_Send(subdata + jobAmount, subdataSize, mpiType_MatrixSubdata, procRank + 1, 0, MPI_COMM_WORLD);
	}
	delete[] subdata;
}

void slaveFinalize(int procRank, int procNum)
{
	const bool isLeaf = procRank == (procNum - 1) || procRank == (procNum / 2 - 1);
	const int parent = procRank == procNum / 2 ? 0 : procRank - 1;

	MPI_Status mpiStatus;

	if (!isLeaf)
	{
		int resultsSize;
		MPI_Recv(&resultsSize, 1, MPI_INT, procRank + 1, 0, MPI_COMM_WORLD, &mpiStatus);
		double *childResults = new double[resultsSize];
		MPI_Recv(childResults, resultsSize, MPI_DOUBLE, procRank + 1, 0, MPI_COMM_WORLD, &mpiStatus);
		addResults(childResults, resultsSize);
		delete[] childResults;
	}

	MPI_Send(&jobAmount, 1, MPI_INT, parent, 0, MPI_COMM_WORLD);
	MPI_Send(results, jobAmount, MPI_DOUBLE, parent, 0, MPI_COMM_WORLD);
	delete[] results;
}

void calculation(int procRank, int procNum)
{
	results = new double[jobAmount];
	for (size_t job = 0; job < jobAmount; job++)
	{
		MatrixSubdata jobData = calculationData[job];
		double result = 0;
		for (int i = 0; i < _N; i++)
		{
			result += MatrixA(jobData.row, i) * MatrixB(i, jobData.coll);
		}
		results[job] = result;
	}
	delete[] calculationData;
}

void takeJob(int jobAmount, MatrixSubdata* subdata, int& num)
{
	calculationData = new MatrixSubdata[jobAmount];
	memcpy(calculationData, subdata, jobAmount * sizeof(MatrixSubdata));
}

void addResults(double* newResults, int newResultsSize)
{
	double* unitedResults = new double[jobAmount + newResultsSize];
	memcpy(unitedResults, results, jobAmount * sizeof(double));
	memcpy(unitedResults + jobAmount, newResults, newResultsSize * sizeof(double));

	delete[] results;
	results = unitedResults;
	jobAmount = jobAmount + newResultsSize;
}

void printSubData(const MatrixSubdata& d)
{
	/*printf("rows: ");
	for (int i = 0; i < _N; i++)
	{
		printf("%f ", d.row[i]);
	}
	printf("\ncolls: ");
	for (int i = 0; i < _N; i++)
	{
		printf("%f ", d.coll[i]);
	}
	printf("\n");*/
}