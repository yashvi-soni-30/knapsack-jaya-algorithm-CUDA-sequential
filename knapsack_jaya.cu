%%writefile knapsack_jaya.cu
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <math.h>
#include <curand_kernel.h>
using namespace std;

#define DIM 200
#define NUM_PARTICLES 1000
#define NUM_ITR 100
#define MUTATION_RATE 0.03f
#define CAPACITY 28

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void initCurand(curandState* states, unsigned long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, tid, 0, &states[tid]);
}

__global__ void jayaUpdateKernel(
    int* d_population, int* d_newPopulation,
    int* d_best, int* d_worst,
    float mutationRate, int numParticles, int dim,
    curandState* states
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    curandState localState = states[i];
    int idx = i * dim;

    for (int j = 0; j < dim; j++) {
        int x = d_population[idx + j];
        int x_best = d_best[j];
        int x_worst = d_worst[j];

        float r1 = curand_uniform(&localState);
        float r2 = curand_uniform(&localState);

        float v = r1 * (x_best - x) - r2 * (x_worst - x);
        float p = sigmoid(v);

        float randVal = curand_uniform(&localState);
        d_newPopulation[idx + j] = (randVal < p) ? 1 : 0;

        // Mutation
        float mut = curand_uniform(&localState);
        if (mut < mutationRate)
            d_newPopulation[idx + j] = 1 - d_newPopulation[idx + j];
    }

    states[i] = localState;  // Save state back
}

float objectiveFunc(const vector<int>& solution, const vector<int>& weights, const vector<int>& values, int capacity) {
    int totalWeight = 0, totalValue = 0;
    for (int i = 0; i < solution.size(); i++) {
        if (solution[i]) {
            totalWeight += weights[i];
            totalValue += values[i];
        }
    }
    if (totalWeight > capacity) return 0;
    return totalValue;
}

int main() {


    srand(time(0));
    vector<int> weights(DIM), values(DIM);
    for (int i = 0; i < DIM; i++) {
        weights[i] = rand() % 10 + 1;
        values[i] = rand() % 20 + 1;
    }

    cout << "Items (Index | Weight | Value):\n";
    for (int i = 0; i < DIM; i++) {
        cout << "Item " << i << ": " << weights[i] << " | " << values[i] << "\n";
    }
    cout << "Knapsack Capacity: " << CAPACITY << "\n\n";

    vector<vector<int>> population(NUM_PARTICLES, vector<int>(DIM));
    for (int i = 0; i < NUM_PARTICLES; i++)
        for (int j = 0; j < DIM; j++)
            population[i][j] = rand() % 2;

    int* d_population;
    int* d_newPopulation;
    int* d_best;
    int* d_worst;
    curandState* d_states;

    cudaMalloc(&d_population, NUM_PARTICLES * DIM * sizeof(int));
    cudaMalloc(&d_newPopulation, NUM_PARTICLES * DIM * sizeof(int));
    cudaMalloc(&d_best, DIM * sizeof(int));
    cudaMalloc(&d_worst, DIM * sizeof(int));
    cudaMalloc(&d_states, NUM_PARTICLES * sizeof(curandState));

    int blockSize = 128;
    int gridSize = (NUM_PARTICLES + blockSize - 1) / blockSize;
    initCurand<<<gridSize, blockSize>>>(d_states, time(0));
    cudaDeviceSynchronize();

    for (int itr = 0; itr < NUM_ITR; itr++) {
        vector<int> best = population[0], worst = population[0];
        float bestFit = objectiveFunc(best, weights, values, CAPACITY);
        float worstFit = bestFit;

        for (const auto& sol : population) {
            float fit = objectiveFunc(sol, weights, values, CAPACITY);
            if (fit > bestFit) {
                bestFit = fit;
                best = sol;
            }
            if (fit < worstFit) {
                worstFit = fit;
                worst = sol;
            }
        }

        vector<int> flatPop;
        for (auto& sol : population)
            flatPop.insert(flatPop.end(), sol.begin(), sol.end());

        cudaMemcpy(d_population, flatPop.data(), NUM_PARTICLES * DIM * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_best, best.data(), DIM * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_worst, worst.data(), DIM * sizeof(int), cudaMemcpyHostToDevice);

        jayaUpdateKernel<<<gridSize, blockSize>>>(
            d_population, d_newPopulation, d_best, d_worst,
            MUTATION_RATE, NUM_PARTICLES, DIM, d_states);
        cudaDeviceSynchronize();

        vector<int> updatedPop(NUM_PARTICLES * DIM);
        cudaMemcpy(updatedPop.data(), d_newPopulation, NUM_PARTICLES * DIM * sizeof(int), cudaMemcpyDeviceToHost);

        for (int i = 0; i < NUM_PARTICLES; i++) {
            vector<int> newSol(updatedPop.begin() + i * DIM, updatedPop.begin() + (i + 1) * DIM);
            if (objectiveFunc(newSol, weights, values, CAPACITY) > objectiveFunc(population[i], weights, values, CAPACITY))
                population[i] = newSol;
        }

        cout << "Iteration " << itr + 1 << ": Best Fitness = " << bestFit << ", Worst Fitness = " << worstFit << "\n";
    }

    // Final best solution
    vector<int> best = population[0];
    float bestFit = objectiveFunc(best, weights, values, CAPACITY);
    for (auto& sol : population) {
        float fit = objectiveFunc(sol, weights, values, CAPACITY);
        if (fit > bestFit) {
            bestFit = fit;
            best = sol;
        }
    }

    cout << "\n Best Solution Found:\n";
    int totalWeight = 0, totalValue = 0;
    for (int i = 0; i < DIM; i++) {
        if (best[i]) {
            cout << "Item " << i << " selected (Weight: " << weights[i]
                 << ", Value: " << values[i] << ")\n";
            totalWeight += weights[i];
            totalValue += values[i];
        }
    }
    cout << "\nTotal Weight = " << totalWeight
         << "\nTotal Value  = " << totalValue << endl;

    cudaFree(d_population);
    cudaFree(d_newPopulation);
    cudaFree(d_best);
    cudaFree(d_worst);
    cudaFree(d_states);

    return 0;
}
